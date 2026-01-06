from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def _humanize_class(folder_name: str) -> str:
    # "n02085620-Chihuahua" -> "Chihuahua"
    if "-" in folder_name:
        name = folder_name.split("-", 1)[1]
    else:
        name = folder_name
    return name.replace("_", " ").strip()


def _load_idx_to_class(class_to_idx_path: str) -> dict[int, str]:
    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    return {int(v): str(k) for k, v in class_to_idx.items()}


def _preprocess(image_path: str, image_size: int = 224) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")

    # resize shorter side to 256 then center crop 224
    w, h = img.size
    if w < h:
        new_w = 256
        new_h = int(h * (256 / w))
    else:
        new_h = 256
        new_w = int(w * (256 / h))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - image_size) // 2
    top = (new_h - image_size) // 2
    img = img.crop((left, top, left + image_size, top + image_size))

    x = np.asarray(img).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    x = (x - mean) / std

    return np.expand_dims(x, axis=0)  # NCHW


def predict(
    image_path: str,
    onnx_path: str = "artifacts/onnx/model.onnx",
    class_to_idx_path: str = "artifacts/class_to_idx.json",
) -> str:
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_file}")

    idx_to_class = _load_idx_to_class(class_to_idx_path)

    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    x = _preprocess(image_path)

    logits = sess.run(["logits"], {"image": x})[0]
    pred_idx = int(np.argmax(logits, axis=1)[0])

    return _humanize_class(idx_to_class[pred_idx])
