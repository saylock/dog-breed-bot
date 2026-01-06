from __future__ import annotations

from pathlib import Path

import torch

from dog_breed_bot.train.lightning_module import BreedResNet101


def export_onnx(
    ckpt_path: str = "artifacts/pt/best.ckpt",
    out_path: str = "artifacts/onnx/model.onnx",
    image_size: int = 224,
) -> str:
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = BreedResNet101.load_from_checkpoint(str(ckpt))
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy,
        str(out),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )

    return str(out)
