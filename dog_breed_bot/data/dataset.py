from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class DogsDatasetConfig:
    csv_path: str
    class_to_idx_path: str
    image_size: int = 224


class DogsDataset(Dataset):
    def __init__(self, cfg: DogsDatasetConfig, augment: bool = False):
        self.df = pd.read_csv(cfg.csv_path)
        with open(cfg.class_to_idx_path, "r", encoding="utf-8") as f:
            self.class_to_idx = json.load(f)

        self.transform = self._build_transform(cfg.image_size, augment)

    @staticmethod
    def _build_transform(image_size: int, augment: bool) -> transforms.Compose:
        if augment:
            tfm = [
                transforms.Resize(256),
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        else:
            tfm = [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
            ]

        tfm += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        return transforms.Compose(tfm)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        label_name = row["class_name"]
        y = int(self.class_to_idx[label_name])

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, y
