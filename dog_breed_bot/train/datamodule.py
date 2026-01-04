from __future__ import annotations

from dataclasses import dataclass

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dog_breed_bot.data.dataset import DogsDataset, DogsDatasetConfig


@dataclass(frozen=True)
class DataModuleConfig:
    train_csv: str
    val_csv: str
    test_csv: str
    class_to_idx: str
    batch_size: int = 32
    num_workers: int = 2
    image_size: int = 224


class DogsDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataModuleConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = DogsDataset(
            DogsDatasetConfig(
                csv_path=self.cfg.train_csv,
                class_to_idx_path=self.cfg.class_to_idx,
                image_size=self.cfg.image_size,
            ),
            augment=True,
        )
        self.val_ds = DogsDataset(
            DogsDatasetConfig(
                csv_path=self.cfg.val_csv,
                class_to_idx_path=self.cfg.class_to_idx,
                image_size=self.cfg.image_size,
            ),
            augment=False,
        )
        self.test_ds = DogsDataset(
            DogsDatasetConfig(
                csv_path=self.cfg.test_csv,
                class_to_idx_path=self.cfg.class_to_idx,
                image_size=self.cfg.image_size,
            ),
            augment=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
