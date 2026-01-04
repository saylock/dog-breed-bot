from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BreedResNet101(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 120,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        in_features = backbone.fc.in_features
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)

    @staticmethod
    def _acc(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        preds = torch.argmax(logits, dim=1)
        return (preds == y).float().mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self._acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self._acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
