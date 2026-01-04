from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from dog_breed_bot.train.datamodule import DataModuleConfig, DogsDataModule
from dog_breed_bot.train.lightning_module import BreedResNet101


@dataclass(frozen=True)
class ModelCfg:
    num_classes: int = 120
    freeze_backbone: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class MlflowCfg:
    tracking_uri: str = "http://127.0.0.1:8080"
    experiment_name: str = "dog-breed-stanford"


@dataclass(frozen=True)
class TrainCfg:
    seed: int = 42
    epochs: int = 15
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    output_dir: str = "artifacts/pt"
    model: ModelCfg = ModelCfg()
    data: DataModuleConfig | None = None
    mlflow: MlflowCfg = MlflowCfg()


def _configs_dir() -> str:
    # .../dog_breed_bot/train/train.py -> repo_root/configs
    return str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base=None, config_path=_configs_dir(), config_name="train")
def main(cfg: TrainCfg) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("CONFIG:\n", OmegaConf.to_yaml(OmegaConf.structured(cfg)))

    dm = DogsDataModule(cfg.data)  # type: ignore[arg-type]
    model = BreedResNet101(
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        freeze_backbone=cfg.model.freeze_backbone,
    )

    ckpt = ModelCheckpoint(
        dirpath=str(out_dir),
        filename="best",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    mlf_logger = MLFlowLogger(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices="auto",
        logger=mlf_logger,
        callbacks=[ckpt],
        log_every_n_steps=20,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
    )

    trainer.fit(model, datamodule=dm)
    print(f"✅ Best checkpoint: {ckpt.best_model_path}")


if __name__ == "__main__":
    main()
