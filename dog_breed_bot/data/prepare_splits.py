from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_manifest(images_dir: Path) -> pd.DataFrame:
    """
    Stanford Dogs format:
      data/raw/stanford_dogs/Images/<breed_folder>/*.jpg
    We keep <breed_folder> as the class label.
    """
    rows: list[dict[str, str]] = []
    for breed_dir in sorted(images_dir.iterdir()):
        if not breed_dir.is_dir():
            continue

        class_name = breed_dir.name
        for img_path in breed_dir.glob("*.jpg"):
            rows.append(
                {
                    "image_path": img_path.as_posix(),
                    "class_name": class_name,
                }
            )

    if not rows:
        raise RuntimeError(f"No images found under: {images_dir}")

    return pd.DataFrame(rows)


def make_balanced_splits(
    df: pd.DataFrame,
    val_per_class: int,
    test_per_class: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []
    test_parts = []

    for class_name, group in df.groupby("class_name"):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        need = val_per_class + test_per_class + 1
        if len(group) < need:
            msg = f"Class {class_name} has only {len(group)} images, " f"need >= {need}"
            raise ValueError(msg)

        val = group.iloc[:val_per_class]
        test = group.iloc[val_per_class : val_per_class + test_per_class]
        train = group.iloc[val_per_class + test_per_class :]

        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    train_df = pd.concat(train_parts, ignore_index=True).sample(
        frac=1.0, random_state=seed
    )
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    test_df = pd.concat(test_parts, ignore_index=True).sample(
        frac=1.0, random_state=seed
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def write_label_map(df: pd.DataFrame, out_path: Path) -> None:
    classes = sorted(df["class_name"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(class_to_idx, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    raw_root = Path("data/raw/stanford_dogs")

    images_dir = raw_root / "Images"
    if not images_dir.exists():
        images_dir = raw_root / "images"

    if not images_dir.exists():
        msg = (
            "Expected Images/images folder at: "
            f"{raw_root / 'Images'} or {raw_root / 'images'}"
        )
        raise FileNotFoundError(msg)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = build_manifest(images_dir)

    val_per_class = 10
    test_per_class = 10

    train_df, val_df, test_df = make_balanced_splits(
        df=df,
        val_per_class=val_per_class,
        test_per_class=test_per_class,
        seed=42,
    )

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    write_label_map(df, Path("artifacts/class_to_idx.json"))

    print("✅ Wrote:")
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    print(f" - {train_path}  (rows={len(train_df)})")
    print(f" - {val_path}    (rows={len(val_df)})")
    print(f" - {test_path}   (rows={len(test_df)})")
    print(" - artifacts/class_to_idx.json")


if __name__ == "__main__":
    main()
