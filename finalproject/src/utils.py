"""Utility helpers for data discovery, reproducibility, and splitting."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_global_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and TensorFlow seeds when available."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_images(root_dir: str | Path, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> pd.DataFrame:
    """Discover labeled images recursively.

    The label is inferred from the parent folder name of each image.
    """

    root = Path(root_dir)
    rows: list[dict[str, str]] = []
    for extension in extensions:
        for image_path in root.rglob(f"*{extension}"):
            if image_path.is_file():
                rows.append({"image_path": str(image_path), "label": image_path.parent.name})
    dataframe = pd.DataFrame(rows).drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return dataframe


def stratified_split(
    dataframe: pd.DataFrame,
    label_column: str = "label",
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
):
    """Create train/validation/test splits with stratification."""

    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        stratify=dataframe[label_column],
        random_state=random_state,
    )
    validation_fraction = validation_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=validation_fraction,
        stratify=train_val_df[label_column],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
