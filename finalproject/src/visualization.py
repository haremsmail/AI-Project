2"""Plotting helpers for the project report and notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
"""  amazha danuse bo ko krdnauay dauakaryakan"""

import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend so plots show on screen
import matplotlib.pyplot as plt
""" auayan bo drust krndy bar chartu auan  bakar de"""
import numpy as np
import pandas as pd
""" this used to for table result"""
import seaborn as sns
""" auayan bo graficky juantr bakar de """
from sklearn.metrics import confusion_matrix, roc_curve, auc
""" higher roc better model """
from sklearn.preprocessing import label_binarize
""" convert class daka bo binary matrix"""

from .config import RESULTS_DIR

sns.set_theme(style="whitegrid", palette="deep")
""" auayn wata graph white backraoundy habe """

""" aua save graphaka daua helper functiel dadane bo graphaka"""
def save_figure(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=180)
    return output_path
""" restulta  wardagre ba graph"""



def plot_class_distribution(dataframe: pd.DataFrame, label_column: str = "label", title: str = "Class Distribution", output_path: str | Path | None = None):
    figure, axis = plt.subplots(figsize=(10, 5))
    order = dataframe[label_column].value_counts().index
    sns.countplot(data=dataframe, x=label_column, order=order, ax=axis)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=45)
    axis.set_xlabel("Class")
    axis.set_ylabel("Image count")
    figure.tight_layout()
    if output_path:
        save_figure(output_path)
    plt.show()  
    return figure


def plot_sample_images(samples: Sequence[tuple[np.ndarray, str]], title: str = "Sample Images", output_path: str | Path | None = None):
    sample_count = min(len(samples), 9)
    figure, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    for index in range(9):
        axis = axes[index]
        axis.axis("off")
        if index < sample_count:
            image, label = samples[index]
            axis.imshow(image)
            axis.set_title(label)
    figure.suptitle(title)
    figure.tight_layout()
    if output_path:
        save_figure(output_path)
    plt.show()  # Show plot on screen
    return figure

""" bo pishan danay auay chand wena la har data setek da haya """
def plot_confusion_matrix(y_true, y_pred, class_names: Sequence[str], title: str = "Confusion Matrix", output_path: str | Path | None = None):
    matrix = confusion_matrix(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)
    figure.tight_layout()
    if output_path:
        save_figure(output_path)
    plt.show()  # Show plot on screen
    return figure


def plot_metric_comparison(results: pd.DataFrame, output_path: str | Path | None = None):
    metrics = [column for column in ["accuracy", "precision", "recall", "f1"] if column in results.columns]
    melted = results.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_title("Model Comparison - Accuracy, Precision, Recall, F1")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    if output_path:
        save_figure(output_path)
    plt.show()  # Show plot on screen
    return figure


def plot_training_curves(history, output_path: str | Path | None = None):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history.history["accuracy"], label="Train")
    if "val_accuracy" in history.history:
        axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    if "val_loss" in history.history:
        axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].legend()
    figure.tight_layout()
    if output_path:
        save_figure(output_path)
    plt.show()  # Show plot on screen
    return figure


def plot_roc_curves(y_true, probabilities, class_names: Sequence[str], title: str = "ROC Curves", output_path: str | Path | None = None):
    class_count = len(class_names)
    if probabilities is None:
        return None
    y_true_binarized = label_binarize(y_true, classes=np.arange(class_count))
    figure, axis = plt.subplots(figsize=(10, 8))
    for class_index, class_name in enumerate(class_names):
        if class_index >= probabilities.shape[1]:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true_binarized[:, class_index], probabilities[:, class_index])
        roc_auc = auc(false_positive_rate, true_positive_rate)
        axis.plot(false_positive_rate, true_positive_rate, label=f"{class_name} (AUC={roc_auc:.2f})")
    axis.plot([0, 1], [0, 1], "k--", alpha=0.6)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(title)
    axis.legend(loc="lower right")
    figure.tight_layout()
    if output_path:
        save_figure(output_path)
    plt.show()  
    return figure
