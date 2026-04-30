"""Model training and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize


@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "model": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


def train_knn(x_train: np.ndarray, y_train: np.ndarray, n_neighbors: int = 7) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    model.fit(x_train, y_train)
    return model


def train_gaussian_nb(x_train: np.ndarray, y_train: np.ndarray) -> GaussianNB:
    model = GaussianNB()
    model.fit(x_train, y_train)
    return model


def train_svm_rbf(x_train: np.ndarray, y_train: np.ndarray, c_value: float = 10.0) -> SVC:
    model = SVC(kernel="rbf", C=c_value, probability=True, class_weight="balanced")
    model.fit(x_train, y_train)
    return model


def evaluate_classifier(model: Any, x_test: np.ndarray, y_test: np.ndarray, class_count: int | None = None) -> tuple[EvaluationResult, np.ndarray, np.ndarray | None]:
    predictions = model.predict(x_test)
    probabilities = None
    roc_auc = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
    elif hasattr(model, "decision_function"):
        decision_values = model.decision_function(x_test)
        if decision_values.ndim == 1:
            probabilities = np.column_stack([1 - decision_values, decision_values])
        else:
            probabilities = decision_values

    average = "weighted"
    result = EvaluationResult(
        model_name=model.__class__.__name__,
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, average=average, zero_division=0)),
        recall=float(recall_score(y_test, predictions, average=average, zero_division=0)),
        f1=float(f1_score(y_test, predictions, average=average, zero_division=0)),
    )

    if probabilities is not None and class_count is not None:
        try:
            classes = np.arange(class_count)
            y_test_binarized = label_binarize(y_test, classes=classes)
            if probabilities.shape[1] == 1:
                roc_auc = float(roc_auc_score(y_test_binarized, probabilities, multi_class="ovr"))
            else:
                roc_auc = float(roc_auc_score(y_test_binarized, probabilities, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None

    result.roc_auc = roc_auc
    return result, predictions, probabilities


def build_cnn_model(input_shape: tuple[int, int, int], class_count: int):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(class_count, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mango_leaf_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 10):
    """Train CNN model"""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from .config import IMAGE_SIZE, BATCH_SIZE
    
    # Reshape data for CNN (add channel dimension)
    X_train_expanded = np.expand_dims(X_train, -1)
    if X_train_expanded.shape[1] != IMAGE_SIZE[0]:
        # Resize if needed
        from tensorflow.image import resize
        X_train_expanded = resize(X_train_expanded, IMAGE_SIZE).numpy()
    
    X_test_expanded = np.expand_dims(X_test, -1)
    if X_test_expanded.shape[1] != IMAGE_SIZE[0]:
        X_test_expanded = resize(X_test_expanded, IMAGE_SIZE).numpy()
    
    # Build model
    class_count = len(np.unique(y_train))
    model = build_cnn_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1), class_count=class_count)
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]
    
    # Train
    model.fit(
        train_datagen.flow(X_train_expanded, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_test_expanded, y_test),
        epochs=epochs,
        callbacks=callbacks,
        verbose=0
    )
    
    return model
