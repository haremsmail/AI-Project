#!/usr/bin/env python
"""Evaluate saved models on new data or batch predictions.

This script loads trained models and applies them to evaluate
new images or generate batch predictions.

Usage:
    python evaluate_models.py --image path/to/leaf.jpg
    python evaluate_models.py --batch path/to/images/
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from src.config import MODELS_DIR, IMAGE_SIZE
from src.features import load_image, handcrafted_features


def load_trained_models():
    """Load all trained models from disk."""
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    handcrafted_scaler = joblib.load(MODELS_DIR / "handcrafted_scaler.joblib")
    deep_scaler = joblib.load(MODELS_DIR / "deep_feature_scaler.joblib")
    svm_model = joblib.load(MODELS_DIR / "svm_handcrafted.joblib")
    cnn_model = load_model(MODELS_DIR / "mango_leaf_cnn.keras")
    return {
        "label_encoder": label_encoder,
        "handcrafted_scaler": handcrafted_scaler,
        "deep_scaler": deep_scaler,
        "svm_model": svm_model,
        "cnn_model": cnn_model,
    }


def predict_single_image(image_path: str, models: dict):
    """Predict class for a single image using CNN."""
    image = load_image(image_path, image_size=IMAGE_SIZE)
    image_batch = np.expand_dims(image, 0) / 255.0

    probabilities = models["cnn_model"].predict(image_batch, verbose=0)
    class_id = np.argmax(probabilities[0])
    confidence = float(probabilities[0][class_id])
    class_name = models["label_encoder"].classes_[class_id]

    return class_name, confidence, probabilities[0]


def batch_predict(image_dir: str, models: dict):
    """Predict classes for all images in a directory."""
    image_dir = Path(image_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [p for p in image_dir.rglob("*") if p.suffix.lower() in image_extensions]

    results = []
    for idx, image_path in enumerate(image_paths, 1):
        try:
            class_name, confidence, probs = predict_single_image(str(image_path), models)
            results.append({"image": image_path.name, "class": class_name, "confidence": confidence})
            print(f"[{idx}/{len(image_paths)}] {image_path.name}: {class_name} ({confidence:.2%})")
        except Exception as error:
            print(f"Error processing {image_path}: {error}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on new images")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--batch", type=str, help="Directory with images")
    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.print_help()
        return

    print("Loading trained models...")
    models = load_trained_models()

    if args.image:
        print(f"\nPredicting for: {args.image}")
        class_name, confidence, probs = predict_single_image(args.image, models)
        print(f"\nPredicted Class: {class_name}")
        print(f"Confidence: {confidence:.2%}")
        print("\nClass Probabilities:")
        for class_idx, prob in enumerate(probs):
            print(f"  {models['label_encoder'].classes_[class_idx]}: {prob:.2%}")

    if args.batch:
        print(f"\nBatch predicting for directory: {args.batch}")
        results = batch_predict(args.batch, models)
        print(f"\nProcessed {len(results)} images")


if __name__ == "__main__":
    main()
