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
import pickle
import numpy as np

from src.config import MODELS_DIR, IMAGE_SIZE
from src.features import load_image, extract_handcrafted_matrix


def load_trained_models():
    """Load all trained models from disk."""
    models = {}
    
    # Load label encoder
    le_path = MODELS_DIR / "label_encoder.pkl"
    if le_path.exists():
        with open(le_path, "rb") as f:
            models["label_encoder"] = pickle.load(f)
    else:
        raise FileNotFoundError(f"Label encoder not found at {le_path}")
    
    # Load scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        models["scaler"] = joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    # Load classical models
    for name, filename in [("knn_model", "knn_model.pkl"), 
                            ("nb_model", "nb_model.pkl"),
                            ("svm_model", "svm_model.pkl")]:
        model_path = MODELS_DIR / filename
        if model_path.exists():
            models[name] = joblib.load(model_path)
    
    # Load CNN model if available
    cnn_path = MODELS_DIR / "cnn_model.keras"
    if cnn_path.exists():
        try:
            from tensorflow.keras.models import load_model
            models["cnn_model"] = load_model(cnn_path)
        except Exception as e:
            print(f"Warning: Could not load CNN model: {e}")
    
    return models


def predict_single_image(image_path: str, models: dict):
    """Predict class for a single image using the best available model (SVM preferred)."""
    # Extract handcrafted features
    image = load_image(image_path, image_size=IMAGE_SIZE)
    features = extract_handcrafted_matrix(image)
    features_scaled = models["scaler"].transform([features])
    
    le = models["label_encoder"]
    
    # Use SVM if available, otherwise fall back to other models
    model_key = None
    for key in ["svm_model", "knn_model", "nb_model"]:
        if key in models:
            model_key = key
            break
    
    if model_key is None:
        raise RuntimeError("No trained classical model found!")
    
    model = models[model_key]
    prediction_encoded = model.predict(features_scaled)[0]
    class_name = le.classes_[prediction_encoded]
    
    # Get probabilities if available
    probabilities = None
    confidence = 0.0
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(probabilities.max())
    
    return class_name, confidence, probabilities


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
        if probs is not None:
            le = models["label_encoder"]
            print("\nClass Probabilities:")
            for class_idx, prob in enumerate(probs):
                print(f"  {le.classes_[class_idx]}: {prob:.2%}")

    if args.batch:
        print(f"\nBatch predicting for directory: {args.batch}")
        results = batch_predict(args.batch, models)
        print(f"\nProcessed {len(results)} images")


if __name__ == "__main__":
    main()
