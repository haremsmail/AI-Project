#!/usr/bin/env python
"""Quick training on real dataset sample for better accuracy."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODELS_DIR, DATA_DIR
from src.utils import discover_images, set_global_seed
from src.features import load_image, extract_handcrafted_matrix
from src.models import train_knn, train_gaussian_nb, train_svm_rbf, evaluate_classifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

print("=" * 70)
print("QUICK TRAINING - Using real dataset samples")
print("=" * 70)

set_global_seed(42)

# Find dataset
raw_data_dir = DATA_DIR / "raw" / "MangoLeafBD Dataset"
if not raw_data_dir.exists():
    print("Error: Dataset not found!")
    sys.exit(1)

# Discover images
print("\nSearching for images...")
df = discover_images(raw_data_dir)

if len(df) == 0:
    print("Error: No images found!")
    sys.exit(1)

print(f"Found {len(df)} images")

# Sample 10 images per class for quick training (80 total)
print("\nSampling 10 images per class...")
df_sample_list = []
for label_name in df['label'].unique():
    label_df = df[df['label'] == label_name]
    sample = label_df.sample(min(10, len(label_df)), random_state=42)
    df_sample_list.append(sample)

df_sample = pd.concat(df_sample_list, ignore_index=True)
print(f"Using {len(df_sample)} images for training")

# Extract features
print("\nExtracting features...")
features_list = []
labels_list = []

for idx, row in df_sample.iterrows():
    try:
        image_path = row.get('image_path') or row.get(0)
        label = row.get('label') or row.get(1)
        
        img = load_image(image_path)
        feat = extract_handcrafted_matrix(img)
        features_list.append(feat)
        labels_list.append(label)
    except Exception as e:
        continue

if len(features_list) == 0:
    print("Error: Could not extract features!")
    sys.exit(1)

X = np.array(features_list)
y = np.array(labels_list)

print(f"Extracted {len(X)} feature vectors")

# Train label encoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train models
print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

try:
    print("\nTraining kNN...")
    knn = train_knn(X_train, y_train, n_neighbors=3)
    knn_result = evaluate_classifier(knn, X_test, y_test, len(le.classes_))
    joblib.dump(knn, MODELS_DIR / "knn_model.pkl")
    print(f"  Accuracy: {knn_result.accuracy:.1%}")
except Exception as e:
    print(f"  Error: {e}")

try:
    print("Training Naive Bayes...")
    nb = train_gaussian_nb(X_train, y_train)
    nb_result = evaluate_classifier(nb, X_test, y_test, len(le.classes_))
    joblib.dump(nb, MODELS_DIR / "nb_model.pkl")
    print(f"  Accuracy: {nb_result.accuracy:.1%}")
except Exception as e:
    print(f"  Error: {e}")

try:
    print("Training SVM...")
    svm = train_svm_rbf(X_train, y_train, c_value=1.0)
    svm_result = evaluate_classifier(svm, X_test, y_test, len(le.classes_))
    joblib.dump(svm, MODELS_DIR / "svm_model.pkl")
    print(f"  Accuracy: {svm_result.accuracy:.1%}")
except Exception as e:
    print(f"  Error: {e}")

# Save label encoder and scaler
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

print("\n" + "=" * 70)
print("SUCCESS! Models trained on real data")
print("=" * 70)
print("\nYou can now run: python main.py")
print("Use option 1 to train on the full dataset for better accuracy")
print("=" * 70)
