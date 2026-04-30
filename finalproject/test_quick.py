#!/usr/bin/env python
"""Quick test of the full pipeline with sample data."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.config import RAW_DATA_DIR, MODELS_DIR
from src.utils import discover_images
from src.features import load_image, extract_handcrafted_matrix
from src.models import train_knn, train_gaussian_nb, train_svm_rbf, evaluate_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np

print("=" * 70)
print("MANGO LEAF DISEASE - QUICK TEST (200 SAMPLES)")
print("=" * 70)

# Find dataset
candidate_roots = [
    RAW_DATA_DIR / "mango_leaf_disease_dataset",
    RAW_DATA_DIR / "MangoLeafBD Dataset",
    RAW_DATA_DIR,
]

df = None
for root in candidate_roots:
    if root.exists():
        df = discover_images(root)
        if not df.empty:
            break

if df is None or df.empty:
    print("ERROR: Dataset not found!")
    sys.exit(1)

print(f"\n✓ Found {len(df)} total images")
print(f"Classes: {df['label'].nunique()}")

# Sample 200 images (25 per class for 8 classes)
print(f"\nUsing sample of 200 images for quick testing (25 per class)...")
df_sample = df.groupby('label').apply(lambda x: x.sample(n=min(25, len(x)), random_state=42)).reset_index(drop=True)
print(f"✓ Sample size: {len(df_sample)}")

# Extract features
print(f"\nExtracting features from {len(df_sample)} images...")
features_list = []
labels_list = []
failed = 0

for idx, row in df_sample.iterrows():
    if idx % 50 == 0:
        print(f"  Processed {idx}/{len(df_sample)}...", end="\r")
    
    try:
        img = load_image(row['image_path'])
        feat = extract_handcrafted_matrix(img)
        features_list.append(feat)
        labels_list.append(row['label'])
    except Exception as e:
        failed += 1
        print(f"  Failed: {Path(row['image_path']).name}")

print(f"\n✓ Features extracted: {len(features_list)} images")
if failed > 0:
    print(f"⚠ Failed: {failed} images")

# Prepare data
X = np.array(features_list)
y = np.array(labels_list)

print(f"\n✓ Feature matrix shape: {X.shape}")
print(f"  Dimensions: {X.shape[1]}")
print(f"  Classes: {np.unique(y)}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Data split:")
print(f"  Training: {len(X_train)} samples")
print(f"  Testing: {len(X_test)} samples")

# Train models
print(f"\n" + "=" * 70)
print("Training models...")
print("=" * 70)

# KNN
print(f"\n[1/3] Training kNN...")
knn_model = train_knn(X_train_scaled, y_train)
knn_result = evaluate_classifier(knn_model, X_test_scaled, y_test, len(le.classes_))
print(f"✓ kNN Accuracy: {knn_result.accuracy:.1%}")
joblib.dump(knn_model, MODELS_DIR / "knn_model.pkl")

# Naive Bayes
print(f"\n[2/3] Training Naive Bayes...")
nb_model = train_gaussian_nb(X_train_scaled, y_train)
nb_result = evaluate_classifier(nb_model, X_test_scaled, y_test, len(le.classes_))
print(f"✓ Naive Bayes Accuracy: {nb_result.accuracy:.1%}")
joblib.dump(nb_model, MODELS_DIR / "nb_model.pkl")

# SVM
print(f"\n[3/3] Training SVM...")
svm_model = train_svm_rbf(X_train_scaled, y_train)
svm_result = evaluate_classifier(svm_model, X_test_scaled, y_test, len(le.classes_))
print(f"✓ SVM Accuracy: {svm_result.accuracy:.1%}")
joblib.dump(svm_model, MODELS_DIR / "svm_model.pkl")

# Save encoder and scaler
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

# Display results
print(f"\n" + "=" * 70)
print("✓ TRAINING COMPLETE!")
print("=" * 70)
print(f"\nModel Accuracy Comparison:")
print(f"  kNN:         {knn_result.accuracy:.1%}")
print(f"  Naive Bayes: {nb_result.accuracy:.1%}")
print(f"  SVM:         {svm_result.accuracy:.1%}")

best_acc = max(knn_result.accuracy, nb_result.accuracy, svm_result.accuracy)
if svm_result.accuracy == best_acc:
    print(f"\n🏆 BEST: SVM with {svm_result.accuracy:.1%} accuracy")
elif nb_result.accuracy == best_acc:
    print(f"\n🏆 BEST: Naive Bayes with {nb_result.accuracy:.1%} accuracy")
else:
    print(f"\n🏆 BEST: kNN with {knn_result.accuracy:.1%} accuracy")

print(f"\n✓ Models saved to: {MODELS_DIR}")
print(f"\nYou can now run: python main.py")
print(f"Then select option 2 to test on a single image!")
print("=" * 70)
