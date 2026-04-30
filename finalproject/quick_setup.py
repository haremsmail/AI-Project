#!/usr/bin/env python
"""Quick setup - creates dummy trained models so UI works immediately."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODELS_DIR, RAW_DATA_DIR
from src.utils import discover_images
from src.features import load_image, extract_handcrafted_matrix
from src.models import train_knn, train_gaussian_nb, train_svm_rbf, evaluate_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

print("=" * 70)
print("QUICK SETUP - Creating trained models...")
print("=" * 70)

# Find dataset
candidate_roots = [
    RAW_DATA_DIR / 'mango_leaf_disease_dataset',
    RAW_DATA_DIR / 'MangoLeafBD Dataset',
    RAW_DATA_DIR,
]

df = None
for root in candidate_roots:
    if root.exists():
        df = discover_images(root)
        if not df.empty:
            print(f"\n[OK] Found dataset with {len(df)} images")
            break

if df is None or df.empty:
    print("ERROR: Dataset not found!")
    sys.exit(1)

# Get only 50 images per class for VERY fast training (400 total)
print("\n[1] Sampling 50 images per class (fast setup)...")
df_sample = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=min(50, len(x)), random_state=42)
).reset_index(drop=True)

print(f"[OK] Using {len(df_sample)} images for training")

# Extract features - with error handling
print(f"\n[2] Extracting features from {len(df_sample)} images...")
features_list = []
labels_list = []
processed = 0

for idx, row in df_sample.iterrows():
    if (idx + 1) % 50 == 0:
        print(f"    Processed {idx + 1}/{len(df_sample)}...", end="\r")
    
    try:
        img = load_image(row['image_path'])
        feat = extract_handcrafted_matrix(img)
        features_list.append(feat)
        labels_list.append(row['label'])
        processed += 1
    except:
        pass  # Skip bad images

print(f"\n[OK] Extracted {processed} feature vectors")

if processed < 10:
    print("ERROR: Could not extract enough features!")
    sys.exit(1)

# Prepare data
X = np.array(features_list)
y = np.array(labels_list)

print(f"\n[3] Preparing data...")
print(f"    Feature shape: {X.shape}")
print(f"    Classes: {len(np.unique(y))}")

# Encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
print(f"\n[4] Training models...")

knn_model = train_knn(X_train_scaled, y_train)
knn_result = evaluate_classifier(knn_model, X_test_scaled, y_test, len(le.classes_))
print(f"    kNN: {knn_result.accuracy:.1%}")
joblib.dump(knn_model, MODELS_DIR / "knn_model.pkl")

nb_model = train_gaussian_nb(X_train_scaled, y_train)
nb_result = evaluate_classifier(nb_model, X_test_scaled, y_test, len(le.classes_))
print(f"    Naive Bayes: {nb_result.accuracy:.1%}")
joblib.dump(nb_model, MODELS_DIR / "nb_model.pkl")

svm_model = train_svm_rbf(X_train_scaled, y_train)
svm_result = evaluate_classifier(svm_model, X_test_scaled, y_test, len(le.classes_))
print(f"    SVM: {svm_result.accuracy:.1%}")
joblib.dump(svm_model, MODELS_DIR / "svm_model.pkl")

# Save metadata
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

print(f"\n" + "=" * 70)
print("SUCCESS! Models are ready.")
print("=" * 70)
print(f"\nYou can now run: python main.py")
print(f"Use option 2 to test on a leaf image!")
print(f"\nNote: These are quick models trained on {processed} samples.")
print(f"For better accuracy, use option 1 in main.py to train on all {len(df)} images.")
print("=" * 70)
