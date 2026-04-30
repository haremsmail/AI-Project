#!/usr/bin/env python
"""Create pre-trained dummy models to get started immediately."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODELS_DIR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

print("=" * 70)
print("INSTANT SETUP - Creating pre-trained models")
print("=" * 70)

# Class labels (8 diseases)
classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
           'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# Create fake training data (just enough to train models)
np.random.seed(42)
n_samples = 100
n_features = 203  # handcrafted features dimension

X_train = np.random.randn(n_samples, n_features).astype(np.float32)
y_train = np.random.randint(0, len(classes), n_samples)

print(f"\nCreating models with synthetic training data...")
print(f"  Samples: {n_samples}")
print(f"  Features: {n_features}")
print(f"  Classes: {len(classes)}")

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train models
print(f"\nTraining models...")

knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(X_train_scaled, y_train)
joblib.dump(knn, MODELS_DIR / "knn_model.pkl")
print(f"  [OK] kNN model saved")

nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
joblib.dump(nb, MODELS_DIR / "nb_model.pkl")
print(f"  [OK] Naive Bayes model saved")

svm = SVC(kernel='rbf', C=10.0, class_weight='balanced', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
joblib.dump(svm, MODELS_DIR / "svm_model.pkl")
print(f"  [OK] SVM model saved")

# Save label encoder
le = LabelEncoder()
le.fit(classes)
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
print(f"  [OK] Label encoder saved")

# Save scaler
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
print(f"  [OK] Scaler saved")

print(f"\n" + "=" * 70)
print("SUCCESS! Models are ready to use.")
print("=" * 70)
print(f"\nYou can now run: python main.py")
print(f"\nOptions:")
print(f"  1 - Train all models (on full 4000 image dataset)")
print(f"  2 - Test on a leaf image")
print(f"  3 - Compare model accuracy")
print(f"  4 - Exit")
print(f"\nModels saved to: {MODELS_DIR}")
print("=" * 70)
