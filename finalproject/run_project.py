#!/usr/bin/env python
"""
SIMPLE PROJECT TEST SCRIPT
Run this to see if the project works!
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("\n" + "="*70)
print("🚀 MANGO LEAF DISEASE CLASSIFICATION - SIMPLE TEST")
print("="*70 + "\n")

# ============================================================================
# STEP 1: Test imports
# ============================================================================
print("STEP 1: Testing imports...")
try:
    from src.config import RAW_DATA_DIR, RESULTS_DIR, MODELS_DIR
    from src.utils import discover_images, stratified_split, set_global_seed
    from src.features import load_image, extract_handcrafted_matrix
    from src.models import train_knn, train_gaussian_nb, train_svm_rbf, evaluate_classifier
    print("✓ All imports successful!\n")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: Check dataset
# ============================================================================
print("STEP 2: Checking dataset...")
try:
    # Try different dataset paths
    dataset_paths = [
        RAW_DATA_DIR / "mango_leaf_disease_dataset",
        RAW_DATA_DIR,
    ]
    
    df = None
    dataset_root = None
    
    for path in dataset_paths:
        if path.exists():
            df = discover_images(path)
            if not df.empty:
                dataset_root = path
                break
    
    if df is None or df.empty:
        print(f"✗ No dataset found at:")
        for p in dataset_paths:
            print(f"  - {p}")
        print("\nPlease download dataset first:")
        print("  1. Run: python setup_dataset.py")
        print("  2. Or download manually from: https://www.kaggle.com/datasets/warcoder/mango-leaf-disease-dataset")
        sys.exit(1)
    
    print(f"✓ Dataset found at: {dataset_root}")
    print(f"  Total images: {len(df)}")
    print(f"  Classes: {df['label'].nunique()}")
    print(f"  Distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"    - {label}: {count}")
    print()
    
except Exception as e:
    print(f"✗ Dataset error: {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: Prepare data
# ============================================================================
print("STEP 3: Preparing data...")
try:
    set_global_seed(42)
    
    # Load a few images as features
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import numpy as np
    
    print("  Loading and extracting features from images...")
    
    # Sample 100 images for quick test
    sample_df = df.sample(n=min(100, len(df)), random_state=42)
    
    features_list = []
    labels_list = []
    
    for idx, row in sample_df.iterrows():
        try:
            img = load_image(row['image_path'])
            feat = extract_handcrafted_matrix(img)
            features_list.append(feat)
            labels_list.append(row['label'])
        except Exception as e:
            print(f"    Warning: Could not process {row['image_path']}: {e}")
            continue
    
    if len(features_list) == 0:
        print("✗ Could not extract features from any images!")
        sys.exit(1)
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Data prepared!")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Feature dimension: {X.shape[1]}")
    print()
    
except Exception as e:
    print(f"✗ Data preparation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 4: Train models
# ============================================================================
print("STEP 4: Training models...")
results = {}

try:
    # KNN
    print("  Training kNN...")
    model_knn = train_knn(X_train_scaled, y_train, n_neighbors=7)
    result_knn = evaluate_classifier(model_knn, X_test_scaled, y_test, len(le.classes_))
    results['kNN'] = result_knn
    print(f"    ✓ kNN Accuracy: {result_knn.accuracy:.3f}")
    
    # Naive Bayes
    print("  Training Gaussian Naive Bayes...")
    model_nb = train_gaussian_nb(X_train_scaled, y_train)
    result_nb = evaluate_classifier(model_nb, X_test_scaled, y_test, len(le.classes_))
    results['Naive Bayes'] = result_nb
    print(f"    ✓ NB Accuracy: {result_nb.accuracy:.3f}")
    
    # SVM
    print("  Training SVM...")
    model_svm = train_svm_rbf(X_train_scaled, y_train, c_value=10.0)
    result_svm = evaluate_classifier(model_svm, X_test_scaled, y_test, len(le.classes_))
    results['SVM'] = result_svm
    print(f"    ✓ SVM Accuracy: {result_svm.accuracy:.3f}")
    
    print()
    
except Exception as e:
    print(f"✗ Training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 5: Display results
# ============================================================================
print("="*70)
print("📊 MODEL COMPARISON RESULTS")
print("="*70)
print()

print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 70)

best_accuracy = 0
best_model = None

for model_name, result in results.items():
    print(f"{model_name:<15} {result.accuracy:.3f}       {result.precision:.3f}       {result.recall:.3f}       {result.f1:.3f}")
    
    if result.accuracy > best_accuracy:
        best_accuracy = result.accuracy
        best_model = model_name

print()
print(f"🏆 BEST MODEL: {best_model} ({best_accuracy:.1%} accuracy)")
print()

# ============================================================================
# STEP 6: Summary
# ============================================================================
print("="*70)
print("✅ PROJECT SUMMARY")
print("="*70)
print()
print("What this script did:")
print("  1. ✓ Loaded dataset (600+ mango leaf images)")
print("  2. ✓ Extracted features (RGB, HSV, GLCM, shape)")
print("  3. ✓ Split data (train/test)")
print("  4. ✓ Trained 3 classifiers (kNN, NB, SVM)")
print("  5. ✓ Evaluated models")
print()
print("Next steps to see FULL results:")
print("  1. Run complete notebook: Run Jupyter notebook cells")
print("  2. This will also train CNN (deep learning model)")
print("  3. Generate detailed report: python generate_report.py")
print()
print("="*70)
print("✅ TEST COMPLETE - PROJECT IS WORKING!")
print("="*70 + "\n")
