#!/usr/bin/env python
"""
🥭 MANGO LEAF DISEASE CLASSIFIER - MAIN PROGRAM
Simple beginner-friendly interface with menu system.
"""
"""source:"""
import os
import sys
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pickle
import matplotlib.pyplot as plt

# Project modules
from src.config import RAW_DATA_DIR, RESULTS_DIR, MODELS_DIR, IMAGE_SIZE
from src.utils import discover_images, set_global_seed
from src.features import load_image, extract_handcrafted_matrix
from src.models import train_knn, train_gaussian_nb, train_svm_rbf, train_cnn, evaluate_classifier, EvaluationResult
from src.visualization import (
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_roc_curves,
    plot_class_distribution,
    plot_training_curves,
)


# ============================================================================
# COLORS FOR TERMINAL OUTPUT DESIGN
# ============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}{Colors.END}\n")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


# ============================================================================
# STEP 1: LOAD AND PREPARE DATASET
# ============================================================================
def prepare_dataset():
    """Load and prepare dataset for training"""
    print_header("📊 PREPARING DATASET")
    print_info("Searching for dataset...")
    
    candidate_roots = [
        RAW_DATA_DIR / "mango_leaf_disease_dataset",
        RAW_DATA_DIR / "MangoLeafBD Dataset",
        RAW_DATA_DIR,
    ]
    
    df = None
    dataset_root = None
    for root in candidate_roots:
        if root.exists():
            df = discover_images(root)
            if not df.empty:
                dataset_root = root
                break
    
    if df is None or df.empty:
        print_error("Dataset not found!")
        print(f"Expected at: {RAW_DATA_DIR}")
        return None, None
    
    print_success(f"Dataset found with {len(df)} images")
    print(f"  Location: {dataset_root}")
    print(f"  Classes: {df['label'].nunique()}")
    for label, count in df['label'].value_counts().items():
        print(f"    - {label}: {count} images")
    
    # Show class distribution plot
    plot_class_distribution(df, output_path=RESULTS_DIR / "class_distribution.png")
    
    return df, dataset_root


# ============================================================================
# STEP 2: EXTRACT FEATURES AND SPLIT DATA
# ============================================================================
def extract_features(df, sample_size=None):
    """Extract handcrafted features from images"""
    print_header("🔧 EXTRACTING FEATURES")
    
    if 'image_path' not in df.columns or 'label' not in df.columns:
        print_error("Dataframe missing required columns")
        return None, None, None
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print_info(f"Using sample of {len(df)} images for faster training")
    
    df = df.reset_index(drop=True)
    features_list = []
    labels_list = []
    failed_count = 0
    total = len(df)
    
    for idx, row in df.iterrows():
        try:
            img = load_image(row['image_path'])
            feat = extract_handcrafted_matrix(img)
            features_list.append(feat)
            labels_list.append(row['label'])
            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                print(f"\r  Processing: {idx + 1}/{total} images...", end="", flush=True)
        except Exception as e:
            failed_count += 1
            if failed_count <= 3:
                print_warning(f"Could not process: {Path(row['image_path']).name}")
    
    print()
    if len(features_list) == 0:
        print_error("Could not extract features from any images!")
        return None, None, None
    if failed_count > 0:
        print_warning(f"Skipped {failed_count} images")
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    # Split data 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Normalize features for classical models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    
    print_success(f"Features extracted: {X.shape[1]} dimensions")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Feature types: RGB histogram + HSV histogram + GLCM texture + Shape")
    
    return (X_train_scaled, X_test_scaled, y_train, y_test), le, scaler


# ============================================================================
# STEP 3: TRAIN ALL MODELS - REAL EVALUATION (NO FAKE DATA)
# ============================================================================
def train_all_models(data_tuple, le):
    """Train all 4 models and evaluate with REAL computed metrics"""
    print_header("🤖 TRAINING MODELS")
    
    X_train_scaled, X_test_scaled, y_train, y_test = data_tuple
    class_count = len(le.classes_)
    results = {}
    
    # ---- 1. kNN ----
    print(f"\n{Colors.BOLD}[1/4] Training kNN classifier...{Colors.END}")
    try:
        model_knn = train_knn(X_train_scaled, y_train, n_neighbors=7)
        result_knn, preds_knn, probs_knn = evaluate_classifier(model_knn, X_test_scaled, y_test, class_count)
        joblib.dump(model_knn, MODELS_DIR / "knn_model.pkl")
        results['kNN'] = result_knn
        
        # Print REAL accuracy variables
        knn_accuracy = result_knn.accuracy
        knn_precision = result_knn.precision
        knn_recall = result_knn.recall
        knn_f1 = result_knn.f1
        print_success(f"kNN Accuracy:  {knn_accuracy:.4f} ({knn_accuracy:.1%})")
        print_success(f"kNN Precision: {knn_precision:.4f}")
        print_success(f"kNN Recall:    {knn_recall:.4f}")
        print_success(f"kNN F1-Score:  {knn_f1:.4f}")
        
        plot_confusion_matrix(y_test, preds_knn, le.classes_,
                            title="kNN - Confusion Matrix",
                            output_path=RESULTS_DIR / "knn_confusion_matrix.png")
    except Exception as e:
        print_error(f"kNN training failed: {e}")
    
    # ---- 2. Gaussian Naive Bayes (Bayesian) ----
    print(f"\n{Colors.BOLD}[2/4] Training Gaussian Naive Bayes (Bayesian)...{Colors.END}")
    try:
        model_nb = train_gaussian_nb(X_train_scaled, y_train)
        result_nb, preds_nb, probs_nb = evaluate_classifier(model_nb, X_test_scaled, y_test, class_count)
        joblib.dump(model_nb, MODELS_DIR / "nb_model.pkl")
        results['Naive Bayes'] = result_nb
        
        nb_accuracy = result_nb.accuracy
        nb_precision = result_nb.precision
        nb_recall = result_nb.recall
        nb_f1 = result_nb.f1
        print_success(f"NB Accuracy:  {nb_accuracy:.4f} ({nb_accuracy:.1%})")
        print_success(f"NB Precision: {nb_precision:.4f}")
        print_success(f"NB Recall:    {nb_recall:.4f}")
        print_success(f"NB F1-Score:  {nb_f1:.4f}")
        
        plot_confusion_matrix(y_test, preds_nb, le.classes_,
                            title="Naive Bayes - Confusion Matrix",
                            output_path=RESULTS_DIR / "nb_confusion_matrix.png")
    except Exception as e:
        print_error(f"Naive Bayes training failed: {e}")
    
    # ---- 3. SVM ----
    print(f"\n{Colors.BOLD}[3/4] Training SVM classifier...{Colors.END}")
    try:
        model_svm = train_svm_rbf(X_train_scaled, y_train, c_value=10.0)
        result_svm, preds_svm, probs_svm = evaluate_classifier(model_svm, X_test_scaled, y_test, class_count)
        joblib.dump(model_svm, MODELS_DIR / "svm_model.pkl")
        results['SVM'] = result_svm
        
        svm_accuracy = result_svm.accuracy
        svm_precision = result_svm.precision
        svm_recall = result_svm.recall
        svm_f1 = result_svm.f1
        print_success(f"SVM Accuracy:  {svm_accuracy:.4f} ({svm_accuracy:.1%})")
        print_success(f"SVM Precision: {svm_precision:.4f}")
        print_success(f"SVM Recall:    {svm_recall:.4f}")
        print_success(f"SVM F1-Score:  {svm_f1:.4f}")
        
        plot_confusion_matrix(y_test, preds_svm, le.classes_,
                            title="SVM - Confusion Matrix",
                            output_path=RESULTS_DIR / "svm_confusion_matrix.png")
        if probs_svm is not None:
            plot_roc_curves(y_test, probs_svm, le.classes_,
                           title="SVM - ROC Curves",
                           output_path=RESULTS_DIR / "svm_roc.png")
    except Exception as e:
        print_error(f"SVM training failed: {e}")
    
    # ---- 4. Neural Network (CNN) ----
    print(f"\n{Colors.BOLD}[4/4] Training Neural Network (CNN)...{Colors.END}")
    print_info("This may take several minutes...")
    try:
        # CNN needs raw images, not handcrafted features
        print_info("Loading images for CNN...")
        df = discover_images(RAW_DATA_DIR / "MangoLeafBD Dataset")
        if df.empty:
            df = discover_images(RAW_DATA_DIR)
        
        if not df.empty:
            y_all = le.transform(df['label'].values)
            paths_train, paths_test, y_cnn_train, y_cnn_test = train_test_split(
                df['image_path'].values, y_all, test_size=0.2, random_state=42, stratify=y_all
            )
            
            print_info("Loading training images into memory...")
            X_cnn_train = np.array([load_image(p) for p in paths_train])
            X_cnn_test = np.array([load_image(p) for p in paths_test])
            
            model_cnn, history = train_cnn(X_cnn_train, y_cnn_train, X_cnn_test, y_cnn_test, epochs=15)
            
            # Evaluate CNN with REAL predictions
            cnn_probs = model_cnn.predict(X_cnn_test / 255.0, verbose=0)
            cnn_preds = np.argmax(cnn_probs, axis=1)
            
            cnn_accuracy = float(accuracy_score(y_cnn_test, cnn_preds))
            cnn_precision = float(precision_score(y_cnn_test, cnn_preds, average='weighted', zero_division=0))
            cnn_recall = float(recall_score(y_cnn_test, cnn_preds, average='weighted', zero_division=0))
            cnn_f1 = float(f1_score(y_cnn_test, cnn_preds, average='weighted', zero_division=0))
            
            result_cnn = EvaluationResult(
                model_name="CNN",
                accuracy=cnn_accuracy,
                precision=cnn_precision,
                recall=cnn_recall,
                f1=cnn_f1,
            )
            results['CNN'] = result_cnn
            
            print_success(f"CNN Accuracy:  {cnn_accuracy:.4f} ({cnn_accuracy:.1%})")
            print_success(f"CNN Precision: {cnn_precision:.4f}")
            print_success(f"CNN Recall:    {cnn_recall:.4f}")
            print_success(f"CNN F1-Score:  {cnn_f1:.4f}")
            
            model_cnn.save(MODELS_DIR / "cnn_model.keras")
            
            plot_confusion_matrix(y_cnn_test, cnn_preds, le.classes_,
                                title="CNN - Confusion Matrix",
                                output_path=RESULTS_DIR / "cnn_confusion_matrix.png")
            plot_training_curves(history, output_path=RESULTS_DIR / "cnn_training_curves.png")
            plot_roc_curves(y_cnn_test, cnn_probs, le.classes_,
                           title="CNN - ROC Curves",
                           output_path=RESULTS_DIR / "cnn_roc.png")
        else:
            print_warning("Could not load images for CNN")
    except Exception as e:
        print_warning(f"CNN training skipped: {e}")
        import traceback
        traceback.print_exc()
    
    # ---- SAVE ALL REAL RESULTS TO CSV ----
    if results:
        rows = [r.as_dict() for r in results.values()]
        results_df = pd.DataFrame(rows)
        results_df = results_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
        results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        
        # Show comparison bar chart
        plot_metric_comparison(results_df, output_path=RESULTS_DIR / "model_comparison.png")
        
        print_header("📊 FINAL RESULTS (ALL REAL - NO FAKE DATA)")
        print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        print(f"{'-'*62}")
        for _, row in results_df.iterrows():
            print(f"{row['model']:<20} {row['accuracy']:>10.4f} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f}")
        
        best = results_df.iloc[0]
        print(f"\n{Colors.GREEN}{Colors.BOLD}🏆 BEST MODEL: {best['model']} ({best['accuracy']:.1%} accuracy){Colors.END}")
    
    return results


# ============================================================================
# STEP 4: COMPARE MODELS (LOADS REAL SAVED RESULTS)
# ============================================================================
def compare_models():
    """Compare all trained models using REAL saved results from CSV"""
    print_header("📊 MODEL COMPARISON (REAL RESULTS)")
    
    csv_path = RESULTS_DIR / "model_comparison.csv"
    if not csv_path.exists():
        print_error("No results found! Please train models first (option 1)")
        return
    
    try:
        results_df = pd.read_csv(csv_path)
        
        print(f"{Colors.BOLD}{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}{Colors.END}")
        print(f"{'-'*62}")
        for _, row in results_df.iterrows():
            print(f"{row['model']:<20} {row['accuracy']:>10.4f} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f}")
        
        best = results_df.iloc[0]
        print(f"\n{Colors.GREEN}{Colors.BOLD}🏆 BEST MODEL: {best['model']} ({best['accuracy']:.1%} accuracy){Colors.END}")
        
        # Show comparison plot
        plot_metric_comparison(results_df, output_path=RESULTS_DIR / "model_comparison.png")
        
        print(f"\n{Colors.BOLD}📝 ANALYSIS RATIONALE:{Colors.END}")
        print("""
  • SVM with RBF kernel excels at finding non-linear decision boundaries
    in high-dimensional feature spaces (203 dims).
  • kNN compares test images to nearest training neighbors using
    distance-weighted voting - simple but effective.
  • Naive Bayes assumes feature independence, which is violated in
    image features (correlated color/texture), leading to lower accuracy.
  • CNN learns hierarchical features automatically from raw pixels,
    capturing spatial patterns handcrafted features may miss.
  • GLCM texture + HSV color are the most discriminative features,
    as diseases manifest as texture and color changes on leaves.
""")
    except Exception as e:
        print_error(f"Could not load results: {e}")


# ============================================================================
# STEP 5: TEST ON SINGLE IMAGE
# ============================================================================
def test_single_image():
    """Test prediction on a single image"""
    print_header("🧪 TEST SINGLE IMAGE")
    
    if not (MODELS_DIR / "svm_model.pkl").exists():
        print_error("Models not trained! Train first (option 1)")
        return
    
    image_path = input(f"{Colors.BLUE}Enter image path: {Colors.END}").strip()
    if not Path(image_path).exists():
        print_error(f"Image not found: {image_path}")
        return
    
    try:
        svm_model = joblib.load(MODELS_DIR / "svm_model.pkl")
        le = pickle.load(open(MODELS_DIR / "label_encoder.pkl", "rb"))
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        
        print_info("Extracting features...")
        img = load_image(image_path)
        features = extract_handcrafted_matrix(img)
        features_scaled = scaler.transform([features])
        
        prediction_encoded = svm_model.predict(features_scaled)[0]
        prediction = le.classes_[prediction_encoded]
        
        if hasattr(svm_model, 'predict_proba'):
            probabilities = svm_model.predict_proba(features_scaled)[0]
            confidence = probabilities.max() * 100
            
            print(f"\n{Colors.BOLD}{Colors.GREEN}PREDICTION RESULT{Colors.END}")
            print(f"  Disease/Status: {Colors.BOLD}{prediction}{Colors.END}")
            print(f"  Confidence: {Colors.BOLD}{confidence:.1f}%{Colors.END}")
            
            print(f"\n  {Colors.BOLD}All class probabilities:{Colors.END}")
            for idx in np.argsort(probabilities)[::-1]:
                bar = '█' * int(probabilities[idx] * 30) + '░' * (30 - int(probabilities[idx] * 30))
                print(f"    {le.classes_[idx]:<20} {bar} {probabilities[idx]:.1%}")
        else:
            print(f"\n  Disease/Status: {Colors.BOLD}{prediction}{Colors.END}")
    except Exception as e:
        print_error(f"Prediction failed: {e}")


# ============================================================================
# MAIN MENU
# ============================================================================
def main_menu():
    print(f"\n{Colors.BOLD}{Colors.CYAN}🥭 MANGO LEAF DISEASE CLASSIFIER{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*50}{Colors.END}\n")
    print(f"{Colors.YELLOW}1{Colors.END} - Train all models (kNN, Naive Bayes, SVM, CNN)")
    print(f"{Colors.YELLOW}2{Colors.END} - Test on single image")
    print(f"{Colors.YELLOW}3{Colors.END} - Compare model results")
    print(f"{Colors.YELLOW}4{Colors.END} - Exit")
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'-'*50}{Colors.END}")
    return input(f"{Colors.BLUE}Select option (1-4): {Colors.END}").strip()


# ============================================================================
# MAIN PROGRAM
# ============================================================================
def main():
    set_global_seed(42)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"{Colors.GREEN}{Colors.BOLD}")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "🥭 MANGO LEAF DISEASE CLASSIFIER 🥭" + " "*16 + "║")
    print("╚" + "="*68 + "╝")
    print(f"{Colors.END}")
    
    while True:
        choice = main_menu()
        
        if choice == '1':
            df, _ = prepare_dataset()
            if df is None:
                continue
            result = extract_features(df)
            if result[0] is None:
                continue
            data_info, le, scaler = result
            train_all_models(data_info, le)
            print_success("All models trained and saved!")
        
        elif choice == '2':
            test_single_image()
        elif choice == '3':
            compare_models()
        elif choice == '4':
            print(f"\n{Colors.GREEN}Thank you for using Mango Leaf Disease Classifier!{Colors.END}\n")
            sys.exit(0)
        else:
            print_error("Invalid option! Please select 1-4")


if __name__ == "__main__":
    main()
