#!/usr/bin/env python
"""Generate a comprehensive analysis report from notebook results.

This script reads saved model comparison results and generates
a professional markdown report with insights and recommendations.

Usage: python generate_report.py
"""

from pathlib import Path
import json

import pandas as pd

from src.config import RESULTS_DIR


def generate_report():
    """Generate the final analysis report."""
    report_path = RESULTS_DIR / "ANALYSIS_REPORT.md"

    try:
        comparison_csv = RESULTS_DIR / "model_comparison.csv"
        if not comparison_csv.exists():
            print("Warning: model_comparison.csv not found. Run the notebook first.")
            return

        results_df = pd.read_csv(comparison_csv)
    except Exception as error:
        print(f"Error reading results: {error}")
        return

    best_model = results_df.iloc[0]
    report = f"""# Mango Leaf Disease Classification - Analysis Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report documents the performance of four machine learning classifiers trained to classify mango leaf images into disease categories or healthy leaves using both handcrafted and deep image features.

### Best Performing Model: **{best_model['model']}**

| Metric | Score |
|--------|-------|
| Accuracy | {best_model['accuracy']:.4f} |
| Precision | {best_model['precision']:.4f} |
| Recall | {best_model['recall']:.4f} |
| F1-Score | {best_model['f1']:.4f} |

---

## Model Comparison

### Performance Metrics Table

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
"""

    for _, row in results_df.iterrows():
        report += f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |\n"

    report += """

---

## Detailed Analysis

### 1. Why CNNs Often Outperform Classical ML on Image Tasks

**Neural Networks (CNN)** excel at image classification because:
- **Automatic Feature Learning**: CNNs learn hierarchical features automatically, from low-level edges to high-level disease patterns.
- **Spatial Relationships**: Convolutional layers preserve spatial information, crucial for leaf images where shape and texture matter.
- **Invariance**: Data augmentation helps models generalize to variations in angle, lighting, and leaf position.
- **Non-Linear Complexity**: Deep layers capture non-linear patterns that handcrafted features cannot represent.

### 2. Handcrafted vs. Deep Features

**Handcrafted Features** (RGB histograms, HSV, GLCM texture, shape):
- Fast to compute
- Interpretable and domain-expert friendly
- May miss subtle patterns in complex images
- Performance depends heavily on feature engineering quality

**Deep Features** (MobileNetV2 embeddings):
- Learned from ImageNet; transfer learning captures general image knowledge
- Better generalization on diverse datasets
- Computationally expensive to extract (but still faster than training from scratch)
- Often outperform handcrafted features

### 3. Classical Classifier Performance

**kNN (k-Nearest Neighbors)**:
- **Strength**: Works well with both handcrafted and deep features when features are well-separated
- **Weakness**: Slow at inference; sensitive to feature scaling and dimensionality
- **Best Use**: Baseline comparison, when you need explainability

**Gaussian Naive Bayes**:
- **Strength**: Fast training and inference; works with fewer samples
- **Weakness**: Assumes feature independence, which is violated in image features (highly correlated)
- **Best Use**: Quick prototyping; rarely best for images due to correlation assumption

**SVM with RBF Kernel**:
- **Strength**: Excellent on well-separated feature spaces; good generalization
- **Weakness**: Training is O(n²) or O(n³); slower than linear classifiers
- **Best Use**: High-dimensional features; small to medium datasets

---

## Computational Cost Analysis

| Model | Training Time | Inference Time | Memory Usage |
|-------|--------------|----------------|--------------|
| Handcrafted + kNN | Fast | Fast | Low |
| Handcrafted + Naive Bayes | Fast | Very Fast | Very Low |
| Handcrafted + SVM | Medium | Medium | Medium |
| Deep Features + kNN | Slow (feature extraction) | Medium | Medium-High |
| Deep Features + SVM | Slow (feature extraction) | Medium | Medium-High |
| CNN | Very Slow | Medium | High |

**Key Takeaway**: If inference speed is critical, use classical models. If maximum accuracy is the goal, invest in CNN training.

---

## Recommendations

1. **Primary Model**: {best_model['model']}
   - Recommended for production deployment
   - Best balance of accuracy, precision, and recall
   - Use for critical disease detection systems

2. **Feature Strategy**:
   - For real-time applications: handcrafted features + SVM
   - For maximum accuracy: deep features or end-to-end CNN
   - Consider ensemble methods combining multiple feature types

3. **Future Improvements**:
   - Collect more labeled data to improve generalization
   - Try other architectures: ResNet50, EfficientNet, Vision Transformer
   - Implement ensemble methods (voting, stacking)
   - Add explainability using GradCAM or SHAP
   - Deploy as REST API using Flask or FastAPI

---

## How to Use Saved Models

```python
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from src.features import load_image, extract_handcrafted_matrix

# Load a classical model
scaler = joblib.load('models/handcrafted_scaler.joblib')
svm_model = joblib.load('models/svm_handcrafted.joblib')

# Predict on a new image
from src.features import handcrafted_features
features = handcrafted_features('path/to/leaf.jpg')
features_scaled = scaler.transform([features])
prediction = svm_model.predict(features_scaled)

# Load CNN model
cnn_model = load_model('models/mango_leaf_cnn.keras')
image = load_image('path/to/leaf.jpg')
image_batch = np.expand_dims(image, 0) / 255.0
prediction = cnn_model.predict(image_batch)
class_id = np.argmax(prediction)
```

---

## Output Files Generated

- `model_comparison.csv` - Detailed metrics for all models
- `cnn_confusion_matrix.png` - Confusion matrix for CNN
- `cnn_training_curves.png` - Training and validation accuracy/loss
- `cnn_roc.png` - ROC curves for CNN
- `model_comparison.png` - Bar chart comparing all models
- `knn_handcrafted_confusion_matrix.png` - kNN results
- `naive_bayes_handcrafted_confusion_matrix.png` - Naive Bayes results
- `svm_rbf_handcrafted_confusion_matrix.png` - SVM results
- `class_distribution.png` - Dataset class balance visualization
- `sample_images.png` - Random samples from each class

---

## Conclusion

This project demonstrates the complete machine learning pipeline for image classification, from dataset discovery through model evaluation. By comparing multiple architectures and feature extraction methods, we identified the best approach for mango leaf disease detection, enabling faster and more accurate disease diagnosis in agricultural settings.

"""

    with open(report_path, "w") as f:
        f.write(report)

    print(f"✓ Report generated: {report_path}")


if __name__ == "__main__":
    generate_report()
