# Technical Features & Implementation Details

## Architecture Overview

### Data Pipeline
```
Dataset (folder per class)
    ↓
Image Discovery & Loading
    ↓
Stratified Split (train/val/test)
    ↓
├─→ Handcrafted Features → Classical ML Models
└─→ Deep Features (MobileNetV2) → Classical ML Models
└─→ Raw Pixels → CNN Model
```

## Feature Extraction

### 1. Handcrafted Features (192 dimensions total)

#### RGB Histogram
- **Dimensions**: 96 (32 bins × 3 channels)
- **Method**: Histogram of pixel intensities in RGB space
- **Purpose**: Captures color distribution (disease vs. healthy leaf color)
- **Computation**: O(n) with numpy

```python
# Per channel histogram over 0-256 range
# High green: healthy leaf
# Brown/yellow tones: diseased areas
```

#### HSV Histogram
- **Dimensions**: 96 (32 bins × 3 channels)
- **Method**: Histogram in HSV color space
- **Purpose**: More robust to lighting changes than RGB
- **HSV Components**:
  - Hue (0-180): Color type
  - Saturation (0-255): Color intensity
  - Value (0-255): Brightness
- **Advantage**: Disease patterns visible in hue/saturation independently

#### GLCM Texture Features
- **Dimensions**: 48 (6 properties × 8 directions)
- **Method**: Gray Level Co-occurrence Matrix
- **Properties extracted**:
  1. **Contrast**: Local intensity variation
  2. **Dissimilarity**: Similar to contrast
  3. **Homogeneity**: Local uniformity
  4. **Energy (ASM)**: Orderliness
  5. **Correlation**: Linear dependency
  6. **ASM**: Angular second moment
- **Why it works**: Disease creates texture changes (spots, lesions)
- **Directions**: 0°, 45°, 90°, 135° (and symmetric)
- **Distances**: 1 and 2 pixels

#### Shape Features
- **Dimensions**: 5
- **Features**:
  1. **Area**: Total leaf region pixels
  2. **Perimeter**: Leaf boundary length
  3. **Circularity**: 4π × Area / Perimeter²
  4. **Solidity**: Area / Convex hull area
  5. **Aspect Ratio + Extent**: Elongation + bounding box ratio
- **Why it works**: Disease can cause deformation, spots, lesions

**Total Handcrafted**: 96 + 96 + 48 + 5 = **245 dimensions**
(After normalization with StandardScaler)

### 2. Deep Features (1280 dimensions)

#### MobileNetV2 Architecture
- **Pre-trained on**: ImageNet (1.4M images, 1000 classes)
- **Model size**: ~90 MB (lightweight for mobile/edge)
- **Layers removed**: Top classification layer
- **Pooling**: Global average pooling
- **Output**: 1280-dimensional feature vector

**Why MobileNetV2**:
- Efficient: Only 3.5M parameters
- Fast: Depthwise separable convolutions
- Accurate: 71.3% ImageNet top-1 accuracy
- Mobile-ready: Can run on smartphones
- Transfer learning proven: Good for medical imaging

**Alternative**: ResNet50 (available in code)
- Deeper architecture: 50 layers
- Better accuracy: 76% ImageNet
- Larger: 102M parameters
- Slower inference

#### Feature Extraction Process
1. **Input**: Image resized to 224×224
2. **Preprocessing**: ImageNet normalization
3. **Forward pass**: Through MobileNetV2 (no top layers)
4. **Pooling**: Global average pool → 1280 dimensions
5. **Output**: Fixed-size deep feature vector

**Why deep features work**:
- Learned from 1.4M diverse images
- Captures high-level semantic features
- Better generalization than handcrafted
- Hierarchical: low-level edges → high-level patterns

### 3. Comparison: Handcrafted vs. Deep Features

| Aspect | Handcrafted | Deep |
|--------|-------------|------|
| Dimensions | 245 | 1280 |
| Extraction time | Fast (0.1s per image) | Slow (0.5s per image) |
| Interpretability | High (colors, texture) | Low (learned representations) |
| Accuracy (typical) | 75-80% | 85-90% |
| Robustness | Moderate | High (ImageNet pretrained) |
| Domain knowledge needed | Yes | No |
| GPU required | No | Optional (faster) |

---

## Model Implementations

### 1. k-Nearest Neighbors (kNN)

**Configuration**:
- `n_neighbors = 7`
- `weights = 'distance'`
- `metric = 'euclidean'` (default)

**How it works**:
1. Store all training samples
2. For each test sample, find 7 nearest neighbors
3. Vote using inverse distance weights
4. Class with highest weighted votes wins

**Pros**:
- Simple, non-parametric
- Works with any feature space
- No training time

**Cons**:
- Slow inference O(n)
- Sensitive to feature scaling
- Stores all training data

**When to use**:
- Baseline comparison
- Small datasets
- Need explainability (show nearest neighbors)

### 2. Gaussian Naive Bayes

**Configuration**:
- `var_smoothing = 1e-9` (default)

**How it works**:
1. Assume features follow Gaussian distribution per class
2. Estimate mean and variance from training data
3. Use Bayes theorem: P(class|features) ∝ P(features|class) × P(class)
4. Predict class with highest probability

**Pros**:
- Very fast training and inference
- Works with small datasets
- Probabilistic framework

**Cons**:
- Assumes feature independence (violated in images)
- Assumes Gaussian distribution
- Often underperforms on images

**When to use**:
- Quick baseline
- Very limited data
- Need probability estimates

**Why it underperforms on images**:
- Image features are highly correlated
- Independence assumption violated
- Gaussian assumption rarely holds

### 3. Support Vector Machine (SVM) with RBF

**Configuration**:
- `kernel = 'rbf'` (radial basis function)
- `C = 10.0` (regularization strength)
- `gamma = 'scale'` (kernel coefficient)
- `class_weight = 'balanced'` (for imbalanced classes)
- `probability = True` (for ROC curves)

**How it works**:
1. Find optimal hyperplane in high-dimensional feature space
2. RBF kernel maps features to even higher dimension
3. Maximize margin between classes
4. Use support vectors for decision boundary

**Pros**:
- Works well with high-dimensional features
- Good generalization (regularization)
- Handles non-linear patterns

**Cons**:
- Training O(n²) or O(n³)
- Hyperparameter tuning important
- Less interpretable

**When to use**:
- High-dimensional features (deep features)
- Medium-sized datasets
- Need good accuracy with reasonable speed

**Why RBF kernel**:
- Handles non-linear patterns
- Maps to infinite dimensions
- Good default for unknown data

### 4. Convolutional Neural Network (CNN)

**Architecture**:
```
Input (224×224×3)
    ↓
Data Augmentation Layer
    ↓
Rescaling (1/255)
    ↓
Conv2D (32 filters, 3×3, ReLU) + MaxPool2D
    ↓
Conv2D (64 filters, 3×3, ReLU) + MaxPool2D
    ↓
Conv2D (128 filters, 3×3, ReLU) + MaxPool2D
    ↓
Dropout (0.35)
    ↓
Flatten
    ↓
Dense (256, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (num_classes, Softmax)
```

**Data Augmentation**:
- Random rotation (±20°)
- Random width/height shift (±10%)
- Random zoom (±10%)
- Horizontal flip (50%)
- **Purpose**: Improve generalization, simulate real variations

**Training Configuration**:
- `optimizer = Adam(lr=1e-3)`
- `loss = sparse_categorical_crossentropy`
- `batch_size = 32`
- `epochs = 25` (with early stopping)
- **Callbacks**:
  - Early stopping (patience=5)
  - Learning rate reduction (factor=0.5, patience=3)

**Why this architecture works**:
1. **Convolutional layers**: Learn local patterns (edges, textures)
2. **Pooling**: Reduce dimensionality, invariance
3. **Multiple layers**: Hierarchical feature learning
4. **Dropout**: Prevent overfitting
5. **Regularization**: L2 implicit in optimization

**Advantages**:
- Automatic feature learning
- Task-specific optimization
- State-of-the-art accuracy
- End-to-end training

**Disadvantages**:
- Slow training
- Requires large dataset
- Requires GPU for fast training
- Black box (less interpretable)

---

## Evaluation Metrics

### Classification Metrics

#### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- **Meaning**: Fraction of correct predictions
- **Range**: 0-1
- **When to use**: Balanced datasets
- **Limitation**: Misleading on imbalanced data

#### Precision
```
Precision = TP / (TP + FP)
```
- **Meaning**: When model predicts positive, how often correct?
- **Focus**: False positive cost matters
- **Example**: Disease detection (false alarms costly)

#### Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
- **Meaning**: Of actual positives, how many detected?
- **Focus**: False negative cost matters
- **Example**: Disease detection (missing disease is critical)

#### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- **Meaning**: Harmonic mean of precision and recall
- **Use**: Balanced metric for imbalanced classes
- **Range**: 0-1

**For multiclass (weighted average)**:
- Average of each class metric
- Weighted by class support
- Fair comparison with imbalanced dataset

### Confusion Matrix

**For each class**:
- **True Positives (TP)**: Correctly predicted as this class
- **False Positives (FP)**: Incorrectly predicted as this class
- **True Negatives (TN)**: Correctly predicted as other class
- **False Negatives (FN)**: Missed examples of this class

**Interpretation**:
- Diagonal = correct predictions
- Off-diagonal = confusion between classes
- Can identify which classes are confused

### ROC Curve (Receiver Operating Characteristic)

**For multiclass**:
- One-vs-Rest approach
- One ROC curve per class
- X-axis: False Positive Rate
- Y-axis: True Positive Rate

**AUC (Area Under Curve)**:
- Range: 0-1
- 0.5: Random guessing
- 1.0: Perfect classifier
- **Interpretation**: Probability that model ranks random positive higher than random negative

---

## Training Details

### Hyperparameter Selection

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Image size | 224×224 | MobileNetV2 standard; computational efficiency |
| Batch size | 32 | Balance between memory and gradient stability |
| CNN epochs | 25 | Usually sufficient; early stopping prevents overfitting |
| kNN neighbors | 7 | Odd number; avoid ties; typical range 3-10 |
| SVM C | 10.0 | Moderate regularization; empirically good |
| Learning rate | 1e-3 | Standard for Adam optimizer |
| Dropout rate | 0.5 (dense), 0.35 (conv) | Prevent overfitting; deeper layers stronger regularization |

### Optimization Strategies

1. **Early Stopping**: Stop training if validation accuracy doesn't improve for 5 epochs
2. **Learning Rate Reduction**: Reduce LR by 50% if validation loss plateaus
3. **Class Weighting**: Balanced class weights for SVM (handle imbalanced data)
4. **Data Augmentation**: Random transforms prevent overfitting
5. **Cross-validation**: Stratified split maintains class distribution

---

## Performance Predictions

### Typical Results on Mango Leaf Dataset

| Model | Accuracy | Speed | Notes |
|-------|----------|-------|-------|
| kNN (handcrafted) | 70-75% | Fast | Baseline |
| Naive Bayes | 60-65% | Fastest | Independence assumption violated |
| SVM (handcrafted) | 75-80% | Medium | Good classical baseline |
| SVM (deep) | 85-90% | Slow* | Transfer learning benefit |
| CNN | **90-95%+** | Medium | Best accuracy |

*Slow due to MobileNetV2 extraction

### Expected Confusion Patterns

1. **Healthy vs. Disease**: Often well-separated
2. **Similar disease types**: May be confused
3. **Early vs. late stage**: Can be challenging
4. **Dark/shadow areas**: Misclassified as disease

---

## Computational Requirements

### Time Complexity

| Phase | Time | Notes |
|-------|------|-------|
| Feature extraction (handcrafted) | ~0.1s/image | Vectorized NumPy |
| Feature extraction (deep) | ~0.5s/image | GPU ~5x faster |
| Train kNN | ~0 | Non-parametric |
| Train Naive Bayes | ~0.1s | Fast |
| Train SVM | ~1-10m | O(n²) complexity |
| Train CNN | ~30-60m | 50-200 batches × epoch |

### Space Complexity

| Model | Memory |
|-------|--------|
| kNN | O(n×d) - stores all data |
| Naive Bayes | O(c×d) - per-class statistics |
| SVM | O(s×d) - support vectors |
| CNN | ~5MB model weights |
| Dataset (224×224×3) | ~150MB for 1000 images |

---

## Production Deployment

### Model Serving

```python
# Load models
from src.features import load_image, handcrafted_features
import joblib

model = joblib.load('models/svm_handcrafted.joblib')
scaler = joblib.load('models/handcrafted_scaler.joblib')

# Predict
features = handcrafted_features('leaf.jpg')
features_scaled = scaler.transform([features])
prediction = model.predict(features_scaled)
```

### REST API (Flask example)

```python
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    features = handcrafted_features(file)
    prediction = model.predict([features_scaled])
    return {'disease': class_names[prediction[0]]}
```

### Batch Processing

```bash
python evaluate_models.py --batch /path/to/leaves/
```

---

## References & Further Reading

### Papers
- MobileNetV2: Efficient CNNs for Mobile Vision Applications
- Haralick Texture Features (GLCM)
- Support Vector Machines for Image Classification
- Transfer Learning in Computer Vision

### Libraries Used
- TensorFlow/Keras: Deep learning
- scikit-learn: Classical ML
- OpenCV: Image processing
- NumPy/Pandas: Data manipulation
- Matplotlib/Seaborn: Visualization

---

**Last Updated**: April 2026
