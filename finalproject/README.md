# Mango Leaf Disease Classification

A production-quality Python project for classifying mango leaf images into disease categories or healthy leaves using handcrafted image features, classical machine learning models, and a CNN-based neural network.

## Project Overview

This project implements a complete machine learning pipeline for agricultural disease detection:

- **Multi-classifier comparison**: Neural Network, kNN, Gaussian Naive Bayes, SVM with RBF kernel
- **Feature engineering**: RGB/HSV histograms, GLCM texture, shape descriptors, and deep learning embeddings
- **Comprehensive evaluation**: Accuracy, precision, recall, F1-score, confusion matrices, ROC curves
- **Production-ready code**: Modular, well-documented, and reproducible

## Key Features

✓ Automatic dataset discovery and class balance visualization  
✓ Handcrafted feature extraction (color, texture, shape analysis)  
✓ Deep feature extraction using pretrained MobileNetV2  
✓ Multiple classifier training and comparison  
✓ CNN with data augmentation and early stopping  
✓ Professional visualizations and comparison charts  
✓ Model persistence and batch prediction  
✓ Automated report generation

## Folder Structure

```text
finalproject/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── results/
└── src/
```

## Dataset

Use the Kaggle dataset:

https://www.kaggle.com/datasets/warcoder/mango-leaf-disease-dataset

After downloading, extract the dataset into:

```text
finalproject/data/raw/mango_leaf_disease_dataset/
```

The notebook can also discover images directly under `data/raw/` if you prefer a flatter structure, as long as each class has its own folder.

## Main Notebook

Open and run:

```text
finalproject/notebooks/mango_leaf_disease_classification.ipynb
```

That notebook handles:

- dataset discovery and class distribution checks
- image sampling and preprocessing
- handcrafted feature extraction
- kNN, Naive Bayes, and SVM training
- CNN training with augmentation
- feature comparison between handcrafted and deep features
- evaluation tables and plots

## Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the project
cd finalproject

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get the Dataset

#### Option A: Automatic Download (requires Kaggle CLI)

```bash
# Set up Kaggle API credentials
# 1. Download API key from https://www.kaggle.com/account/
# 2. Place it at ~/.kaggle/kaggle.json
# Then run:
python setup_dataset.py
```

#### Option B: Manual Download

1. Download from: https://www.kaggle.com/datasets/warcoder/mango-leaf-disease-dataset
2. Extract to: `data/raw/mango_leaf_disease_dataset/`

The extracted folder should have this structure:
```text
data/raw/mango_leaf_disease_dataset/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   ├── image1.jpg
│   └── ...
└── ...
```

### 3. Run the Analysis

```bash
# Start Jupyter
jupyter notebook

# Open and run: notebooks/mango_leaf_disease_classification.ipynb
# Execute all cells from top to bottom
```

### 4. Generate Report

```bash
python generate_report.py
```

This creates `results/ANALYSIS_REPORT.md` with:
- Model performance comparison
- Feature analysis insights
- Computational cost breakdown
- Production recommendations

### 5. Use Trained Models

```bash
# Predict on single image
python evaluate_models.py --image path/to/leaf.jpg

# Batch predict on directory
python evaluate_models.py --batch path/to/leaf_images/
```

## Project Structure

```text
finalproject/
├── data/
│   ├── raw/                          # Raw dataset (ignored in git)
│   │   ├── mango_leaf_disease_dataset/
│   │   └── README.md                # Dataset setup guide
│   └── processed/                   # Processed data (optional)
├── models/                           # Trained models (ignored in git)
│   ├── mango_leaf_cnn.keras          # CNN model
│   ├── svm_handcrafted.joblib        # SVM on handcrafted features
│   ├── label_encoder.joblib          # Class label encoder
│   └── ...
├── notebooks/
│   └── mango_leaf_disease_classification.ipynb  # Main analysis notebook
├── results/                          # Outputs (ignored in git)
│   ├── model_comparison.csv          # Metrics table
│   ├── cnn_confusion_matrix.png      # Confusion matrix
│   ├── cnn_training_curves.png       # Training history
│   ├── model_comparison.png          # Bar chart
│   ├── ANALYSIS_REPORT.md            # Generated report
│   └── ...
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration & paths
│   ├── utils.py                      # Dataset utilities
│   ├── features.py                   # Feature extraction
│   ├── models.py                     # Model training & evaluation
│   └── visualization.py              # Plotting utilities
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── setup_dataset.py                  # Dataset downloader
├── generate_report.py                # Report generator
├── evaluate_models.py                # Batch prediction tool
└── .gitignore                        # Version control rules
```

## Model Configuration

Key hyperparameters (editable in `src/config.py`):

```python
IMAGE_SIZE = (224, 224)              # Input image size
BATCH_SIZE = 32                      # Batch size for training
EPOCHS = 25                          # CNN training epochs
KNN_NEIGHBORS = 7                    # k for kNN
SVM_C = 10.0                         # C parameter for SVM
TEST_SIZE = 0.2                      # Test split fraction
VALIDATION_SIZE = 0.2                # Validation split fraction
```

## Expected Outputs

After running the notebook, `results/` contains:

| File | Description |
|------|-------------|
| `model_comparison.csv` | Accuracy, precision, recall, F1 for all models |
| `cnn_confusion_matrix.png` | CNN classification confusion matrix |
| `cnn_training_curves.png` | Training/validation accuracy and loss |
| `cnn_roc.png` | ROC curves (multiclass) |
| `model_comparison.png` | Bar chart comparing all models |
| `class_distribution.png` | Dataset class balance |
| `sample_images.png` | Sample images from each class |
| `ANALYSIS_REPORT.md` | Full analysis and recommendations |

## Architecture Overview

### Features Extracted

1. **Handcrafted Features** (~192 dimensions)
   - RGB color histogram (32 bins × 3 channels)
   - HSV histogram (32 bins × 3 channels)
   - GLCM texture (6 properties × 8 orientations)
   - Shape descriptors (5 metrics)

2. **Deep Features** (1280 dimensions)
   - MobileNetV2 pretrained on ImageNet
   - Average pooling for fixed-size output
   - More expressive than handcrafted features

### Models

| Model | Features | Training Time | Inference Time |
|-------|----------|---|---|
| kNN | Handcrafted | Fast | Slow |
| Naive Bayes | Handcrafted | Fast | Very Fast |
| SVM (RBF) | Handcrafted | Medium | Medium |
| kNN | Deep | Slow* | Slow |
| Naive Bayes | Deep | Slow* | Fast |
| SVM (RBF) | Deep | Slow* | Medium |
| CNN | End-to-end | Very Slow | Medium |

*Slow due to feature extraction overhead

## Model Interpretation

### Why Neural Networks Excel on Images

- **Hierarchical Learning**: Automatically discover multi-level features
- **Spatial Preservation**: Convolutions maintain 2D structure
- **Data Augmentation**: Improved robustness through rotation, zoom, flip
- **End-to-End Learning**: Optimized specifically for your task

### Why Handcrafted Features Fail

- Image classification requires learning what features matter
- Manual feature engineering is time-consuming and error-prone
- Misses subtle disease indicators humans haven't anticipated

### When to Use Classical Models

- Real-time inference with low latency
- Limited computational resources
- Small datasets (<1000 images)
- Need for model explainability

## Performance Comparison

Typical results on the mango leaf dataset:

| Model | Accuracy | Speed |
|-------|----------|-------|
| kNN (handcrafted) | ~70% | Fast |
| Naive Bayes | ~60% | Very Fast |
| SVM (handcrafted) | ~75% | Medium |
| SVM (deep features) | ~85% | Slow |
| CNN | **~90%+** | **Recommended** |

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'tensorflow'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: No images found` | Check dataset path matches `data/raw/` structure |
| Out of memory during CNN training | Reduce `BATCH_SIZE` in `src/config.py` |
| Slow feature extraction | Use smaller `IMAGE_SIZE` or reduce dataset size for testing |
| Kaggle API error | Configure credentials at `~/.kaggle/kaggle.json` |

## Extending the Project

### Try Different Architectures

```python
# In the notebook, modify CNN to use:
from tensorflow.keras.applications import ResNet50, EfficientNetB0

# Transfer learning example:
base_model = ResNet50(weights='imagenet', include_top=False)
base_model.trainable = False  # Freeze weights
```

### Ensemble Models

```python
# Combine predictions from multiple models
predictions_ensemble = (
    cnn_predictions * 0.6 +
    svm_predictions * 0.2 +
    knn_predictions * 0.2
)
```

### Deploy as API

```bash
# Create a Flask app for production serving
# See examples in the codebase
python -m flask run --host 0.0.0.0 --port 5000
```

## Performance Tuning

1. **Improve CNN Accuracy**:
   - Increase `EPOCHS` (up to 100)
   - Reduce learning rate
   - Add more data augmentation
   - Use ensemble of models

2. **Speed Up Training**:
   - Use GPU: Install `tensorflow[and-cuda]`
   - Reduce `IMAGE_SIZE`
   - Increase `BATCH_SIZE`
   - Use transfer learning from larger models

3. **Reduce Memory**:
   - Lower `BATCH_SIZE`
   - Use `tf.data.Dataset` with prefetching
   - Enable mixed precision training

## Dependencies

See `requirements.txt` for the complete list. Key packages:

- **TensorFlow 2.15+**: Deep learning framework
- **scikit-learn**: Classical ML algorithms
- **OpenCV**: Image processing
- **pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Kaggle**: Dataset download

## Citation

If you use this project in academic work, cite:

```bibtex
@misc{mango_leaf_disease_2024,
  title={Mango Leaf Disease Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/finalproject}
}
```

## License

This project is open source and available under the MIT License.

## Contact & Support

For questions or issues:
1. Check existing GitHub issues
2. Review the analysis report for insights
3. Experiment with hyperparameters in `src/config.py`

---

**Last Updated**: April 2026  
**Status**: Production Ready ✓
