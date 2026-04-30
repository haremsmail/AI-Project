# 🥭 MANGO LEAF DISEASE CLASSIFIER - SIMPLE VERSION

**Beginner-friendly program to classify mango leaf diseases using AI**

---

## 🚀 QUICK START (One Command!)

```powershell
python main.py
```

That's it! A menu will appear.

---

## 📋 MENU OPTIONS

When you run `python main.py`, you'll see:

```
🥭 MANGO LEAF DISEASE CLASSIFIER
==================================================

1 - Train all models
2 - Test on single image  
3 - Compare model accuracy
4 - Exit

Select option (1-4):
```

---

## 🎯 WHAT EACH OPTION DOES

### **Option 1: Train All Models**
- Trains 4 AI models:
  - **kNN** (k-Nearest Neighbors)
  - **Naive Bayes** 
  - **SVM** (Support Vector Machine)
  - **CNN** (Convolutional Neural Network / Deep Learning)
- Saves trained models automatically
- Shows accuracy for each model
- Takes 5-10 minutes (depending on your PC)

### **Option 2: Test on Single Image**
- Enter the path to a mango leaf image
- The AI will predict the disease
- Shows:
  - **Disease detected**: Healthy, Powdery Mildew, etc.
  - **Confidence**: How certain is the prediction (%)
  - **Best Model**: Which AI model made the prediction

Example:
```
Enter image path: C:\Users\YourName\leaf.jpg

PREDICTION RESULT
  Disease/Status: Powdery Mildew
  Confidence: 94%
  Best Model Used: SVM
```

### **Option 3: Compare Model Accuracy**
- Shows a table comparing all trained models
- Displays: Accuracy, Precision, Recall, F1-Score
- Shows which model performs best

Example:
```
Model            Accuracy  Precision  Recall    F1-Score
------------------------------------------------------------
kNN              95.2%     95.1%      95.2%     95.1%
Naive Bayes      92.1%     91.8%      92.1%     91.9%
SVM              97.3%     97.2%      97.3%     97.2%
CNN              94.8%     94.6%      94.8%     94.7%

🏆 BEST MODEL: SVM (97.3% accuracy)
```

### **Option 4: Exit**
- Closes the program

---

## 📊 DATASET

The program automatically uses the dataset at:
```
data/raw/
```

It detects these mango leaf disease categories:
- **Anthracnose** - Fungal disease
- **Bacterial Canker** - Bacterial disease
- **Cutting Weevil** - Insect damage
- **Die Back** - Plant deterioration
- **Gall Midge** - Insect damage
- **Healthy** - No disease
- **Powdery Mildew** - Fungal disease
- **Sooty Mould** - Fungal disease

---

## 🔧 SETUP (First Time Only)

### 1. Install Python (if not already installed)
- Download from: https://www.python.org/downloads/
- Make sure to check "Add Python to PATH"

### 2. Create Virtual Environment
```powershell
cd c:\Desktop\week1\AI-Project\finalproject
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Download Dataset (Optional)
```powershell
python setup_dataset.py
```

---

## ✅ REQUIREMENTS

All required packages are listed in `requirements.txt`:
- tensorflow (Deep Learning)
- scikit-learn (Machine Learning)
- numpy, pandas (Data Processing)
- opencv-python (Image Processing)
- matplotlib, seaborn (Visualization)
- joblib (Model Saving)
- And more...

Install with:
```powershell
pip install -r requirements.txt
```

---

## 📁 PROJECT STRUCTURE

```
finalproject/
├── main.py                  ← RUN THIS FILE
├── requirements.txt         ← All dependencies
├── src/
│   ├── config.py           ← Settings
│   ├── utils.py            ← Utilities
│   ├── features.py         ← Feature extraction
│   ├── models.py           ← Model training
│   └── visualization.py    ← Charts & plots
├── data/
│   └── raw/                ← Dataset location
├── models/                 ← Saved trained models
└── results/                ← Output results
```

---

## 💡 TIPS

1. **First time?** Start with Option 1 to train models
2. **Takes too long?** Check your internet (downloading models)
3. **Have an image?** Try Option 2 to test it
4. **Want to compare?** Use Option 3 to see model performance

---

## 🎓 LEARNING OUTCOME

You'll learn:
- How AI classifies images
- How different machine learning models compare
- Feature extraction from images
- Model training and evaluation
- Agricultural applications of AI

---

## 📞 TROUBLESHOOTING

### Error: "Dataset not found"
- Make sure dataset is in: `data/raw/`
- Run: `python setup_dataset.py`

### Error: "Models not trained"
- Run Option 1 first to train models
- Models are saved in: `models/`

### Error: "Image not found"
- Make sure the image path is correct
- Use full path like: `C:\Users\Name\image.jpg`

### Slow performance?
- Large dataset takes time to process (5-10 minutes first run)
- Subsequent runs are faster

---

## 📝 AUTHOR

Created as an educational project for Mango Leaf Disease Classification

---

## 🎯 GOAL

**Simple AI for Everyone:**
- No coding knowledge needed
- Just run one command
- Get results immediately

---

**Enjoy! 🥭**
