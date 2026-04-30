# 🔧 WINDOWS LONG PATH FIX - Complete Guide

**Your Error**: `OSError: [Errno 2] No such file or directory` with long file paths

This is a Windows limitation. File paths longer than 260 characters fail unless Long Path support is enabled.

---

## ⚡ QUICK FIX (5 minutes)

### Step 1: Enable Long Paths (Windows Registry)

**IMPORTANT**: Run as Administrator!

1. Press `Win + R` (Windows key + R)
2. Type: `regedit` and press Enter
3. A Registry Editor window opens
4. Navigate to this path (copy-paste into address bar):
   ```
   HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
   ```
5. Look for `LongPathsEnabled` in the right panel
6. **If found**: Double-click it and change value to `1`
7. **If NOT found**: Right-click empty space → New → DWORD (32-bit) Value
   - Name: `LongPathsEnabled`
   - Value: `1`
   - Click OK

8. Close Registry Editor
9. **RESTART YOUR COMPUTER** (very important!)

### Step 2: Clean Up and Reinstall

```batch
REM Open Command Prompt as Administrator
REM Navigate to project:

cd C:\Desktop\week1\AI-Project\finalproject

REM Delete old environment
rmdir /s /q venv

REM Create fresh environment
python -m venv venv

REM Activate it
venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install packages (one group at a time)
pip install numpy pandas matplotlib seaborn pillow
pip install opencv-python scikit-image scikit-learn joblib tqdm
pip install tensorflow keras
pip install jupyter ipykernel
```

### Step 3: Run It

```batch
REM Make sure venv is activated
venv\Scripts\activate.bat

REM Start Jupyter
python -m jupyter notebook
```

---

## 🆘 If Quick Fix Didn't Work

Try these (in order):

### Option 1: Use Automated Python Setup

```batch
cd C:\Desktop\week1\AI-Project\finalproject
python setup_auto.py
```

### Option 2: Use PowerShell Instead

```powershell
# Right-click PowerShell and select "Run as Administrator"

# Navigate to project
cd C:\Desktop\week1\AI-Project\finalproject

# Run setup
.\setup.ps1
```

### Option 3: Install to Short Path

Move project to root of a drive (shorter path):

```batch
REM Move to C:\ml_project (shorter than current path)
REM Then setup from there
```

### Option 4: Disable Antivirus

Windows Defender or other antivirus might block long paths. Try:
- Temporarily disable antivirus
- Run setup again
- Re-enable antivirus

### Option 5: Fresh Python Installation

If all else fails:
1. Uninstall Python completely (Programs and Features)
2. Download Python 3.11 from https://www.python.org/
3. Run installer with these settings:
   - ✓ "Add Python to PATH"
   - ✓ "Install for all users"
4. Install to `C:\Python311` (not Program Files to avoid spaces)
5. Try setup again

---

## 🎯 Complete Step-by-Step Setup

**Once Long Paths are fixed**, follow this:

```batch
REM 1. Open Command Prompt as Administrator
REM    (Press Win + X, select "Command Prompt (Admin)")

REM 2. Navigate to project
cd C:\Desktop\week1\AI-Project\finalproject

REM 3. Check Python
python --version

REM 4. Create environment
python -m venv venv

REM 5. Activate
venv\Scripts\activate.bat

REM 6. You should see (venv) at start of command line

REM 7. Upgrade pip
python -m pip install --upgrade pip

REM 8. Install core packages
pip install numpy pandas matplotlib seaborn

REM 9. Install image libraries
pip install pillow opencv-python scikit-image scikit-learn

REM 10. Install ML libraries
pip install joblib tqdm tensorflow keras

REM 11. Install Jupyter
pip install jupyter ipykernel

REM 12. Start Jupyter
python -m jupyter notebook

REM 13. Browser will open automatically
REM 14. Navigate to: notebooks/mango_leaf_disease_classification.ipynb
REM 15. Click it to open
REM 16. Click "Run All" button or run cells individually
```

---

## ✅ How to Tell It's Working

You'll see:
```
✓ (venv) appears in command line
✓ pip commands complete successfully
✓ Jupyter starts and opens browser
✓ Notebook loads without errors
```

---

## 🚀 Test Everything Works

```batch
REM With venv activated, test:

REM 1. Import TensorFlow
python -c "import tensorflow as tf; print(f'TF version: {tf.__version__}')"

REM 2. Import scikit-learn
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"

REM 3. Start Jupyter
python -m jupyter notebook
```

All should work without errors!

---

## 📞 Still Stuck?

Try these alternative approaches:

### Use Google Colab (Online, No Setup)
1. Go to: https://colab.research.google.com/
2. Click "Upload"
3. Upload: `notebooks/mango_leaf_disease_classification.ipynb`
4. Run in cloud (automatic setup!)

### Use WSL (Windows Subsystem for Linux)
- No Windows path limitations
- Better Python support
- Professional development environment
- Setup guide: https://learn.microsoft.com/en-us/windows/wsl/

### Use Docker
- Pre-configured environment
- Zero setup needed
- Same setup every time

### Run Without Jupyter
```batch
REM Instead of Jupyter, run Python directly:
python run_project.py
```

---

## 🔍 Debug: Find Long Paths

Check if your file paths exceed 260 characters:

```batch
cd C:\Desktop\week1\AI-Project\finalproject
REM Path should be around 50 characters - OK

REM Check venv paths:
dir venv\Lib\site-packages\* | find /v "." | more
REM If many files = long paths issue
```

---

## 📋 Checklist Before Setup

- [ ] Windows 10 or newer
- [ ] Python 3.9+ installed
- [ ] 5 GB disk space available
- [ ] Admin access on computer
- [ ] 8 GB+ RAM available
- [ ] Internet connection (for downloading packages)

---

## ⏱ Estimated Times

- Enable Long Paths: 5 minutes (mostly restart)
- Clean venv: 2 minutes
- Install packages: 10-15 minutes (first time, slower)
- Total: **25-30 minutes**

---

## 🎯 Success Indicators

After setup completes, you should have:

```
finalproject/
├── venv/                  # Virtual environment (1+ GB)
├── notebooks/
│   └── mango_leaf_disease_classification.ipynb
├── data/
│   ├── raw/             # Dataset images go here
│   └── processed/
├── models/              # Models saved here
├── results/             # Visualizations saved here
├── src/                 # Python modules
└── requirements.txt
```

---

## 🚀 Once Everything Works

```batch
REM Start the notebook
python -m jupyter notebook

REM In browser:
# 1. Click: notebooks/mango_leaf_disease_classification.ipynb
# 2. Notebook opens
# 3. Click "Kernel" → "Restart & Run All"
# 4. Wait 10-20 minutes for completion
# 5. Check "results/" folder for generated visualizations
```

---

**Created**: 2026-04-28  
**Purpose**: Fix Windows Long Path issues for Python package installation  
**Status**: Complete setup guide with multiple solutions