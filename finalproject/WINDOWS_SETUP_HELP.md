# Windows Setup Troubleshooting Guide

## Problem: "OSError: [Errno 2] No such file or directory" with Long Path

This occurs when Windows has **Long Path support disabled** and file paths exceed 260 characters.

---

## ✅ Quick Fix (3 Steps)

### Step 1: Enable Windows Long Path Support

**Option A: Automatic (Requires Admin)**
```batch
REM Run as Administrator
FIX_WINDOWS_PATHS.bat
REM Then RESTART your computer
```

**Option B: Manual Registry Edit**
1. Press `Win + R` and type: `regedit`
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Look for: `LongPathsEnabled` value
4. If it exists, set it to: `1`
5. If it doesn't exist:
   - Right-click → New → DWORD (32-bit) Value
   - Name it: `LongPathsEnabled`
   - Set value to: `1`
6. Click OK and close Registry Editor
7. **RESTART YOUR COMPUTER**

### Step 2: Delete and Recreate Virtual Environment

```batch
REM Close any open Python/Jupyter windows first
rmdir /s /q venv
python -m venv venv
```

### Step 3: Use Simplified Setup

**Option A: Automated Setup**
```batch
python setup_auto.py
```

**Option B: Step by Step**
```batch
REM Activate environment
venv\Scripts\activate.bat

REM Install in smaller batches
pip install numpy pandas matplotlib seaborn pillow
pip install opencv-python scikit-image scikit-learn joblib tqdm
pip install tensorflow keras
pip install jupyter ipykernel
```

---

## 🚀 After Fixing

### Start the Project

```batch
REM Activate environment
venv\Scripts\activate.bat

REM Option 1: Run Jupyter Notebook
python -m jupyter notebook

REM Option 2: Run main script
python run_project.py
```

### Open Notebook

1. Jupyter will open in your browser automatically
2. Navigate to: `notebooks/mango_leaf_disease_classification.ipynb`
3. Click on it to open
4. Click "Run All" to execute all cells

---

## 📋 Complete Setup Procedure

### Fresh Start (Recommended)

```batch
REM 1. Navigate to project
cd C:\Desktop\week1\AI-Project\finalproject

REM 2. Fix Windows paths (as Administrator)
FIX_WINDOWS_PATHS.bat
REM Then RESTART your computer

REM 3. Create clean virtual environment
rmdir /s /q venv 2>nul
python -m venv venv

REM 4. Activate
venv\Scripts\activate.bat

REM 5. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

REM 6. Install core dependencies
pip install numpy pandas matplotlib seaborn pillow

REM 7. Install image processing
pip install opencv-python scikit-image scikit-learn joblib tqdm

REM 8. Install deep learning
pip install tensorflow keras

REM 9. Install Jupyter
pip install jupyter ipykernel

REM 10. Create directories
mkdir data\raw data\processed models results

REM 11. Run project
python -m jupyter notebook
```

---

## 🔧 Troubleshooting

### Still Getting Long Path Errors?

**Try these in order:**

1. **Restart computer** (most important!)
   ```batch
   shutdown /r /t 30
   ```

2. **Run as Administrator**
   - Right-click Command Prompt → Run as Administrator
   - Then run setup commands

3. **Use simpler installation**
   ```batch
   pip install --no-cache-dir numpy pandas matplotlib seaborn pillow opencv-python scikit-learn tensorflow
   ```

4. **Install to different drive** (if C: is full)
   ```batch
   REM Move project to D: or E: drive
   REM Try setup again
   ```

5. **Disable antivirus temporarily**
   - Windows Defender or antivirus might block long paths
   - Temporarily disable and retry setup

6. **Clean Python installation**
   ```batch
   REM Completely uninstall Python from Programs and Features
   REM Download fresh from https://www.python.org/
   REM Reinstall with "Add Python to PATH" checked
   REM Then setup project again
   ```

### Jupyter Not Found?

```batch
REM After activating venv, try:
python -m jupyter notebook

REM If still fails:
pip install --upgrade jupyter ipykernel
python -m jupyter notebook
```

### Can't Activate Virtual Environment?

**Windows:**
```batch
REM Try explicit path
C:\Desktop\week1\AI-Project\finalproject\venv\Scripts\activate.bat
```

**Or use Python directly:**
```batch
python -m pip install --upgrade pip
REM Then install packages directly
```

---

## 💡 Prevention Tips

1. **Keep paths short** - Avoid deeply nested directories
2. **Update Windows** - Windows 10 v1607+ supports long paths
3. **Use Python 3.9+** - Better long path support
4. **Consider WSL** - Windows Subsystem for Linux (no path issues)
5. **Use PowerShell** - Sometimes better than Command Prompt

---

## ✅ Success Indicators

After successful setup, you should see:

```
✓ Virtual environment created
✓ Packages installed
✓ Jupyter starts successfully
✓ Notebook opens in browser
✓ Code cells execute without errors
```

---

## 📞 Still Not Working?

Try completely different approach:

### Option 1: Use Python Directly
```batch
python run_project.py
```

### Option 2: Use Google Colab (Online, No Setup)
1. Go to https://colab.research.google.com/
2. Click "Upload" 
3. Upload `notebooks/mango_leaf_disease_classification.ipynb`
4. Run in cloud (no local setup needed)

### Option 3: Use WSL (Windows Subsystem for Linux)
- No path length limits
- Better Python support
- Professional development environment

---

**Last Updated**: 2026-04-28

For more help, see: `README.md` or `QUICKSTART.md`