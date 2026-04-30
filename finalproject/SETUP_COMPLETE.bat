@echo off
REM ============================================================================
REM MANGO LEAF DISEASE CLASSIFICATION - COMPLETE SETUP
REM Windows Long Path Fix + Full Project Setup
REM ============================================================================
REM
REM IMPORTANT: Run this file as Administrator!
REM     Right-click on file → Run as Administrator
REM
REM ============================================================================

setlocal enabledelayedexpansion

cls
echo.
echo ============================================================================
echo   MANGO LEAF DISEASE CLASSIFICATION - COMPLETE SETUP
echo ============================================================================
echo.
echo This script will:
echo   1. Enable Windows Long Path support
echo   2. Create virtual environment
echo   3. Install all dependencies
echo   4. Create project directories
echo.
echo IMPORTANT: You may need to RESTART your computer after this!
echo.

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script requires Administrator privileges!
    echo.
    echo Please:
    echo   1. Right-click this file
    echo   2. Select "Run as Administrator"
    echo   3. Click "Yes" when prompted
    echo.
    pause
    exit /b 1
)

echo ✓ Running as Administrator
echo.

REM Enable Long Path Support
echo [STEP 1/5] Enabling Windows Long Path support...
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f >nul 2>&1
if %errorLevel% equ 0 (
    echo ✓ Long Path support enabled
    echo.
    echo ⚠ You should RESTART your computer after setup for changes to take effect!
    echo.
) else (
    echo ⚠ Could not enable Long Path support (might be already enabled)
    echo.
)

REM Check Python
echo [STEP 2/5] Checking Python installation...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please:
    echo   1. Download Python from: https://www.python.org/
    echo   2. Install with "Add Python to PATH" checked
    echo   3. Run this script again
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✓ %PYTHON_VERSION% found
echo.

REM Remove old venv if exists
echo [STEP 3/5] Preparing virtual environment...
if exist venv (
    echo Removing old virtual environment...
    rmdir /s /q venv >nul 2>&1
    timeout /t 2 /nobreak >nul
)

echo Creating new virtual environment...
python -m venv venv
if %errorLevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment created
echo.

REM Activate venv
call venv\Scripts\activate.bat
if %errorLevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated
echo.

REM Upgrade pip
echo [STEP 4/5] Installing packages...
echo.
echo [4a] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel --quiet
echo ✓ pip upgraded
echo.

echo [4b] Installing data science libraries...
pip install numpy pandas matplotlib seaborn pillow --quiet
if %errorLevel% equ 0 (
    echo ✓ Data science libraries installed
) else (
    echo ⚠ Data science libraries had issues (continuing)
)
echo.

echo [4c] Installing image processing libraries...
pip install opencv-python scikit-image scikit-learn joblib tqdm --quiet
if %errorLevel% equ 0 (
    echo ✓ Image processing libraries installed
) else (
    echo ⚠ Image processing libraries had issues (continuing)
)
echo.

echo [4d] Installing deep learning (TensorFlow/Keras)...
echo       This might take a few minutes...
pip install tensorflow keras --quiet
if %errorLevel% equ 0 (
    echo ✓ Deep learning libraries installed
) else (
    echo ⚠ Deep learning libraries had issues (continuing)
)
echo.

echo [4e] Installing Jupyter...
pip install jupyter ipykernel --quiet
if %errorLevel% equ 0 (
    echo ✓ Jupyter installed
) else (
    echo ✗ Jupyter installation failed
    echo   Try: pip install --upgrade jupyter ipykernel
)
echo.

REM Create directories
echo [STEP 5/5] Creating project directories...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist models mkdir models
if not exist results mkdir results
echo ✓ Directories created
echo.

REM Success message
echo.
echo ============================================================================
echo ✓ SETUP COMPLETED SUCCESSFULLY!
echo ============================================================================
echo.
echo IMPORTANT NEXT STEPS:
echo.
echo 1. RESTART YOUR COMPUTER
echo    (This is essential for Long Path changes to take effect!)
echo.
echo 2. Download the dataset:
echo    Visit: https://www.kaggle.com/datasets/warcoder/mango-leaf-disease-dataset
echo    Extract to: finalproject/data/raw/
echo.
echo 3. Start the project by running THIS script again, then:
echo.
echo    After restart, to start Jupyter:
echo    1. Open Command Prompt
echo    2. Navigate to this folder: cd C:\Desktop\week1\AI-Project\finalproject
echo    3. Run: python -m jupyter notebook
echo    4. Open: notebooks/mango_leaf_disease_classification.ipynb
echo    5. Click "Run All" button
echo.
echo TROUBLESHOOTING:
echo.
echo If you get an error about (venv) not being recognized:
echo   - Make sure you're in the correct directory
echo   - Manually run: venv\Scripts\activate.bat
echo.
echo If Jupyter is not found:
echo   - Manually run: pip install --upgrade jupyter
echo.
echo If you get long path errors:
echo   - Restart your computer (very important!)
echo   - Run setup again as Administrator
echo.
echo For detailed help, see: FIX_WINDOWS_PATHS_GUIDE.md
echo.
pause
