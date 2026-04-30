@echo off
REM Setup script for Mango Leaf Disease Classification Project
REM Simplified version to avoid long path issues

echo.
echo ======================================
echo Mango Leaf Classification Project Setup
echo ======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

echo ✓ Python found: 
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo ✓ Virtual environment already exists
) else (
    python -m venv venv
    if %errorLevel% equ 0 (
        echo ✓ Virtual environment created
    ) else (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo ✓ Virtual environment activated
echo.

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel --quiet
echo ✓ pip upgraded
echo.

REM Install dependencies in batches to avoid long path issues
echo Installing dependencies (this may take 5-10 minutes)...
echo.

echo [1/4] Installing data science libraries...
pip install numpy pandas matplotlib seaborn pillow --quiet
if %errorLevel% neq 0 goto :error

echo [2/4] Installing image processing libraries...
pip install opencv-python scikit-image scikit-learn joblib tqdm --quiet
if %errorLevel% neq 0 goto :error

echo [3/4] Installing deep learning libraries...
pip install tensorflow keras --quiet
if %errorLevel% neq 0 goto :error

echo [4/4] Installing Jupyter (simple version)...
pip install jupyter --quiet
if %errorLevel% neq 0 goto :error

echo.
echo ======================================
echo ✓ Setup completed successfully!
echo ======================================
echo.
echo To start the project:
echo.
echo 1. Activate the environment (if not already):
echo    venv\Scripts\activate.bat
echo.
echo 2. Run the Jupyter notebook:
echo    python -m jupyter notebook
echo.
echo 3. Then open: notebooks/mango_leaf_disease_classification.ipynb
echo.
echo Or run the main script:
echo    python run_project.py
echo.
pause
exit /b 0

:error
echo.
echo ✗ Installation failed!
echo Please try these troubleshooting steps:
echo.
echo 1. Run as Administrator
echo 2. Restart your computer
echo 3. Run: FIX_WINDOWS_PATHS.bat (requires Admin)
echo.
pause
exit /b 1
