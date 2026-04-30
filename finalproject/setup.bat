@echo off
REM Quick setup script for Windows PowerShell

echo ================================
echo Mango Leaf Disease Classification
echo Quick Setup Script
echo ================================

REM Check Python version
python --version

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ✓ Setup complete!
echo.
echo Next steps:
echo 1. Download dataset from Kaggle:
echo    python setup_dataset.py
echo.
echo 2. Run the notebook:
echo    jupyter notebook
echo    Then open: notebooks\mango_leaf_disease_classification.ipynb
echo.
echo 3. Generate report:
echo    python generate_report.py
echo.
