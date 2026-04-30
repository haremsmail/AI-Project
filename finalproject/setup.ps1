# Mango Leaf Classification Setup (PowerShell)
# Run as Administrator for best results

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Mango Leaf Classification Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.9+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}
Write-Host "✓ $pythonVersion found" -ForegroundColor Green
Write-Host ""

# Create venv
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (-Not (Test-Path "venv")) {
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Failed to create venv" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}
Write-Host ""

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "✓ Environment activated" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel -q
Write-Host "✓ Pip upgraded" -ForegroundColor Green
Write-Host ""

# Install packages in batches
Write-Host "Installing dependencies (5-10 minutes)..." -ForegroundColor Yellow
Write-Host ""

$packages = @(
    @{name="Data Science"; pkg="numpy pandas matplotlib seaborn"},
    @{name="Image Processing"; pkg="pillow opencv-python scikit-image"},
    @{name="Machine Learning"; pkg="scikit-learn joblib tqdm"},
    @{name="Deep Learning"; pkg="tensorflow keras"},
    @{name="Jupyter"; pkg="jupyter ipykernel"}
)

foreach ($item in $packages) {
    Write-Host "[*] Installing $($item.name)..." -ForegroundColor Yellow
    pip install $item.pkg.Split() -q
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ $($item.name) installed" -ForegroundColor Green
    } else {
        Write-Host "⚠ $($item.name) had issues (continuing)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "✓ Setup completed!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Start Jupyter:" -ForegroundColor White
Write-Host "   python -m jupyter notebook" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Open: notebooks/mango_leaf_disease_classification.ipynb" -ForegroundColor Gray
Write-Host ""
Write-Host "Or run:" -ForegroundColor Cyan
Write-Host "   python run_project.py" -ForegroundColor Gray
Write-Host ""
