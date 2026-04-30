#!/usr/bin/env python
"""
Setup script for Mango Leaf Disease Classification Project
Handles Windows Long Path issues automatically
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def enable_windows_long_paths():
    """Enable Windows Long Path support via Registry"""
    if platform.system() != 'Windows':
        return True
    
    print("Attempting to enable Windows Long Path support...")
    try:
        import winreg
        reg = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        key = winreg.OpenKey(reg, r'SYSTEM\CurrentControlSet\Control\FileSystem', 0, winreg.KEY_WRITE)
        winreg.SetValueEx(key, 'LongPathsEnabled', 0, winreg.REG_DWORD, 1)
        winreg.CloseKey(key)
        print("✓ Long Path support enabled!")
        return True
    except PermissionError:
        print("⚠ Could not enable Long Path support (requires Administrator)")
        print("  Run: FIX_WINDOWS_PATHS.bat as Administrator")
        return False
    except Exception as e:
        print(f"⚠ Could not enable Long Path: {e}")
        return False

def run_command(cmd, description):
    """Run a shell command"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"✗ Error: {result.stderr}")
            return False
        print(f"✓ {description} completed")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("="*50)
    print("Mango Leaf Classification - Setup")
    print("="*50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print(f"✗ Python 3.8+ required (found {sys.version_info.major}.{sys.version_info.minor})")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Enable long paths on Windows
    if platform.system() == 'Windows':
        enable_windows_long_paths()
    
    # Create virtual environment
    venv_path = Path('venv')
    if not venv_path.exists():
        print("\nCreating virtual environment...")
        if not run_command(f"{sys.executable} -m venv venv", "Create venv"):
            sys.exit(1)
    else:
        print("\n✓ Virtual environment already exists")
    
    # Determine activation command
    if platform.system() == 'Windows':
        activate_cmd = r"venv\Scripts\activate.bat && "
        pip_cmd = f"{sys.executable} -m pip"
    else:
        activate_cmd = "source venv/bin/activate && "
        pip_cmd = f"{sys.executable} -m pip"
    
    # Upgrade pip
    print("\nUpgrading pip...")
    subprocess.run(f"{pip_cmd} install --upgrade pip setuptools wheel", shell=True, capture_output=True)
    
    # Install dependencies
    deps = {
        'Core': 'numpy pandas matplotlib seaborn pillow',
        'Image': 'opencv-python scikit-image scikit-learn joblib tqdm',
        'ML': 'tensorflow keras',
        'Jupyter': 'jupyter ipykernel'
    }
    
    for category, packages in deps.items():
        print(f"\nInstalling {category} packages...")
        cmd = f"{pip_cmd} install {packages} --quiet"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {category} installed")
        else:
            print(f"⚠ {category} installation had issues (continuing anyway)")
    
    # Create directories
    print("\nCreating project directories...")
    for directory in ['data/raw', 'data/processed', 'models', 'results']:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")
    
    # Final summary
    print("\n" + "="*50)
    print("✓ Setup completed successfully!")
    print("="*50)
    print("\nNext steps:")
    print("\n1. Activate environment:")
    if platform.system() == 'Windows':
        print("   venv\\Scripts\\activate.bat")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Start Jupyter Notebook:")
    print("   python -m jupyter notebook")
    
    print("\n3. Open the notebook:")
    print("   notebooks/mango_leaf_disease_classification.ipynb")
    
    print("\nOr run the project directly:")
    print("   python run_project.py")
    print()

if __name__ == '__main__':
    main()
