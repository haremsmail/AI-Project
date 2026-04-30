#!/bin/bash
# Quick setup script for Unix-based systems (macOS, Linux)

echo "================================"
echo "Mango Leaf Disease Classification"
echo "Quick Setup Script"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download dataset from Kaggle:"
echo "   python setup_dataset.py"
echo ""
echo "2. Run the notebook:"
echo "   jupyter notebook"
echo "   Then open: notebooks/mango_leaf_disease_classification.ipynb"
echo ""
echo "3. Generate report:"
echo "   python generate_report.py"
echo ""
