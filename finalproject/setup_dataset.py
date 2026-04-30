#!/usr/bin/env python
"""Download and extract the Kaggle Mango Leaf Disease Dataset.

This script helps you download the dataset directly from Kaggle.
Requires: kaggle CLI configured with API credentials.

Setup:
1. Download your API key from https://www.kaggle.com/account/
2. Place kaggle.json in ~/.kaggle/
3. Run: python setup_dataset.py
"""

from pathlib import Path
import subprocess
import sys

from src.config import RAW_DATA_DIR


def download_dataset():
    """Download the mango leaf disease dataset from Kaggle."""
    dataset_id = "warcoder/mango-leaf-disease-dataset"
    target_dir = RAW_DATA_DIR

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {dataset_id}")
    print(f"Destination: {target_dir}")

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(target_dir), "--unzip"],
            check=True,
        )
        print("\n✓ Dataset downloaded and extracted successfully!")
        print(f"Check the contents at: {target_dir}")
    except FileNotFoundError:
        print("Error: kaggle CLI not found.")
        print("Install it with: pip install kaggle")
        print("Then configure it with your API key from https://www.kaggle.com/account/")
        sys.exit(1)
    except subprocess.CalledProcessError as error:
        print(f"Error downloading dataset: {error}")
        print("Make sure your Kaggle API key is configured correctly.")
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
