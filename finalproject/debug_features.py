#!/usr/bin/env python
"""Debug feature extraction issues."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.config import RAW_DATA_DIR
from src.utils import discover_images
from src.features import load_image, extract_handcrafted_matrix

# Find dataset
print("Looking for dataset...")
candidate_roots = [
    RAW_DATA_DIR / "mango_leaf_disease_dataset",
    RAW_DATA_DIR / "MangoLeafBD Dataset",
    RAW_DATA_DIR,
]

df = None
dataset_root = None

for root in candidate_roots:
    if root.exists():
        df = discover_images(root)
        if not df.empty:
            dataset_root = root
            break

if df is None or df.empty:
    print("ERROR: Dataset not found!")
    sys.exit(1)

print(f"✓ Found {len(df)} images at {dataset_root}")

# Try first 10 images
print("\nTesting feature extraction on first 10 images...")
for idx in range(min(10, len(df))):
    row = df.iloc[idx]
    image_path = row['image_path']
    label = row['label']
    
    try:
        print(f"\n[{idx}] Loading: {Path(image_path).name}")
        img = load_image(image_path)
        print(f"    ✓ Image loaded, shape: {img.shape}, dtype: {img.dtype}")
        
        feat = extract_handcrafted_matrix(img)
        print(f"    ✓ Features extracted, shape: {feat.shape}, dtype: {feat.dtype}")
        print(f"    Label: {label}")
        
    except Exception as e:
        print(f"    ✗ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n✓ Debug complete")
