#!/usr/bin/env python
"""Super simple runner - trains on 200 sample images only."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("MANGO LEAF DISEASE CLASSIFIER - Sample Training")
print("=" * 60)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import main functions
from main import prepare_dataset, extract_features, train_all_models

print("\n[1] Preparing dataset...")
df, root = prepare_dataset()
if df is None:
    print("ERROR: Could not prepare dataset!")
    sys.exit(1)

# Use only 200 images for quick testing
print("\n[2] Using sample (200 images)...")
df_sample = df.sample(n=min(200, len(df)), random_state=42)
print(f"Sample size: {len(df_sample)} images")
print(f"Columns: {df_sample.columns.tolist()}")

print("\n[3] Extracting features...")
result = extract_features(df_sample)
if result[0] is None:
    print("ERROR: Could not extract features!")
    sys.exit(1)

data_tuple, le, scaler, raw_data = result

print("\n[4] Training all models...")
train_all_models(data_tuple, le)

print("\n" + "=" * 60)
print("SUCCESS! Models are trained and ready.")
print("Run: python main.py")
print("Then select option 2 to test on an image!")
print("=" * 60)
