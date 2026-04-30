#!/usr/bin/env python
"""Debug dataframe structure."""

from src.config import RAW_DATA_DIR
from src.utils import discover_images
from pathlib import Path

candidate_roots = [
    RAW_DATA_DIR / 'mango_leaf_disease_dataset',
    RAW_DATA_DIR / 'MangoLeafBD Dataset',
    RAW_DATA_DIR,
]

for root in candidate_roots:
    if root.exists():
        df = discover_images(root)
        if not df.empty:
            print(f'Original DF columns: {df.columns.tolist()}')
            print(f'Original DF shape: {df.shape}')
            
            # Sample it
            df_sample = df.sample(n=min(200, len(df)), random_state=42)
            print(f'\nSampled DF columns: {df_sample.columns.tolist()}')
            print(f'Sampled DF shape: {df_sample.shape}')
            
            # Try iterating
            print(f'\nTesting iteration:')
            for idx, row in df_sample.head(3).iterrows():
                try:
                    label = row['label']
                    image_path = row['image_path']
                    print(f'Row {idx}: OK - label={label}, image_path={Path(image_path).name}')
                except KeyError as e:
                    print(f'Row {idx}: ERROR - {e}')
                    print(f'  Row keys: {row.index.tolist()}')
            break
