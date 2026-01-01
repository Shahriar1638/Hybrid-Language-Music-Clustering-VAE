"""
Script to normalize all genre names to lowercase.
"""
import pandas as pd
import numpy as np

print("="*50)
print("Normalizing Genre Names to Lowercase")
print("="*50)

# 1. Fix Datasets/updated_metadata.csv (main dataset)
print("\n1. Fixing Datasets/updated_metadata.csv...")
df = pd.read_csv('Datasets/updated_metadata.csv')
print(f"   Before: {df['genre'].nunique()} unique genres")
df['genre'] = df['genre'].str.lower()
df.to_csv('Datasets/updated_metadata.csv', index=False)
print(f"   After: {df['genre'].nunique()} unique genres")
print("   Genres:", sorted(df['genre'].unique()))

# 2. Fix processed_data/metadata.csv
print("\n2. Fixing processed_data/metadata.csv...")
df_proc = pd.read_csv('processed_data/metadata.csv')
print(f"   Before: {df_proc['genre'].nunique()} unique genres")
df_proc['genre'] = df_proc['genre'].str.lower()
df_proc.to_csv('processed_data/metadata.csv', index=False)
print(f"   After: {df_proc['genre'].nunique()} unique genres")

# 3. Fix processed_data/labels.npy
print("\n3. Fixing processed_data/labels.npy...")
labels = np.load('processed_data/labels.npy', allow_pickle=True)
print(f"   Before: {len(np.unique(labels))} unique labels")
labels_lower = np.array([str(l).lower() for l in labels])
np.save('processed_data/labels.npy', labels_lower)
print(f"   After: {len(np.unique(labels_lower))} unique labels")

# Show final distribution
print("\n" + "="*50)
print("Final Genre Distribution in Processed Data:")
print("="*50)
unique, counts = np.unique(labels_lower, return_counts=True)
for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
    print(f"  {u:15s}: {c:5d}")

print(f"\nTotal: {len(labels_lower)} samples, {len(unique)} genres")
print("\nDone! All genre names normalized to lowercase.")
