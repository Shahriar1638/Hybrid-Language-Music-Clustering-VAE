"""
Data Quality Check Script for VAE Clustering Project
=====================================================
This script analyzes the quality of your processed data to help you:
1. Verify if preprocessing was done correctly
2. Identify potential issues that might affect model training
3. Suggest hyperparameter adjustments based on data characteristics

Run this script after preprocessing but before training your VAE model.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configuration
PROCESSED_DATA_DIR = "processed_data"
RESULTS_DIR = "results"
REPORT_FILE = "data_quality_report.txt"

def load_processed_data():
    """Load all processed data files."""
    data = {}
    
    # Load numpy arrays
    files_to_load = {
        'features_normalized': 'features_normalized.npy',
        'features_raw': 'features_raw.npy',
        'labels': 'labels.npy'
    }
    
    for key, filename in files_to_load.items():
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        if os.path.exists(filepath):
            data[key] = np.load(filepath, allow_pickle=True)
            print(f"[OK] Loaded {filename}: shape {data[key].shape}, dtype {data[key].dtype}")
        else:
            print(f"[X] Missing {filename}")
            data[key] = None
    
    # Load metadata
    metadata_path = os.path.join(PROCESSED_DATA_DIR, 'metadata.csv')
    if os.path.exists(metadata_path):
        data['metadata'] = pd.read_csv(metadata_path)
        print(f"[OK] Loaded metadata.csv: {len(data['metadata'])} rows")
    else:
        print(f"[X] Missing metadata.csv")
        data['metadata'] = None
    
    # Load config
    config_path = os.path.join(PROCESSED_DATA_DIR, 'config.pkl')
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            data['config'] = pickle.load(f)
        print(f"[OK] Loaded config.pkl")
    else:
        data['config'] = None
    
    return data

def check_data_consistency(data, report_lines):
    """Check if all data components are consistent."""
    report_lines.append("\n" + "="*60)
    report_lines.append("1. DATA CONSISTENCY CHECK")
    report_lines.append("="*60)
    
    issues = []
    
    if data['features_normalized'] is not None and data['labels'] is not None:
        if len(data['features_normalized']) != len(data['labels']):
            issues.append(f"[ERROR] Features ({len(data['features_normalized'])}) and labels ({len(data['labels'])}) have different lengths!")
        else:
            report_lines.append(f"[OK] Features and labels have same length: {len(data['features_normalized'])} samples")
    
    if data['metadata'] is not None and data['features_normalized'] is not None:
        if len(data['metadata']) != len(data['features_normalized']):
            issues.append(f"[ERROR] Metadata ({len(data['metadata'])}) and features ({len(data['features_normalized'])}) have different lengths!")
        else:
            report_lines.append(f"[OK] Metadata and features have same length")
    
    if issues:
        for issue in issues:
            report_lines.append(issue)
    else:
        report_lines.append("[OK] All data components are consistent!")
    
    return len(issues) == 0

def analyze_features(data, report_lines):
    """Analyze feature statistics and quality."""
    report_lines.append("\n" + "="*60)
    report_lines.append("2. FEATURE ANALYSIS")
    report_lines.append("="*60)
    
    features = data['features_normalized']
    if features is None:
        report_lines.append("[ERROR] No features to analyze")
        return
    
    n_samples, n_features = features.shape
    report_lines.append(f"\nDataset Shape: {n_samples} samples x {n_features} features")
    
    # Basic statistics
    report_lines.append("\n--- Basic Statistics ---")
    report_lines.append(f"Mean: {np.mean(features):.6f}")
    report_lines.append(f"Std: {np.std(features):.6f}")
    report_lines.append(f"Min: {np.min(features):.6f}")
    report_lines.append(f"Max: {np.max(features):.6f}")
    
    # Check for NaN/Inf values
    report_lines.append("\n--- Missing/Invalid Values ---")
    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()
    
    if nan_count > 0:
        report_lines.append(f"[ERROR] Found {nan_count} NaN values ({100*nan_count/features.size:.2f}%)")
    else:
        report_lines.append("[OK] No NaN values")
    
    if inf_count > 0:
        report_lines.append(f"[ERROR] Found {inf_count} Inf values ({100*inf_count/features.size:.2f}%)")
    else:
        report_lines.append("[OK] No Inf values")
    
    # Check normalization quality
    report_lines.append("\n--- Normalization Quality ---")
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    
    near_zero_mean = np.sum(np.abs(feature_means) < 0.01)
    near_unit_std = np.sum(np.abs(feature_stds - 1.0) < 0.1)
    
    report_lines.append(f"Features with mean ~ 0: {near_zero_mean}/{n_features} ({100*near_zero_mean/n_features:.1f}%)")
    report_lines.append(f"Features with std ~ 1: {near_unit_std}/{n_features} ({100*near_unit_std/n_features:.1f}%)")
    
    if near_zero_mean < n_features * 0.8:
        report_lines.append("[WARN] Warning: Normalization may not be optimal. Consider re-normalizing.")
    
    # Check for constant features
    constant_features = np.sum(feature_stds < 1e-6)
    if constant_features > 0:
        report_lines.append(f"[WARN] Warning: {constant_features} features have near-zero variance (constant)")
    
    # Check for highly correlated features
    report_lines.append("\n--- Feature Correlations ---")
    if n_features < 500:  # Only compute for smaller feature sets
        corr_matrix = np.corrcoef(features.T)
        high_corr_pairs = np.sum(np.abs(corr_matrix) > 0.95) - n_features  # Exclude diagonal
        report_lines.append(f"Highly correlated feature pairs (|r| > 0.95): {high_corr_pairs // 2}")
        if high_corr_pairs > n_features:
            report_lines.append("[WARN] Warning: Many highly correlated features. Consider PCA dimensionality reduction.")
    else:
        report_lines.append("(Skipped for large feature set)")
    
    return features

def analyze_labels(data, report_lines):
    """Analyze label distribution."""
    report_lines.append("\n" + "="*60)
    report_lines.append("3. LABEL DISTRIBUTION ANALYSIS")
    report_lines.append("="*60)
    
    labels = data['labels']
    if labels is None:
        report_lines.append("[ERROR] No labels to analyze")
        return
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_labels)
    
    report_lines.append(f"\nNumber of classes: {n_classes}")
    report_lines.append(f"Total samples: {len(labels)}")
    
    report_lines.append("\n--- Class Distribution ---")
    for label, count in sorted(zip(unique_labels, counts), key=lambda x: -x[1]):
        percentage = 100 * count / len(labels)
        bar = "#" * int(percentage / 2)
        report_lines.append(f"{label:15s}: {count:5d} ({percentage:5.1f}%) {bar}")
    
    # Check for class imbalance
    report_lines.append("\n--- Class Balance Metrics ---")
    imbalance_ratio = max(counts) / min(counts)
    report_lines.append(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 3:
        report_lines.append("[WARN] Warning: High class imbalance detected!")
        report_lines.append("   Suggestions:")
        report_lines.append("   - Consider oversampling minority classes")
        report_lines.append("   - Use class weights in training")
        report_lines.append("   - Try SMOTE or data augmentation")
    elif imbalance_ratio > 2:
        report_lines.append("[WARN] Moderate class imbalance. Consider using class weights.")
    else:
        report_lines.append("[OK] Class distribution is reasonably balanced")
    
    return unique_labels, counts

def suggest_hyperparameters(data, report_lines):
    """Suggest hyperparameters based on data characteristics."""
    report_lines.append("\n" + "="*60)
    report_lines.append("4. HYPERPARAMETER RECOMMENDATIONS")
    report_lines.append("="*60)
    
    features = data['features_normalized']
    labels = data['labels']
    
    if features is None:
        return
    
    n_samples, n_features = features.shape
    n_classes = len(np.unique(labels)) if labels is not None else 12
    
    report_lines.append("\n--- VAE Architecture Suggestions ---")
    
    # Input dimension
    report_lines.append(f"\nInput dimension: {n_features}")
    
    # Latent dimension suggestions
    if n_features > 200:
        suggested_latent = [32, 64, 128]
    elif n_features > 100:
        suggested_latent = [16, 32, 64]
    else:
        suggested_latent = [8, 16, 32]
    report_lines.append(f"Suggested latent dimensions: {suggested_latent}")
    
    # Encoder architecture based on feature count
    if n_features > 200:
        report_lines.append(f"Suggested encoder layers: [{n_features}, 512, 256, 128]")
        report_lines.append(f"Suggested decoder layers: [128, 256, 512, {n_features}]")
    elif n_features > 100:
        report_lines.append(f"Suggested encoder layers: [{n_features}, 256, 128]")
        report_lines.append(f"Suggested decoder layers: [128, 256, {n_features}]")
    else:
        report_lines.append(f"Suggested encoder layers: [{n_features}, 128, 64]")
        report_lines.append(f"Suggested decoder layers: [64, 128, {n_features}]")
    
    # Training suggestions
    report_lines.append("\n--- Training Suggestions ---")
    
    if n_samples < 1000:
        suggested_batch = 16
        suggested_epochs = 200
    elif n_samples < 5000:
        suggested_batch = 32
        suggested_epochs = 150
    else:
        suggested_batch = 64
        suggested_epochs = 100
    
    report_lines.append(f"Dataset size: {n_samples} samples")
    report_lines.append(f"Suggested batch size: {suggested_batch}")
    report_lines.append(f"Suggested epochs: {suggested_epochs}")
    report_lines.append(f"Suggested learning rate: 1e-3 to 1e-4")
    report_lines.append(f"Suggested beta (KL weight): 0.1 to 1.0 (start lower)")
    
    # Clustering suggestions
    report_lines.append("\n--- Clustering Suggestions ---")
    report_lines.append(f"Number of known classes: {n_classes}")
    report_lines.append(f"Suggested K for K-Means: {n_classes}")
    report_lines.append(f"Alternative K values to try: {max(2, n_classes-2)} to {n_classes+2}")
    
    # Data-specific recommendations
    report_lines.append("\n--- Data-Specific Recommendations ---")
    
    # Check if data might need more preprocessing
    feature_stds = np.std(features, axis=0)
    if np.any(feature_stds < 0.1):
        report_lines.append("[WARN] Some features have low variance. Consider:")
        report_lines.append("   - Removing low-variance features")
        report_lines.append("   - Feature selection techniques")
    
    # Sample size recommendations
    samples_per_class = n_samples / n_classes
    if samples_per_class < 50:
        report_lines.append("[WARN] Low samples per class. Consider:")
        report_lines.append("   - Data augmentation")
        report_lines.append("   - Regularization (dropout, weight decay)")
        report_lines.append("   - Smaller model capacity")
    
    return suggested_latent, suggested_batch, suggested_epochs

def visualize_data(data, report_lines):
    """Create visualization of data quality."""
    report_lines.append("\n" + "="*60)
    report_lines.append("5. VISUALIZATIONS")
    report_lines.append("="*60)
    
    features = data['features_normalized']
    labels = data['labels']
    
    if features is None:
        return
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Data Quality Visualization', fontsize=14, fontweight='bold')
    
    # 1. Feature distribution (histogram of mean values per feature)
    ax1 = axes[0, 0]
    feature_means = np.mean(features, axis=0)
    ax1.hist(feature_means, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Feature Mean Value')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Feature Means')
    ax1.axvline(x=0, color='red', linestyle='--', label='Expected (0)')
    ax1.legend()
    
    # 2. Feature variance distribution
    ax2 = axes[0, 1]
    feature_stds = np.std(features, axis=0)
    # Handle cases with low variance by using fewer bins
    try:
        ax2.hist(feature_stds, bins='auto', color='coral', edgecolor='black', alpha=0.7)
    except ValueError:
        ax2.hist(feature_stds, bins=10, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Feature Std Dev')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Feature Standard Deviations')
    ax2.axvline(x=1, color='red', linestyle='--', label='Expected (1)')
    ax2.legend()
    
    # 3. Sample distribution (t-SNE or random 2D projection)
    ax3 = axes[1, 0]
    if features.shape[0] > 500:
        # Use random projection for speed
        np.random.seed(42)
        idx = np.random.choice(features.shape[1], 2, replace=False)
        proj = features[:, idx]
    else:
        proj = features[:, :2]
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax3.scatter(proj[mask, 0], proj[mask, 1], c=[colors[i]], 
                       label=label, alpha=0.6, s=20)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax3.scatter(proj[:, 0], proj[:, 1], c='steelblue', alpha=0.5, s=20)
    ax3.set_xlabel('Random Feature 1')
    ax3.set_ylabel('Random Feature 2')
    ax3.set_title('2D Random Projection of Samples')
    
    # 4. Class distribution bar chart
    ax4 = axes[1, 1]
    if labels is not None:
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_idx = np.argsort(-counts)
        bars = ax4.bar(range(len(unique_labels)), counts[sorted_idx], 
                       color='teal', edgecolor='black', alpha=0.7)
        ax4.set_xticks(range(len(unique_labels)))
        ax4.set_xticklabels([unique_labels[i] for i in sorted_idx], 
                           rotation=45, ha='right', fontsize=9)
        ax4.set_xlabel('Genre')
        ax4.set_ylabel('Sample Count')
        ax4.set_title('Class Distribution')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts[sorted_idx]):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    viz_path = os.path.join(RESULTS_DIR, 'data_quality_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    report_lines.append(f"\n[OK] Visualizations saved to: {viz_path}")
    print(f"\n[OK] Visualizations saved to: {viz_path}")

def main():
    """Main function to run all quality checks."""
    print("\n" + "="*60)
    print("   DATA QUALITY CHECK - VAE Clustering Project")
    print("="*60 + "\n")
    
    report_lines = []
    report_lines.append("DATA QUALITY REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated: {pd.Timestamp.now()}")
    
    # Load data
    print("Loading processed data...")
    data = load_processed_data()
    
    # Run checks
    print("\n" + "="*40)
    print("Running quality checks...")
    print("="*40)
    
    check_data_consistency(data, report_lines)
    analyze_features(data, report_lines)
    analyze_labels(data, report_lines)
    suggest_hyperparameters(data, report_lines)
    visualize_data(data, report_lines)
    
    # Summary
    report_lines.append("\n" + "="*60)
    report_lines.append("6. SUMMARY")
    report_lines.append("="*60)
    
    if data['features_normalized'] is not None:
        n_samples, n_features = data['features_normalized'].shape
        n_classes = len(np.unique(data['labels'])) if data['labels'] is not None else "Unknown"
        
        report_lines.append(f"\n* Total samples: {n_samples}")
        report_lines.append(f"* Feature dimensions: {n_features}")
        report_lines.append(f"* Number of classes: {n_classes}")
        report_lines.append(f"* Data files present: {sum(1 for v in data.values() if v is not None)}/5")
    
    report_lines.append("\n" + "="*60)
    report_lines.append("END OF REPORT")
    report_lines.append("="*60)
    
    # Save report
    report_path = os.path.join(RESULTS_DIR, REPORT_FILE)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Print report to console
    print("\n" + "\n".join(report_lines))
    print(f"\n[OK] Full report saved to: {report_path}")
    
    return report_lines

if __name__ == "__main__":
    main()
