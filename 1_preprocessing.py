# ============================================================================
# CELL 1: Import Required Libraries
# ============================================================================

import os
import numpy as np
import pandas as pd
import librosa
import warnings
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# ============================================================================
# CELL 2: Configuration and Paths
# ============================================================================

# Configuration parameters
CONFIG = {
    'sample_rate': 22050,          # Standard sample rate for audio processing
    'duration': 30,                 # Duration in seconds to load from each audio file
    'n_mels': 128,                  # Number of mel bands
    'n_fft': 2048,                  # FFT window size
    'hop_length': 512,              # Hop length for STFT
    'n_mfcc': 40,                   # Number of MFCC coefficients
    'max_samples_per_class': 50,   # Maximum samples per class to load (for faster processing)
}

# Define paths
BASE_PATH = r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\Datasets"
BANGLA_PATH = os.path.join(BASE_PATH, "Bangla_Datasets")
ENGLISH_PATH = os.path.join(BASE_PATH, "English_Datasets")
METADATA_PATH = os.path.join(BASE_PATH, "updated_metadata.csv")
OUTPUT_PATH = r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\processed_data"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"Configuration loaded!")
print(f"Bangla datasets path: {BANGLA_PATH}")
print(f"English datasets path: {ENGLISH_PATH}")
print(f"Output path: {OUTPUT_PATH}")

# ============================================================================
# CELL 3: Audio Feature Extraction Functions
# ============================================================================

def extract_mel_spectrogram(audio, sr):
    """Extract mel spectrogram from audio signal."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=CONFIG['n_mels'],
        n_fft=CONFIG['n_fft'],
        hop_length=CONFIG['hop_length']
    )
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def extract_mfcc(audio, sr):
    """Extract MFCC features from audio signal."""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=CONFIG['n_mfcc'],
        n_fft=CONFIG['n_fft'],
        hop_length=CONFIG['hop_length']
    )
    return mfcc


def extract_spectral_features(audio, sr):
    """Extract various spectral features."""
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=CONFIG['hop_length'])
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=CONFIG['hop_length'])
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=CONFIG['hop_length'])
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=CONFIG['hop_length'])
    
    # RMS energy
    rms = librosa.feature.rms(y=audio, hop_length=CONFIG['hop_length'])
    
    return {
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_rolloff': spectral_rolloff,
        'zcr': zcr,
        'rms': rms
    }


def extract_chroma_features(audio, sr):
    """Extract chroma features."""
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=CONFIG['n_fft'],
        hop_length=CONFIG['hop_length']
    )
    return chroma


def extract_all_features(audio, sr):
    """Extract all features and return as a fixed-size feature vector."""
    # Extract mel spectrogram
    mel_spec = extract_mel_spectrogram(audio, sr)
    
    # Extract MFCC
    mfcc = extract_mfcc(audio, sr)
    
    # Extract spectral features
    spectral = extract_spectral_features(audio, sr)
    
    # Extract chroma features
    chroma = extract_chroma_features(audio, sr)
    
    # Aggregate features into fixed-size vectors using statistical measures
    features = []
    
    # Mel spectrogram statistics (mean and std across time for each mel band)
    features.extend(np.mean(mel_spec, axis=1))
    features.extend(np.std(mel_spec, axis=1))
    
    # MFCC statistics
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    
    # Spectral feature statistics
    for name, feat in spectral.items():
        features.append(np.mean(feat))
        features.append(np.std(feat))
    
    # Chroma statistics
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))
    
    return np.array(features)

print("Feature extraction functions defined!")

# ============================================================================
# CELL 4: Load Audio File Function
# ============================================================================

def load_audio_file(file_path):
    """Load an audio file with error handling."""
    try:
        audio, sr = librosa.load(
            file_path,
            sr=CONFIG['sample_rate'],
            duration=CONFIG['duration']
        )
        
        # Pad if audio is shorter than duration
        expected_samples = CONFIG['sample_rate'] * CONFIG['duration']
        if len(audio) < expected_samples:
            audio = np.pad(audio, (0, expected_samples - len(audio)), mode='constant')
        
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

print("Audio loading function defined!")

# ============================================================================
# CELL 5: Collect All Audio Files
# ============================================================================

def collect_audio_files():
    """Collect all audio file paths with their labels."""
    audio_files = []
    
    # Collect Bangla songs
    print("Collecting Bangla song files...")
    if os.path.exists(BANGLA_PATH):
        for genre_folder in os.listdir(BANGLA_PATH):
            genre_path = os.path.join(BANGLA_PATH, genre_folder)
            if os.path.isdir(genre_path):
                files_in_genre = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
                # Limit samples per class
                files_in_genre = files_in_genre[:CONFIG['max_samples_per_class']]
                for audio_file in files_in_genre:
                    audio_files.append({
                        'path': os.path.join(genre_path, audio_file),
                        'language': 'bn',
                        'genre': genre_folder,
                        'filename': audio_file
                    })
    
    # Collect English songs
    print("Collecting English song files...")
    if os.path.exists(ENGLISH_PATH):
        for genre_folder in os.listdir(ENGLISH_PATH):
            genre_path = os.path.join(ENGLISH_PATH, genre_folder)
            if os.path.isdir(genre_path):
                files_in_genre = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
                # Limit samples per class
                files_in_genre = files_in_genre[:CONFIG['max_samples_per_class']]
                for audio_file in files_in_genre:
                    audio_files.append({
                        'path': os.path.join(genre_path, audio_file),
                        'language': 'en',
                        'genre': genre_folder,
                        'filename': audio_file
                    })
    
    print(f"Total audio files collected: {len(audio_files)}")
    return audio_files

# Collect files
audio_files = collect_audio_files()
print(f"\nTotal files to process: {len(audio_files)}")

# ============================================================================
# CELL 6: Process All Audio Files and Extract Features
# ============================================================================

def process_audio_files(audio_files):
    """Process all audio files and extract features."""
    all_features = []
    all_labels = []
    all_metadata = []
    failed_files = []
    
    print(f"Processing {len(audio_files)} audio files...")
    
    for file_info in tqdm(audio_files, desc="Extracting features"):
        file_path = file_info['path']
        
        # Load audio
        audio, sr = load_audio_file(file_path)
        
        if audio is not None:
            try:
                # Extract features
                features = extract_all_features(audio, sr)
                
                all_features.append(features)
                all_labels.append(file_info['genre'])
                all_metadata.append({
                    'language': file_info['language'],
                    'genre': file_info['genre'],
                    'filename': file_info['filename']
                })
            except Exception as e:
                failed_files.append((file_path, str(e)))
        else:
            failed_files.append((file_path, "Failed to load"))
    
    print(f"\nSuccessfully processed: {len(all_features)} files")
    print(f"Failed to process: {len(failed_files)} files")
    
    return np.array(all_features), all_labels, all_metadata, failed_files

# Process files
features, labels, metadata, failed = process_audio_files(audio_files)

print(f"\nFeature matrix shape: {features.shape}")
print(f"Number of labels: {len(labels)}")

# ============================================================================
# CELL 7: Display Feature Statistics
# ============================================================================

print("\n" + "="*60)
print("FEATURE EXTRACTION SUMMARY")
print("="*60)

print(f"\nFeature vector size per sample: {features.shape[1]}")
print(f"Total number of samples: {features.shape[0]}")

# Feature breakdown
print("\nFeature breakdown:")
print(f"  - Mel spectrogram (mean + std): {CONFIG['n_mels'] * 2} features")
print(f"  - MFCC (mean + std): {CONFIG['n_mfcc'] * 2} features")
print(f"  - Spectral features (5 types × 2 stats): 10 features")
print(f"  - Chroma features (mean + std): 24 features")

# Label distribution
print("\nLabel distribution:")
label_counts = pd.Series(labels).value_counts()
for label, count in label_counts.items():
    print(f"  - {label}: {count}")

# Language distribution
languages = [m['language'] for m in metadata]
lang_counts = pd.Series(languages).value_counts()
print("\nLanguage distribution:")
for lang, count in lang_counts.items():
    print(f"  - {'Bangla' if lang == 'bn' else 'English'}: {count}")

# ============================================================================
# CELL 8: Handle Missing Values and Normalize Features
# ============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Check for NaN or inf values
print(f"\nNaN values in features: {np.isnan(features).sum()}")
print(f"Inf values in features: {np.isinf(features).sum()}")

# Replace inf with NaN, then impute
features_clean = np.where(np.isinf(features), np.nan, features)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features_clean)

# Normalize features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features_imputed)

print(f"\nAfter preprocessing:")
print(f"  - NaN values: {np.isnan(features_normalized).sum()}")
print(f"  - Feature mean (should be ~0): {features_normalized.mean():.4f}")
print(f"  - Feature std (should be ~1): {features_normalized.std():.4f}")

# ============================================================================
# CELL 9: Save Preprocessed Data
# ============================================================================

print("\n" + "="*60)
print("SAVING PREPROCESSED DATA")
print("="*60)

# Create DataFrame with metadata
metadata_df = pd.DataFrame(metadata)
metadata_df['label'] = labels

# Save features
np.save(os.path.join(OUTPUT_PATH, 'features_raw.npy'), features)
np.save(os.path.join(OUTPUT_PATH, 'features_normalized.npy'), features_normalized)

# Save labels
np.save(os.path.join(OUTPUT_PATH, 'labels.npy'), np.array(labels))

# Save metadata
metadata_df.to_csv(os.path.join(OUTPUT_PATH, 'metadata.csv'), index=False)

# Save scaler and imputer for later use
with open(os.path.join(OUTPUT_PATH, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(OUTPUT_PATH, 'imputer.pkl'), 'wb') as f:
    pickle.dump(imputer, f)

# Save configuration
with open(os.path.join(OUTPUT_PATH, 'config.pkl'), 'wb') as f:
    pickle.dump(CONFIG, f)

print(f"\nFiles saved to: {OUTPUT_PATH}")
print("  - features_raw.npy")
print("  - features_normalized.npy")
print("  - labels.npy")
print("  - metadata.csv")
print("  - scaler.pkl")
print("  - imputer.pkl")
print("  - config.pkl")

# ============================================================================
# CELL 10: Verify Saved Data
# ============================================================================

print("\n" + "="*60)
print("VERIFICATION")
print("="*60)

# Load and verify
features_loaded = np.load(os.path.join(OUTPUT_PATH, 'features_normalized.npy'))
labels_loaded = np.load(os.path.join(OUTPUT_PATH, 'labels.npy'), allow_pickle=True)
metadata_loaded = pd.read_csv(os.path.join(OUTPUT_PATH, 'metadata.csv'))

print(f"\nLoaded features shape: {features_loaded.shape}")
print(f"Loaded labels count: {len(labels_loaded)}")
print(f"Loaded metadata shape: {metadata_loaded.shape}")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)
print("\nYou can now proceed to the VAE training and clustering notebook.")
