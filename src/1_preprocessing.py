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

CONFIG = {
    'sample_rate': 22050,          
    'duration': 30,                 
    'n_mels': 128,                  
    'n_fft': 2048,                  
    'hop_length': 512,              
    'n_mfcc': 40,                   
    'max_samples_per_class': 160,
}

BASE_PATH = r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\Datasets"
BANGLA_PATH = os.path.join(BASE_PATH, "Bangla_Datasets")
ENGLISH_PATH = os.path.join(BASE_PATH, "English_Datasets")
METADATA_PATH = os.path.join(BASE_PATH, "updated_metadata.csv")
OUTPUT_PATH = r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\processed_data1"

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
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=CONFIG['hop_length'])
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=CONFIG['hop_length'])
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=CONFIG['hop_length'])
    
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=CONFIG['hop_length'])
    
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
    mel_spec = extract_mel_spectrogram(audio, sr)
    
    mfcc = extract_mfcc(audio, sr)
    
    spectral = extract_spectral_features(audio, sr)
    
    chroma = extract_chroma_features(audio, sr)
    
    features = []
    
    features.extend(np.mean(mel_spec, axis=1))
    features.extend(np.std(mel_spec, axis=1))
    
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    
    for name, feat in spectral.items():
        features.append(np.mean(feat))
        features.append(np.std(feat))
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
        
        expected_samples = CONFIG['sample_rate'] * CONFIG['duration']
        if len(audio) < expected_samples:
            audio = np.pad(audio, (0, expected_samples - len(audio)), mode='constant')
        
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

print("Audio loading function defined!")

# ============================================================================
# CELL 5: Load Metadata and Collect All Audio Files
# ============================================================================

print("Loading metadata for genre lookup...")
metadata_csv = pd.read_csv(METADATA_PATH)
genre_lookup = dict(zip(metadata_csv['ID'].astype(str), metadata_csv['genre']))
print(f"Loaded {len(genre_lookup)} genre entries from metadata")

def collect_audio_files():
    """Collect all audio file paths with their labels from METADATA (not folder names)."""
    audio_files = []
    skipped_files = 0
    
    print("Collecting Bangla song files...")
    if os.path.exists(BANGLA_PATH):
        for genre_folder in os.listdir(BANGLA_PATH):
            genre_path = os.path.join(BANGLA_PATH, genre_folder)
            if os.path.isdir(genre_path):
                files_in_genre = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
                files_in_genre = files_in_genre[:CONFIG['max_samples_per_class']]
                for audio_file in files_in_genre:
                    file_id = os.path.splitext(audio_file)[0]
                    if file_id in genre_lookup:
                        audio_files.append({
                            'path': os.path.join(genre_path, audio_file),
                            'language': 'bn',
                            'genre': genre_lookup[file_id],
                            'filename': audio_file,
                            'file_id': file_id
                        })
                    else:
                        skipped_files += 1
    
    print("Collecting English song files...")
    if os.path.exists(ENGLISH_PATH):
        for genre_folder in os.listdir(ENGLISH_PATH):
            genre_path = os.path.join(ENGLISH_PATH, genre_folder)
            if os.path.isdir(genre_path):
                files_in_genre = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
                files_in_genre = files_in_genre[:CONFIG['max_samples_per_class']]
                for audio_file in files_in_genre:
                    file_id = os.path.splitext(audio_file)[0]
                    if file_id in genre_lookup:
                        audio_files.append({
                            'path': os.path.join(genre_path, audio_file),
                            'language': 'en',
                            'genre': genre_lookup[file_id],
                            'filename': audio_file,
                            'file_id': file_id
                        })
                    else:
                        skipped_files += 1
    
    print(f"Total audio files collected: {len(audio_files)}")
    if skipped_files > 0:
        print(f"Skipped {skipped_files} files (not found in metadata)")
    return audio_files

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
        
        audio, sr = load_audio_file(file_path)
        
        if audio is not None:
            try:
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

print("\nFeature breakdown:")
print(f"  - Mel spectrogram (mean + std): {CONFIG['n_mels'] * 2} features")
print(f"  - MFCC (mean + std): {CONFIG['n_mfcc'] * 2} features")
print(f"  - Spectral features (5 types × 2 stats): 10 features")
print(f"  - Chroma features (mean + std): 24 features")

print("\nLabel distribution:")
label_counts = pd.Series(labels).value_counts()
for label, count in label_counts.items():
    print(f"  - {label}: {count}")

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

print(f"\nNaN values in features: {np.isnan(features).sum()}")
print(f"Inf values in features: {np.isinf(features).sum()}")

features_clean = np.where(np.isinf(features), np.nan, features)

imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features_clean)

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

metadata_df = pd.DataFrame(metadata)
metadata_df['label'] = labels

np.save(os.path.join(OUTPUT_PATH, 'features_raw.npy'), features)
np.save(os.path.join(OUTPUT_PATH, 'features_normalized.npy'), features_normalized)

np.save(os.path.join(OUTPUT_PATH, 'labels.npy'), np.array(labels))

metadata_df.to_csv(os.path.join(OUTPUT_PATH, 'metadata.csv'), index=False)

with open(os.path.join(OUTPUT_PATH, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(OUTPUT_PATH, 'imputer.pkl'), 'wb') as f:
    pickle.dump(imputer, f)

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
