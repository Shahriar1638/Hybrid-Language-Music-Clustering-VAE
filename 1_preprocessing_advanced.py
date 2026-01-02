# ============================================================================
# ADVANCED PREPROCESSING FOR CONVOLUTIONAL VAE (OPTIMIZED)
# Generates high-res mel spectrograms (2D) and lyrics embeddings
# features: Parallel Processing (CPU 100%), High Res, Strict Filtering
# ============================================================================

import os
import numpy as np
import pandas as pd
import librosa
import warnings
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed, cpu_count
import multiprocessing

warnings.filterwarnings('ignore')

print("=" * 60)
print("ADVANCED PREPROCESSING FOR CONVOLUTIONAL VAE (HIGH RES + PARALLEL)")
print("=" * 60)

# ============================================================================
# CELL 1: Configuration and Paths
# ============================================================================

CONFIG = {
    'sample_rate': 22050,           # Standard sample rate for audio processing
    'duration': 30,                 # Duration in seconds to load from each audio file
    'n_mels': 256,                  # INCREASED: Number of mel bands (High Res Height)
    'n_fft': 2048,                  # FFT window size
    'hop_length': 512,              # Hop length for STFT
    'n_mfcc': 60,                   # INCREASED: Number of MFCC coefficients
    'fixed_time_steps': 1024,       # INCREASED: Time dimension (Width) - Captures ~24s
    'max_samples_per_class': 120,   # Maximum samples per class to load (Increased slightly)
    'lyrics_max_features': 100,     # Max TF-IDF features for lyrics
}

# Define paths
BASE_PATH = r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\Datasets"
BANGLA_PATH = os.path.join(BASE_PATH, "Bangla_Datasets")
ENGLISH_PATH = os.path.join(BASE_PATH, "English_Datasets")
METADATA_PATH = os.path.join(BASE_PATH, "updated_metadata.csv")
OUTPUT_PATH = r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\processed_data2"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"\nConfiguration loaded!")
print(f"Bangla datasets path: {BANGLA_PATH}")
print(f"English datasets path: {ENGLISH_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print(f"\nSpectrogram dimensions: {CONFIG['n_mels']} x {CONFIG['fixed_time_steps']} (High Resolution)")

# ============================================================================
# CELL 2: Load Metadata with Lyrics
# ============================================================================

print("\n" + "=" * 60)
print("LOADING METADATA")
print("=" * 60)

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

metadata_df = pd.read_csv(METADATA_PATH)
print(f"Metadata shape: {metadata_df.shape}")

# Create lookup dictionaries
genre_lookup = dict(zip(metadata_df['ID'].astype(str), metadata_df['genre']))
lyrics_lookup = dict(zip(metadata_df['ID'].astype(str), metadata_df['lyrics'].fillna('')))

print(f"Loaded {len(genre_lookup)} genre entries from metadata")
print(f"Loaded {len(lyrics_lookup)} lyrics entries from metadata")

# ============================================================================
# CELL 3: Audio Loading and Feature Extraction Functions
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
        return None, None


def extract_mel_spectrogram(audio, sr):
    """Extract mel spectrogram with fixed dimensions for CNN."""
    mel = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=CONFIG['n_mels'],
        n_fft=CONFIG['n_fft'], 
        hop_length=CONFIG['hop_length']
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Resize to fixed time steps (width)
    if mel_db.shape[1] > CONFIG['fixed_time_steps']:
        # Center crop or take first N steps? Taking first N for consistency.
        mel_db = mel_db[:, :CONFIG['fixed_time_steps']]
    else:
        # Pad with minimum value (silence)
        pad_width = CONFIG['fixed_time_steps'] - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=mel_db.min())
    
    return mel_db


def extract_mfcc(audio, sr):
    """Extract MFCC features with fixed dimensions for CNN."""
    mfcc = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=CONFIG['n_mfcc'],
        n_fft=CONFIG['n_fft'], 
        hop_length=CONFIG['hop_length']
    )
    
    # Resize to fixed time steps
    if mfcc.shape[1] > CONFIG['fixed_time_steps']:
        mfcc = mfcc[:, :CONFIG['fixed_time_steps']]
    else:
        # Pad with minimum value
        pad_width = CONFIG['fixed_time_steps'] - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant', constant_values=mfcc.min())
    
    return mfcc


def extract_flattened_features(audio, sr):
    """
    Extract flattened statistical features (for compatibility with MLP-based models).
    Using reduced statistics to keep vector size reasonable alongside High Res images.
    """
    # Extract mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, # Keep 128 for stat features to match old scaling roughly
        n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length']
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=20, # Keep 20 for stat features
        n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length']
    )
    
    # Extract spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=CONFIG['hop_length'])
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=CONFIG['hop_length'])
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=CONFIG['hop_length'])
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=CONFIG['hop_length'])
    rms = librosa.feature.rms(y=audio, hop_length=CONFIG['hop_length'])
    
    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length']
    )
    
    # Aggregate features
    features = []
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    
    for feat in [spectral_centroid, spectral_bandwidth, spectral_rolloff, zcr, rms]:
        features.append(np.mean(feat))
        features.append(np.std(feat))
    
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))
    
    return np.array(features)

def process_single_file(file_info):
    """
    Worker function for parallel processing. 
    Returns dict with features or None if failed.
    """
    audio, sr = load_audio_file(file_info['path'])
    
    if audio is None:
        return {'status': 'failed', 'path': file_info['path'], 'error': 'Load failed'}
    
    try:
        # Extract features
        mel_spec = extract_mel_spectrogram(audio, sr)
        mfcc = extract_mfcc(audio, sr)
        flat_feat = extract_flattened_features(audio, sr)
        
        return {
            'status': 'success',
            'mel_spec': mel_spec,
            'mfcc': mfcc,
            'flat_feat': flat_feat,
            'genre': file_info['genre'],
            'lyrics': file_info['lyrics'],
            'language': file_info['language'],
            'filename': file_info['filename'],
            'file_id': file_info['file_id']
        }
    except Exception as e:
        return {'status': 'failed', 'path': file_info['path'], 'error': str(e)}

print("Feature extraction functions defined!")

# ============================================================================
# CELL 4: Collect Audio Files with STRICT Filtering
# ============================================================================

def collect_audio_files():
    """Collect all audio file paths with strict lyrics filtering and exclude Jazz."""
    audio_files = []
    skipped_files = 0
    skipped_reasons = {'not_in_metadata': 0, 'empty_lyrics': 0, 'short_lyrics': 0, 'jazz_excluded': 0}
    
    paths_to_check = [
        (BANGLA_PATH, 'bn'),
        (ENGLISH_PATH, 'en')
    ]
    
    print("\nScanning directories...")
    
    for base_path, lang in paths_to_check:
        if not os.path.exists(base_path):
            continue
            
        print(f"Scanning {base_path}...")
        for genre_folder in os.listdir(base_path):
            genre_path = os.path.join(base_path, genre_folder)
            if not os.path.isdir(genre_path):
                continue
                
            files_in_genre = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
            files_in_genre = files_in_genre[:CONFIG['max_samples_per_class']]
            
            for audio_file in files_in_genre:
                file_id = os.path.splitext(audio_file)[0]
                
                # Check 1: Existence in metadata
                if file_id not in genre_lookup:
                    skipped_files += 1
                    skipped_reasons['not_in_metadata'] += 1
                    continue
                
                current_genre = genre_lookup[file_id]
                
                # Check 1.5: Exclude Jazz
                if str(current_genre).strip().lower() == 'jazz':
                    skipped_files += 1
                    skipped_reasons['jazz_excluded'] += 1
                    continue
                
                lyrics = lyrics_lookup.get(file_id, '')
                
                # Check 2: Strict Lyrics Validation
                if not isinstance(lyrics, str):
                    skipped_files += 1
                    skipped_reasons['empty_lyrics'] += 1
                    continue
                
                clean_lyrics = lyrics.strip()
                # Remove common placeholders
                if clean_lyrics.lower() in ['nan', 'none', 'null', 'instrumental', '', ' ']:
                    skipped_files += 1
                    skipped_reasons['empty_lyrics'] += 1
                    continue
                    
                # Check 3: Length
                if len(clean_lyrics) < 15: # Increased threshold strictly
                    skipped_files += 1
                    skipped_reasons['short_lyrics'] += 1
                    continue
                    
                audio_files.append({
                    'path': os.path.join(genre_path, audio_file),
                    'language': lang,
                    'genre': current_genre,
                    'filename': audio_file,
                    'file_id': file_id,
                    'lyrics': lyrics
                })
    
    print(f"\nTotal audio files collected: {len(audio_files)}")
    print(f"Skipped files summary:")
    print(f"  - Not in metadata: {skipped_reasons['not_in_metadata']}")
    print(f"  - Jazz excluded: {skipped_reasons['jazz_excluded']}")
    print(f"  - Empty/Invalid lyrics: {skipped_reasons['empty_lyrics']}")
    print(f"  - Length too short (<15 chars): {skipped_reasons['short_lyrics']}")
    
    return audio_files

print("\n" + "=" * 60)
print("COLLECTING AUDIO FILES")
print("=" * 60)

audio_files = collect_audio_files()

if len(audio_files) == 0:
    raise ValueError("No audio files collected! Check paths and metadata.")

# ============================================================================
# CELL 5: Process All Audio Files (PARALLEL CPU)
# ============================================================================

print("\n" + "=" * 60)
print(f"EXTRACTING FEATURES (Using {cpu_count()} CPU Cores)")
print("=" * 60)

# Run parallel processing
# n_jobs=-1 uses all available cores
results = Parallel(n_jobs=-1, verbose=5)(
    delayed(process_single_file)(f) for f in audio_files
)

# Aggregate results
mel_spectrograms = []
mfccs = []
flattened_features = []
labels = []
lyrics_list = []
metadata_list = []
failed_count = 0

for res in results:
    if res['status'] == 'success':
        mel_spectrograms.append(res['mel_spec'])
        mfccs.append(res['mfcc'])
        flattened_features.append(res['flat_feat'])
        labels.append(res['genre'])
        lyrics_list.append(res['lyrics'])
        metadata_list.append({
            'language': res['language'],
            'genre': res['genre'],
            'filename': res['filename'],
            'file_id': res['file_id']
        })
    else:
        failed_count += 1

# Convert to numpy arrays
mel_spectrograms = np.array(mel_spectrograms)
mfccs = np.array(mfccs)
flattened_features = np.array(flattened_features)
labels = np.array(labels)

print(f"\nSuccessfully processed: {len(mel_spectrograms)} files")
print(f"Failed processing: {failed_count} files")

# ============================================================================
# CELL 6: Create Lyrics Embeddings
# ============================================================================

print("\n" + "=" * 60)
print("CREATING LYRICS EMBEDDINGS")
print("=" * 60)

def create_lyrics_embeddings(lyrics_list, max_features=100):
    """Create TF-IDF embeddings for lyrics."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    # Handle empty lyrics (though should be filtered now)
    lyrics_cleaned = [l if l and len(str(l)) > 0 else ' ' for l in lyrics_list]
    embeddings = vectorizer.fit_transform(lyrics_cleaned).toarray()
    return embeddings, vectorizer

lyrics_embeddings, tfidf_vectorizer = create_lyrics_embeddings(
    lyrics_list, 
    max_features=CONFIG['lyrics_max_features']
)
print(f"Lyrics embeddings shape: {lyrics_embeddings.shape}")

# Ensure we have consistency
assert len(mel_spectrograms) == len(lyrics_embeddings), "Mismatch between audio and lyrics samples!"

# ============================================================================
# CELL 7: Display Feature Statistics
# ============================================================================

print("\n" + "=" * 60)
print("FEATURE EXTRACTION SUMMARY")
print("=" * 60)

print(f"\n1. Mel Spectrograms (for CNN):")
print(f"   Shape: {mel_spectrograms.shape}")
print(f"   Dimensions: {mel_spectrograms.shape[1]} (mel bands) x {mel_spectrograms.shape[2]} (time steps)")

print(f"\n2. MFCCs (for CNN):")
print(f"   Shape: {mfccs.shape}")

print(f"\n3. Flattened Features (for MLP):")
print(f"   Shape: {flattened_features.shape}")

print(f"\n4. Lyrics Embeddings:")
print(f"   Shape: {lyrics_embeddings.shape}")

print(f"\n5. Total Valid Samples: {len(labels)}")

# ============================================================================
# CELL 8: Normalize Features
# ============================================================================

print("\n" + "=" * 60)
print("NORMALIZING FEATURES")
print("=" * 60)

# Normalize mel spectrograms (standard scaler on flattened, then reshape)
# Using memory-efficient approach if possible, but for 2000 samples, fit_transform is fine
mel_scaler = StandardScaler()
N, H, W = mel_spectrograms.shape
mel_flat = mel_spectrograms.reshape(N, -1)
mel_normalized_flat = mel_scaler.fit_transform(mel_flat)
mel_normalized = mel_normalized_flat.reshape(N, H, W)

# Normalize MFCCs
mfcc_scaler = StandardScaler()
N_mfcc, H_mfcc, W_mfcc = mfccs.shape
mfcc_flat = mfccs.reshape(N_mfcc, -1)
mfcc_normalized_flat = mfcc_scaler.fit_transform(mfcc_flat)
mfcc_normalized = mfcc_normalized_flat.reshape(N_mfcc, H_mfcc, W_mfcc)

# Normalize flattened features
from sklearn.impute import SimpleImputer
features_clean = np.where(np.isinf(flattened_features), np.nan, flattened_features)
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features_clean)

flat_scaler = StandardScaler()
features_normalized = flat_scaler.fit_transform(features_imputed)

print("Normalization complete.")

# ============================================================================
# CELL 9: Save Preprocessed Data
# ============================================================================

print("\n" + "=" * 60)
print("SAVING PREPROCESSED DATA")
print("=" * 60)

# Create DataFrame with metadata
metadata_df_out = pd.DataFrame(metadata_list)
metadata_df_out['label'] = labels

# Save 2D features for CNN
np.save(os.path.join(OUTPUT_PATH, 'mel_spectrograms_raw.npy'), mel_spectrograms)
np.save(os.path.join(OUTPUT_PATH, 'mel_spectrograms_normalized.npy'), mel_normalized)
np.save(os.path.join(OUTPUT_PATH, 'mfccs_raw.npy'), mfccs)
np.save(os.path.join(OUTPUT_PATH, 'mfccs_normalized.npy'), mfcc_normalized)

# Save flattened features
np.save(os.path.join(OUTPUT_PATH, 'features_raw.npy'), flattened_features)
np.save(os.path.join(OUTPUT_PATH, 'features_normalized.npy'), features_normalized)

# Save lyrics embeddings & labels
np.save(os.path.join(OUTPUT_PATH, 'lyrics_embeddings.npy'), lyrics_embeddings)
np.save(os.path.join(OUTPUT_PATH, 'labels.npy'), labels)
metadata_df_out.to_csv(os.path.join(OUTPUT_PATH, 'metadata.csv'), index=False)

# Save scalers
with open(os.path.join(OUTPUT_PATH, 'mel_scaler.pkl'), 'wb') as f: pickle.dump(mel_scaler, f)
with open(os.path.join(OUTPUT_PATH, 'mfcc_scaler.pkl'), 'wb') as f: pickle.dump(mfcc_scaler, f)
with open(os.path.join(OUTPUT_PATH, 'flat_scaler.pkl'), 'wb') as f: pickle.dump(flat_scaler, f)
with open(os.path.join(OUTPUT_PATH, 'imputer.pkl'), 'wb') as f: pickle.dump(imputer, f)
with open(os.path.join(OUTPUT_PATH, 'tfidf_vectorizer.pkl'), 'wb') as f: pickle.dump(tfidf_vectorizer, f)
with open(os.path.join(OUTPUT_PATH, 'config.pkl'), 'wb') as f: pickle.dump(CONFIG, f)

print(f"\nFiles saved to: {OUTPUT_PATH}")
print(f"Mel Spectrogram resolution: {mel_normalized.shape[1:]}")
print("Pre-processing complete! You can now run the VAE training.")
