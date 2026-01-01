"""
Generate Lyrics for Bangla and English Datasets
Uses Whisper 'large' model to transcribe audio and updates metadata.
"""

import os
import csv
import torch
import whisper
from tqdm import tqdm
from pathlib import Path
import glob

# Configuration
BASE_DIR = r"f:\BRACU\Semester 12 Final\CSE425\Projectto2\Datasets"
METADATA_FILE = os.path.join(BASE_DIR, "updated_metadata.csv")
BANGLA_DIR = os.path.join(BASE_DIR, "Bangla_Datasets")
ENGLISH_DIR = os.path.join(BASE_DIR, "English_Datasets")
MODEL_SIZE = "large-v3"

def get_actual_genre_map(base_dir):
    """
    Returns a dictionary mapping lowercase genre to actual folder name.
    e.g. {'adhunik': 'Adhunik', 'rock': 'rock'}
    """
    mapping = {}
    if os.path.exists(base_dir):
        for name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, name)):
                mapping[name.lower()] = name
    return mapping

def find_audio_file(row, bangla_genre_map, english_genre_map):
    """
    Finds the audio file path based on metadata row.
    """
    file_id = row['ID']
    language = row['language']
    genre = row['genre']
    genre_lower = genre.lower()
    
    # Determine search directory map based on language
    if language == 'bn':
        search_dirs = [(BANGLA_DIR, bangla_genre_map)]
    elif language == 'en':
        search_dirs = [(ENGLISH_DIR, english_genre_map)]
    else:
        # Fallback: check both
        search_dirs = [(BANGLA_DIR, bangla_genre_map), (ENGLISH_DIR, english_genre_map)]
    
    for root_dir, genre_map in search_dirs:
        # Resolve actual genre folder name
        actual_genre = genre_map.get(genre_lower)
        if not actual_genre:
            # Try direct match just in case
            if os.path.exists(os.path.join(root_dir, genre)):
                actual_genre = genre
            else:
                continue
                
        # Construct path
        # Try exact likely extensions
        for ext in ['.wav', '.mp3']:
            path = os.path.join(root_dir, actual_genre, f"{file_id}{ext}")
            if os.path.exists(path):
                return path
            
    return None

def main():
    print("=" * 60)
    print(f"Lyrics Generator using Whisper '{MODEL_SIZE}'")
    print("=" * 60)

    # 1. Load Whisper Model
    print(f"Loading Whisper model ({MODEL_SIZE})... this may take time and RAM.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = whisper.load_model(MODEL_SIZE, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Prepare Genre Maps (to handle capitalization differences)
    bangla_genre_map = get_actual_genre_map(BANGLA_DIR)
    english_genre_map = get_actual_genre_map(ENGLISH_DIR)

    # 3. Load Metadata
    print(f"Reading metadata: {METADATA_FILE}")
    rows = []
    fieldnames = []
    
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if 'lyrics' not in fieldnames:
        print("Adding 'lyrics' column...")
        fieldnames.append('lyrics')

    total_files = len(rows)
    print(f"Found {total_files} tracks to process.")

    # 4. Processing Loop
    success_count = 0
    skip_count = 0
    fail_count = 0
    save_interval = 10 

    for i, row in enumerate(tqdm(rows, desc="Transcribing")):
        # Skip if lyrics already exist and are not empty
        if 'lyrics' in row and row['lyrics'] and len(row['lyrics'].strip()) > 0:
            skip_count += 1
            continue
        
        # Determine language for whisper hint
        lang_code = row['language'] if row['language'] in ['en', 'bn'] else None
        
        # Find Audio File
        audio_path = find_audio_file(row, bangla_genre_map, english_genre_map)
        
        if audio_path:
            try:
                # Transcribe
                # Using fp16=False to be safe on CPU or older GPUs if mixed precision fails
                result = model.transcribe(audio_path, language=lang_code, fp16=(device=="cuda"))
                text = result['text'].strip()
                result_text = text.replace('\n', ' ') # Flatten to single line for CSV safety
                row['lyrics'] = result_text
                success_count += 1
            except Exception as e:
                print(f"\nError transcribing {audio_path}: {e}")
                row['lyrics'] = "" # failure placeholder
                fail_count += 1
        else:
            # print(f"\nAudio not found for ID: {row['ID']} (Genre: {row['genre']})") 
            row['lyrics'] = ""
            fail_count += 1
            
        # Periodic Save
        if (i + 1) % save_interval == 0:
            with open(METADATA_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    # Final Save
    print("Saving final updated metadata...")
    with open(METADATA_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print(f"Processing Complete.")
    print(f"Total: {total_files}")
    print(f"Successful: {success_count}")
    print(f"Skipped (Already existed): {skip_count}")
    print(f"Failed (Missing file/Error): {fail_count}")
    print(f"Updated file: {METADATA_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
