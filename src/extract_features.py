import os
import re
import librosa
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

AUDIO_DIR = "Dataset/Audio"
CSV_PATH = "Dataset/CSV/dataset.csv"

AUDIO_OUT = "features/audio"
LYRICS_OUT = "features/lyrics"

os.makedirs(AUDIO_OUT, exist_ok=True)
os.makedirs(LYRICS_OUT, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Read CSV
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

def normalize_track_name(name):
    # Lowercase
    name = name.lower()
    # Replace non-alphanumeric with underscore
    name = re.sub(r'[^a-z0-9]', '_', name)
    # Merge multiple underscores
    name = re.sub(r'_+', '_', name)
    # Strip leading/trailing underscores
    name = name.strip('_')
    # Append suffix
    return f"{name}_30s.wav"

print(f"Processing {len(df)} tracks...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    name = row["track_name"]
    lyrics = row["lyrics"]

    # -------- Audio Feature (Mel Spectrogram) --------
    filename = normalize_track_name(name)
    audio_path = os.path.join(AUDIO_DIR, filename)
    
    if not os.path.exists(audio_path):
        # file missing, skip
        continue

    try:
        # Load audio with librosa
        # Verify it loads correctly
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Calculate Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = librosa.util.fix_length(mel, size=300, axis=1)

        # Save audio features
        np.save(f"{AUDIO_OUT}/{filename.replace('.wav', '')}.npy", mel)

        # -------- Lyrics Embedding --------
        # Embed lyrics
        if pd.notna(lyrics):
            emb = model.encode(str(lyrics))
            np.save(f"{LYRICS_OUT}/{filename.replace('.wav', '')}.npy", emb)
            
    except Exception as e:
        print(f"Error processing {name} ({filename}): {e}")

print("Feature extraction completed.")
