import os
import librosa
import numpy as np
import pandas as pd

AUDIO_DIR = "Dataset/Audio"
CSV_PATH = "Dataset/CSV/dataset.csv"
OUT_PATH = "data/easy_features.npy"

os.makedirs("data", exist_ok=True)

df = pd.read_csv(CSV_PATH)
features = []

for _, row in df.iterrows():
    track = row["track_name"].lower().replace(" ", "_") + "_30s.wav"
    path = os.path.join(AUDIO_DIR, track)

    if not os.path.exists(path):
        continue

    y, sr = librosa.load(path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = np.mean(mfcc, axis=1)
    features.append(mfcc)

features = np.array(features)
np.save(OUT_PATH, features)

print(f"Easy task features saved to {OUT_PATH}")
