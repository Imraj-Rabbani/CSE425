import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def normalize_track_name(name):
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return f"{name}_30s"


class MusicDataset(Dataset):
    def __init__(self, csv_path):
        raw_df = pd.read_csv(csv_path)
        self.data = []
        
        print(f"Scanning {len(raw_df)} samples for valid features...")
        valid_count = 0
        
        for _, row in raw_df.iterrows():
            name = normalize_track_name(row["track_name"])
            
            # Check for suffixes (the extract script saves as .npy)
            # The name returned by normalize_track_name does NOT have .wav extension in the string,
            # but the extract script saves files as "{name}_30s.wav.npy" or similar?
            # Wait, let's verify extract_features.py logic.
            # It did: return f"{name}_30s.wav"
            # And saved as: np.save(f"{AUDIO_OUT}/{filename.replace('.wav', '')}.npy", mel)
            # So if normalize returns "foo_30s.wav", filename is "foo_30s.wav".
            # Saved as "features/audio/foo_30s.npy".
            # So here we need to be careful.
            
            # normalize_track_name in this file (dataset.py) returns f"{name}_30s" (NO .wav)
            # So audio_path = f"features/audio/{name}.npy" -> "features/audio/foo_30s.npy".
            # This matches "foo_30s.wav".replace(".wav", "") + ".npy".
            # So the logic seems correct.
            
            audio_path = f"features/audio/{name}.npy"
            lyrics_path = f"features/lyrics/{name}.npy" 
            
            if os.path.exists(audio_path) and os.path.exists(lyrics_path):
                self.data.append({
                    "name": name, 
                    "audio_path": audio_path, 
                    "lyrics_path": lyrics_path
                })
                valid_count += 1
                
        print(f"Kept {valid_count} valid samples out of {len(raw_df)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            audio = np.load(item["audio_path"])
            lyrics = np.load(item["lyrics_path"])

            audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            lyrics = torch.tensor(lyrics, dtype=torch.float32)

            return audio, lyrics
        except Exception as e:
            # Fallback in case file gets deleted or corrupted
            print(f"Error loading {item['name']}: {e}")
            # return zero tensors or handle appropriately?
            # For simplicity let's crash so we know, or verify.
            raise e
