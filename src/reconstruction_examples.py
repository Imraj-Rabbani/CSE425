import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import MusicDataset
from beta_vae import BetaVAE

MODEL_PATH = "results/beta_vae_beta4.pth"

dataset = MusicDataset("Dataset/CSV/dataset.csv")

model = BetaVAE()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

audio, lyrics = dataset[0]

with torch.no_grad():
    recon, _, _, _ = model(audio.unsqueeze(0), lyrics.unsqueeze(0))

orig = audio.squeeze().numpy()
recon = recon.squeeze().numpy()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(orig, aspect="auto", origin="lower")
plt.title("Original Mel Spectrogram")

plt.subplot(1, 2, 2)
plt.imshow(recon, aspect="auto", origin="lower")
plt.title("Reconstructed Mel Spectrogram")

plt.tight_layout()
plt.show()
