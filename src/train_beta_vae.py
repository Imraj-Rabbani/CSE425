import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MusicDataset
from beta_vae import BetaVAE

device = "cuda" if torch.cuda.is_available() else "cpu"

BETA = 4.0   # <-- KEY HARD TASK PARAMETER

dataset = MusicDataset("Dataset/CSV/dataset.csv")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = BetaVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Reduced LR


def beta_vae_loss(recon, x, mu, logvar):
    # MSE Loss
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    
    # KL Divergence
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + BETA * kl


print(f"Training Beta-VAE with beta = {BETA}")

for epoch in range(12):
    total = 0
    batches = 0

    for audio, lyrics in tqdm(loader):
        try:
            # Normalize audio from approx [-80, 0] to [0, 1]
            audio = (audio + 80.0) / 80.0
            audio = torch.clamp(audio, 0.0, 1.0)
            
            audio = audio.to(device)
            lyrics = lyrics.to(device)

            recon, mu, logvar, _ = model(audio, lyrics)
            loss = beta_vae_loss(recon, audio, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total += loss.item()
            batches += 1

        except Exception as e:
            continue

    print(f"Epoch {epoch + 1}, Loss: {total / max(1, batches):.4f}")

os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), f"results/beta_vae_beta{int(BETA)}.pth")
print(f"Model saved to results/beta_vae_beta{int(BETA)}.pth")
