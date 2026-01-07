import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MusicDataset
from vae import VAE

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MusicDataset("Dataset/CSV/dataset.csv")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def loss_fn(recon, x, mu, logvar):
    # MSE Loss
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    
    # KL Divergence
    # We allow some KL divergence early on? Or just standard VAE.
    # Standard: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Weight KL slightly less if needed, but standard is just sum.
    # Note: If recon moves to [0,1], MSE will be small. KL might dominate.
    # Let's reduce KL weight slightly to encourage reconstruction first.
    return recon_loss + 0.0001 * kl 


print("Starting training...")
for epoch in range(12):
    total_loss = 0
    batches = 0

    for audio, lyrics in tqdm(loader):
        try:
            # Normalize audio from approx [-80, 0] dB to [0, 1]
            # Since silence is -80, max is 0. 
            # (val + 80) / 80 -> -80->0, 0->1.
            audio = (audio + 80.0) / 80.0
            
            # Simple clamp to ensure we stay in reasonable bounds if outliers exist
            audio = torch.clamp(audio, 0.0, 1.0)
            
            audio = audio.to(device)
            lyrics = lyrics.to(device)

            recon, mu, logvar, _ = model(audio, lyrics)
            loss = loss_fn(recon, audio, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        except Exception as e:
            # print(f"Skipping batch: {e}")
            continue

    print(f"Epoch {epoch + 1}, Loss: {total_loss / max(1, batches):.6f}")

# Ensure directory exists
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/vae.pth")
print("Model saved to results/vae.pth")
