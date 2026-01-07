import torch
import torch.nn.functional as F
import numpy as np
from vae_easy import EasyVAE

X = np.load("data/easy_features.npy")
X = torch.tensor(X, dtype=torch.float32)

model = EasyVAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    recon, mu, logvar, z = model(X)

    recon_loss = F.mse_loss(recon, X)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "results/easy_vae.pth")
print("Easy VAE saved to results/easy_vae.pth")
