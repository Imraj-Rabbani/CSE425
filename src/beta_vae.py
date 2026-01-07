import torch
import torch.nn as nn


class BetaVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # -------- Encoder --------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64 * 32 * 75 + 384, latent_dim)
        self.fc_logvar = nn.Linear(64 * 32 * 75 + 384, latent_dim)

        # -------- Decoder --------
        self.fc_decode = nn.Linear(latent_dim, 64 * 32 * 75)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, audio, lyrics):
        x = self.encoder(audio)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, lyrics], dim=1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        h = self.fc_decode(z)
        h = h.view(-1, 64, 32, 75)
        recon = self.decoder(h)

        return recon, mu, logvar, z
