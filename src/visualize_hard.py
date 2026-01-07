import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd

X = np.load("results/hard_latents.npy")
clusters = np.load("results/hard_clusters.npy")
genres = np.load("results/hard_genres.npy")

reducer = umap.UMAP(random_state=42)
X_2d = reducer.fit_transform(X)

# -------- Plot by Cluster --------
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap="tab10", s=10)
plt.title("Beta-VAE Latent Space (Colored by Cluster)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# -------- Plot by Genre --------
genre_codes = pd.Categorical(genres).codes

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=genre_codes, cmap="tab20", s=10)
plt.title("Beta-VAE Latent Space (Colored by Genre)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.colorbar(label="Genre")
plt.tight_layout()
plt.show()
