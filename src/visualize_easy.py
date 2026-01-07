import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans

Z = np.load("results/easy_latent.npy")

labels = KMeans(n_clusters=5, random_state=42).fit_predict(Z)
emb = umap.UMAP(random_state=42).fit_transform(Z)

plt.figure(figsize=(8, 6))
plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10")
plt.title("Easy Task: VAE Latent Space (UMAP)")
plt.tight_layout()
plt.show()
