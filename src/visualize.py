import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans

X = np.load("results/latent_vectors.npy")

reducer = umap.UMAP(random_state=42)
X_2d = reducer.fit_transform(X)

labels = KMeans(n_clusters=5, random_state=42).fit_predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10")
plt.title("Latent Space Clustering (UMAP)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()
