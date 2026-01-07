import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from vae_easy import EasyVAE

X = np.load("data/easy_features.npy")
Xt = torch.tensor(X, dtype=torch.float32)

model = EasyVAE()
model.load_state_dict(torch.load("results/easy_vae.pth"))
model.eval()

with torch.no_grad():
    _, _, _, Z = model(Xt)

Z = Z.numpy()

kmeans = KMeans(n_clusters=5, random_state=42)
labels_vae = kmeans.fit_predict(Z)

sil_vae = silhouette_score(Z, labels_vae)
ch_vae = calinski_harabasz_score(Z, labels_vae)

Z_pca = PCA(n_components=8).fit_transform(X)
labels_pca = kmeans.fit_predict(Z_pca)

sil_pca = silhouette_score(Z_pca, labels_pca)
ch_pca = calinski_harabasz_score(Z_pca, labels_pca)

print("\n--- EASY TASK RESULTS ---")
print(f"VAE + KMeans → Silhouette: {sil_vae:.3f}, CH: {ch_vae:.1f}")
print(f"PCA + KMeans → Silhouette: {sil_pca:.3f}, CH: {ch_pca:.1f}")

np.save("results/easy_latent.npy", Z)
