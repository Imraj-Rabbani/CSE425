import torch
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

from dataset import MusicDataset
from vae import VAE

dataset = MusicDataset("Dataset/CSV/dataset.csv")

model = VAE()
model.load_state_dict(torch.load("results/vae.pth", map_location="cpu"))
model.eval()

latents = []

with torch.no_grad():
    for i in range(len(dataset)):
        try:
            audio, lyrics = dataset[i]
            _, _, _, z = model(audio.unsqueeze(0), lyrics.unsqueeze(0))
            latents.append(z.squeeze(0).numpy())
        except Exception:
            continue

X = np.array(latents)
np.save("results/latent_vectors.npy", X)

print("Latent vectors shape:", X.shape)

clusterers = {
    "KMeans": KMeans(n_clusters=5, random_state=42),
    "Agglomerative": AgglomerativeClustering(n_clusters=5),
    "DBSCAN": DBSCAN(eps=1.5)
}

for name, model in clusterers.items():
    labels = model.fit_predict(X)

    if len(set(labels)) > 1:
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        print(f"{name} → Silhouette: {sil:.3f}, Davies-Bouldin: {db:.3f}")
    else:
        print(f"{name} → Single cluster (metric skipped)")
