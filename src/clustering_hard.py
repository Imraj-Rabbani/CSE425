import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

from dataset import MusicDataset
from beta_vae import BetaVAE

# ---------------- CONFIG ----------------
N_CLUSTERS = 8
MODEL_PATH = "results/beta_vae_beta4.pth"
CSV_PATH = "Dataset/CSV/dataset.csv"
# ---------------------------------------

def cluster_purity(y_true, y_pred):
    contingency = pd.crosstab(y_pred, y_true)
    return np.sum(np.max(contingency.values, axis=1)) / np.sum(contingency.values)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MusicDataset(CSV_PATH)
df = pd.read_csv(CSV_PATH)

model = BetaVAE().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

latents = []
genres = []

with torch.no_grad():
    for i in range(len(dataset)):
        try:
            audio, lyrics = dataset[i]
            audio = audio.unsqueeze(0).to(device)
            lyrics = lyrics.unsqueeze(0).to(device)

            _, _, _, z = model(audio, lyrics)
            latents.append(z.squeeze(0).cpu().numpy())
            genres.append(df.iloc[i]["playlist_genre"])
        except:
            continue

X = np.array(latents)
y_true = np.array(genres)

print("Latent shape:", X.shape)

# -------- Clustering --------
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
y_pred = kmeans.fit_predict(X)

# -------- Metrics --------
sil = silhouette_score(X, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
ari = adjusted_rand_score(y_true, y_pred)
purity = cluster_purity(y_true, y_pred)

print("\n--- HARD TASK METRICS ---")
print(f"Silhouette Score : {sil:.4f}")
print(f"NMI              : {nmi:.4f}")
print(f"ARI              : {ari:.4f}")
print(f"Cluster Purity   : {purity:.4f}")

# Save for later steps
np.save("results/hard_latents.npy", X)
np.save("results/hard_clusters.npy", y_pred)
np.save("results/hard_genres.npy", y_true)