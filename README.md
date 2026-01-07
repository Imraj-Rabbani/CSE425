
# CSE425: Multi-Modal Music Representation Learning using VAEs

This repository contains the implementation for a course project on **unsupervised music representation learning and clustering** using **Variational Autoencoders (VAEs)**.  
The project explores progressively complex settings, starting from a basic VAE to a multi-modal VAE and finally a Beta-VAE for disentangled latent representations.



---

## 📂 Project Structure

```

CSE425/
├── src/
│   ├── extract_features.py        # Audio + lyrics feature extraction
│   ├── train_vae.py               # Basic VAE (Easy Task)
│   ├── clustering_easy.py         # PCA & VAE clustering (Easy Task)
│   ├── train_multimodal_vae.py    # Multi-modal VAE (Medium Task)
│   ├── clustering.py              # Multi-modal clustering (Medium Task)
│   ├── train_beta_vae.py          # Beta-VAE (Hard Task)
│   ├── clustering_hard.py         # Advanced clustering + metrics
│   └── visualize.py               # UMAP / t-SNE visualizations
│
├── features/
│   ├── audio/                     # Extracted audio features
│   └── lyrics/                    # Extracted lyric embeddings
│
├── results/
│   ├── easy/                      # Easy task results
│   ├── medium/                    # Medium task results
│   └── hard/                      # Hard task results
│
├── figures/                       # Figures used in the report
│
├── Dataset/                       # (Not included in repo)
│   ├── Audio/
│   └── CSV/
│
└── requirements.txt
└── README.md

````

---

## 📦 Dataset

The dataset used in this project is **not publicly included** in the repository.

It consists of:
- Multilingual songs
- Fields: `track_name`, `lyrics`, `genre`
- Audio generated from lyrics using a text-to-speech pipeline

📩 **If you require access to the dataset for academic purposes, please contact me via GitHub or Email.**

---

## ⚙️ Environment Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Imraj-Rabbani/CSE425.git
cd CSE425
````

### 2️⃣ Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```


---

## 🔧 Step 1: Feature Extraction (Required for All Tasks)

Before running any task, extract features from audio and lyrics:

```bash
python3 src/extract_features.py
```

This will generate:

* `features/audio/*.npy`
* `features/lyrics/*.npy`

---

## 🟢 Easy Task: Basic VAE + Clustering

### Objective

* Train a basic VAE on audio features
* Perform clustering using K-Means
* Compare with PCA baseline
* Visualize latent space

### Commands

```bash
python3 src/train_vae.py
python3 src/clustering_easy.py
```

### Outputs

* VAE model checkpoint
* PCA vs VAE clustering metrics
* Latent space visualizations (UMAP / t-SNE)

---

## 🟡 Medium Task: Multi-Modal VAE (Audio + Lyrics)

### Objective

* Learn joint representations using audio and lyrics
* Perform clustering with multiple algorithms
* Analyze multi-modal latent space

### Commands

```bash
python3 src/train_multimodal_vae.py
python3 src/clustering.py
python3 src/visualize.py
```

### Outputs

* Multi-modal VAE model
* Clustering metrics (Silhouette, Davies-Bouldin)
* Multi-modal latent space plots

---

## 🔴 Hard Task: Beta-VAE + Advanced Evaluation

### Objective

* Train a Beta-VAE for disentangled representations
* Perform multi-modal clustering
* Evaluate using advanced metrics (ARI, NMI, Purity)
* Analyze reconstructions and genre alignment

### Commands

```bash
python3 src/train_beta_vae.py
python3 src/clustering_hard.py
python3 src/visualize.py
```

### Outputs

* Beta-VAE model
* Advanced clustering metrics
* Disentangled latent space visualizations
* Reconstruction examples

---

## 📊 Evaluation Metrics Used

* Silhouette Score
* Calinski–Harabasz Index
* Davies–Bouldin Index
* Normalized Mutual Information (NMI)
* Adjusted Rand Index (ARI)
* Cluster Purity

---

## 📈 Visualizations

All plots generated during experiments are saved in:

```
figures/
```

These include:

* Latent space UMAP/t-SNE plots
* Genre distribution across clusters
* Reconstruction comparisons

---

## 🔁 Reproducibility

* All experiments are deterministic given the same dataset
* Scripts are modular and task-specific
* Clear separation between Easy, Medium, and Hard tasks

---

## 📌 Notes

* Training was performed on CPU-based systems
* Results may vary slightly depending on hardware and dataset size
* Genre labels are used **only for evaluation**, not during training

---

## 📫 Contact

For dataset access or questions:

* GitHub: [https://github.com/Imraj-Rabbani](https://github.com/Imraj-Rabbani)
* Email: imraj.rabbani@g.bracu.ac.bd
