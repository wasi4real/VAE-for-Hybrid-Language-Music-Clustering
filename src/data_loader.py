# Install required packages
!pip install -q librosa scikit-learn umap-learn pandas matplotlib seaborn kagglehub sentence-transformers torch

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sentence_transformers import SentenceTransformer
import joblib
import shutil

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set base directory
BASE_DIR = "/content/drive/MyDrive/CSE425_Project_Runtime"

# Create project structure
dirs = [
    "data/processed",
    "results/plots",
    "results/metrics",
    "models",
    "deployment"
]

for d in dirs:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

"""# Data Acquisition"""

import kagglehub

# Download datasets
print("Downloading datasets...")
bangla_path = kagglehub.dataset_download("thisisjibon/banglabeats3sec")
gtzan_path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

# Set directories based on discovered structure
BANGLABEATS_DIR = os.path.join(bangla_path, "wavs3sec")
GTZAN_DIR = os.path.join(gtzan_path, "Data", "genres_original")

print(f"BanglaBeats directory: {BANGLABEATS_DIR}")
print(f"GTZAN directory: {GTZAN_DIR}")

# Load BanglaBeats dataset
bangla_data = []
genres_bangla = [d for d in os.listdir(BANGLABEATS_DIR)
                 if os.path.isdir(os.path.join(BANGLABEATS_DIR, d))]

for genre in genres_bangla:
    genre_path = os.path.join(BANGLABEATS_DIR, genre)
    wav_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]

    for file_name in wav_files:
        bangla_data.append({
            'id': file_name.split('.')[0],
            'title': file_name,
            'language': 'bangla',
            'genre': genre,
            'file_path': os.path.join(genre_path, file_name)
        })

df_bangla = pd.DataFrame(bangla_data)

# Load GTZAN dataset
gtzan_data = []
genres_gtzan = [d for d in os.listdir(GTZAN_DIR)
                if os.path.isdir(os.path.join(GTZAN_DIR, d))]

for genre in genres_gtzan:
    genre_path = os.path.join(GTZAN_DIR, genre)
    wav_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]

    for file_name in wav_files:
        gtzan_data.append({
            'id': file_name.split('.')[0],
            'title': file_name,
            'language': 'english',
            'genre': genre,
            'file_path': os.path.join(genre_path, file_name)
        })

df_english = pd.DataFrame(gtzan_data)

print(f"Bangla tracks: {len(df_bangla)}")
print(f"English tracks: {len(df_english)}")

"""Dataset Balancing & Merging"""

# Balance by sampling equal number from each language
samples_per_language = min(1000, len(df_bangla), len(df_english))
df_bangla_balanced = df_bangla.sample(n=samples_per_language, random_state=42)
df_english_balanced = df_english.sample(n=samples_per_language, random_state=42)

# Merge datasets
df_hybrid = pd.concat([df_bangla_balanced, df_english_balanced], ignore_index=True)

# Save metadata
df_hybrid.to_csv(os.path.join(BASE_DIR, "data/processed/hybrid_metadata.csv"), index=False)

print(f"Final dataset: {len(df_hybrid)} tracks")
print("Language distribution:")
print(df_hybrid['language'].value_counts())