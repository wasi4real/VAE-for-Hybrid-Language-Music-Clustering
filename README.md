# VAE for Hybrid Language Music Clustering

Unsupervised learning pipeline for clustering English and Bangla music tracks using Variational Autoencoders (VAEs). This project implements three VAE architectures with multi-modal audio-text features and evaluates clustering performance across multiple algorithms and metrics.

## 📋 Project Overview

This project addresses the challenge of cross-lingual music clustering by implementing a progressive VAE framework:

- **Easy Task**: Basic VAE for audio feature extraction + K-Means
- **Medium Task**: Convolutional VAE + hybrid features + advanced clustering
- **Hard Task**: Conditional Beta-VAE + multi-modal clustering with disentanglement

## 🏗️ Architecture

### VAE Variants
1. **Basic VAE**: Linear encoder-decoder with 16D latent space
2. **Conv VAE**: 1D convolutional architecture for MFCC sequences
3. **Conditional Beta-VAE**: Genre-conditioned with β=4.0 for disentanglement

### Feature Engineering
- **Audio**: 13 MFCC coefficients with statistical moments (52D)
- **Text**: Audio-derived descriptions embedded via multilingual SentenceTransformer (384D)
- **Hybrid**: Concatenated audio + text features (436D)

### Clustering Methods
- K-Means (baseline)
- Agglomerative Clustering
- DBSCAN (original + UMAP space)
- Spectral Clustering
- PCA + K-Means

## 📊 Results

### Key Metrics (Top Performers)

| Method | Task | Silhouette | ARI | NMI | Purity |
|--------|------|------------|-----|-----|--------|
| Agglomerative | Hard | 0.7527 | 0.0043 | 0.0331 | 0.0945 |
| Hybrid K-Means | Medium | 0.7479 | 0.0042 | 0.0345 | 0.0960 |
| DBSCAN (UMAP) | Medium | 0.5309 | 0.0218 | 0.1061 | 0.1311 |
| Cond VAE | Hard | -0.0463 | 0.0749 | 0.1742 | 0.1761 |

### Key Findings
1. VAE latent spaces require UMAP projection for effective clustering
2. Audio-derived text features prevent data leakage from genre labels
3. Agglomerative clustering on hybrid features performs best
4. Conditional VAE achieves best latent space organization but lower reconstruction

## 🗂️ Repository Structure

```text
CSE425_Project/
├── README.md
├── data/
│   ├── raw/ (empty - datasets downloaded via kagglehub)
│   └── processed/ (final dataframe: hybrid_metadata.csv, scalers, latent_features.pkl)
├── notebooks/
│   └── CSE425_Project_Final.ipynb (full notebook)
├── src/
│   ├── data_loader.py (dataset loading)
│   ├── feature_extractor.py (MFCC + text features)
│   ├── vae_models.py (BasicVAE, ConvVAE, ConditionalVAE classes)
│   ├── clustering.py (all clustering methods)
│   ├── evaluation.py (metrics calculation)
├── models/ (model params: basic_vae.pth, conv_vae.pth, cond_vae.pth, autoencoder.pth)
├── results/
│   ├── metrics/ (CSV files)
│   └── plots/ (PNG images)
└── deployment/ (final consolidation)
