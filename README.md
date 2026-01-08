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
VAE-for-Hybrid-Language-Music-Clustering/
├── data/
│ ├── raw/ # (Empty - datasets downloaded via kagglehub)
│ └── processed/ # Processed datasets and features
├── notebooks/
│ └── CSE425_Project_Final.ipynb # Complete implementation notebook
├── src/
│ ├── data_loader.py # Dataset loading and merging
│ ├── feature_extractor.py # MFCC + text feature extraction
│ ├── vae_models.py # VAE architecture definitions
│ ├── clustering.py # Clustering algorithms
│ └── evaluation.py # Metrics calculation
├── models/ # Trained model weights (.pth files)
├── results/
│ ├── metrics/ # Evaluation metrics in CSV
│ └── plots/ # All visualizations (UMAP, reconstructions, etc.)
├── requirements.txt # Python dependencies
└── README.md # This file

text

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/wasi4real/VAE-for-Hybrid-Language-Music-Clustering.git
cd VAE-for-Hybrid-Language-Music-Clustering
pip install -r requirements.txt
Run Complete Pipeline
python
# From notebook:
notebooks/CSE425_Project_Final.ipynb

# Or from Python modules:
python src/data_loader.py      # Load and merge datasets
python src/feature_extractor.py # Extract features
python src/vae_models.py       # Train VAEs
python src/clustering.py       # Apply clustering
python src/evaluation.py       # Calculate metrics
Requirements
See requirements.txt for complete list. Key packages:

torch, librosa, scikit-learn, umap-learn

sentence-transformers, pandas, matplotlib

📈 Visualizations
Generated Plots
umap_visualizations.png - UMAP projections of all feature spaces

reconstruction_examples.png - VAE reconstruction comparisons

genre_distribution.png - Cluster composition by genre

All plots available in results/plots/

📝 Dataset Information
Sources
GTZAN: 1,000 English tracks across 10 genres (100 per genre)

BanglaBeats: 16,170 Bangla tracks across 8 genres

Preprocessing
Balanced sampling: 1,000 tracks from each language

Audio: 3-second segments, 22050Hz, MFCC extraction

Text: Audio-derived descriptions (tempo, energy, brightness, rhythm)

Features: StandardScaler normalization

🎯 Project Tasks Completed
Easy Task
✓ Basic VAE implementation

✓ K-Means clustering on latent features

✓ UMAP visualization

✓ Comparison with PCA + K-Means baseline

Medium Task
✓ Convolutional VAE architecture

✓ Hybrid audio-text features

✓ Multiple clustering algorithms (K-Means, Agglomerative, DBSCAN)

✓ Davies-Bouldin Index evaluation

Hard Task
✓ Conditional Beta-VAE (β=4.0)

✓ Multi-modal clustering with audio + text

✓ All metrics: Silhouette, ARI, NMI, Cluster Purity

✓ Latent space visualizations and disentanglement analysis

🛠️ Technical Details
Data Leakage Prevention
Text features derived from audio analysis only

No genre information in feature generation

Genre labels used only for conditioning (CVAE) and evaluation

Workarounds Implemented
UMAP projection for high-dimensional clustering

Audio-derived text features for multi-modality

Balanced language sampling for fair evaluation

📚 References
Kingma & Welling (2013). Auto-Encoding Variational Bayes

Logan (2000). Mel Frequency Cepstral Coefficients for music modeling

McInnes et al. (2018). UMAP: Uniform Manifold Approximation and Projection

👤 Author
Moin Mostakim
Neural Networks Course Project (CSE425)

📄 License
This project is for academic purposes as part of CSE425 coursework.

text
