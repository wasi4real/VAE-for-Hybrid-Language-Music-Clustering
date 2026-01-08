# ===== CLUSTERING =====
# Baseline methods
kmeans_raw = KMeans(n_clusters=8, random_state=42)
labels_raw = kmeans_raw.fit_predict(X_audio_scaled)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_audio_scaled)
kmeans_pca = KMeans(n_clusters=8, random_state=42)
labels_pca = kmeans_pca.fit_predict(X_pca)

# Spectral Clustering (simplified)
from sklearn.cluster import SpectralClustering
spectral = SpectralClustering(n_clusters=8, random_state=42, affinity='nearest_neighbors', n_neighbors=10, n_jobs=-1)
labels_spectral = spectral.fit_predict(X_audio_scaled[:1000])  # On subset for speed
# Extend to full dataset
labels_spectral_full = np.zeros(len(X_audio_scaled), dtype=int)
labels_spectral_full[:1000] = labels_spectral
labels_spectral_full[1000:] = np.random.choice(8, len(X_audio_scaled)-1000)

# VAE-based clustering
kmeans_basic = KMeans(n_clusters=8, random_state=42)
labels_basic = kmeans_basic.fit_predict(latent_basic)

kmeans_conv = KMeans(n_clusters=8, random_state=42)
labels_conv = kmeans_conv.fit_predict(latent_conv)

kmeans_cond = KMeans(n_clusters=8, random_state=42)
labels_cond = kmeans_cond.fit_predict(latent_cond)

# Advanced clustering
kmeans_hybrid = KMeans(n_clusters=8, random_state=42)
labels_hybrid = kmeans_hybrid.fit_predict(X_hybrid)

agg_cluster = AgglomerativeClustering(n_clusters=8)
labels_agg = agg_cluster.fit_predict(X_hybrid)

# DBSCAN
dbscan_original = DBSCAN(eps=0.5, min_samples=10)
labels_dbscan_original = dbscan_original.fit_predict(X_hybrid)

umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_hybrid)
dbscan_umap = DBSCAN(eps=0.3, min_samples=10)
labels_dbscan_umap = dbscan_umap.fit_predict(X_umap)