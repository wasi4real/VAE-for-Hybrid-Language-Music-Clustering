"""All Metrics"""

clustering_results = {
    'Raw K-Means': labels_raw,
    'PCA + K-Means': labels_pca,
    'Spectral Clustering': labels_spectral_full,
    'Autoencoder + K-Means': labels_ae,
    'Hybrid K-Means': labels_hybrid,
    'Basic VAE': labels_basic,
    'Conv VAE': labels_conv,
    'Cond VAE': labels_cond,
    'Agglomerative': labels_agg,
    'DBSCAN (Original)': labels_dbscan_original,
    'DBSCAN (UMAP)': labels_dbscan_umap
}

# Calculate metrics
metrics_data = []
external_data = []
true_labels = df_final['genre_encoded'].values

for name, labels in clustering_results.items():
    if len(np.unique(labels)) > 1:
        sil = silhouette_score(X_hybrid, labels)
        ch = calinski_harabasz_score(X_hybrid, labels)
        db = davies_bouldin_score(X_hybrid, labels)
        metrics_data.append([name, sil, ch, db])

        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)

        contingency = pd.crosstab(true_labels, labels)
        purity = np.sum(np.max(contingency.values, axis=0)) / len(true_labels)

        external_data.append([name, ari, nmi, purity])

# Create and save results
metrics_df = pd.DataFrame(metrics_data, columns=['Method', 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin'])
external_df = pd.DataFrame(external_data, columns=['Method', 'ARI', 'NMI', 'Purity'])

comprehensive_df = pd.merge(metrics_df, external_df, on='Method', how='outer')
comprehensive_df['Task'] = comprehensive_df['Method'].apply(
    lambda x: 'Baseline' if 'Raw' in x or 'PCA' in x or 'Spectral' in x or 'Autoencoder' in x else
              'Easy' if 'Basic' in x else
              'Medium' if 'Conv' in x or 'Hybrid' in x or 'DBSCAN' in x or 'Agglomerative' in x else
              'Hard'
)

comprehensive_df = comprehensive_df.sort_values('Silhouette', ascending=False)
comprehensive_df.to_csv(os.path.join(BASE_DIR, "results/metrics/comprehensive_results.csv"), index=False)
