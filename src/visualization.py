"""# Analysis & Visualization

Umap Visualizations
"""

# ===== 14. VISUALIZATIONS =====
# UMAP visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Raw features
umap_raw = umap.UMAP(random_state=42).fit_transform(X_audio_scaled)
sc1 = axes[0,0].scatter(umap_raw[:,0], umap_raw[:,1], c=labels_raw, cmap='tab20', alpha=0.6, s=10)
axes[0,0].set_title('Raw Features')
plt.colorbar(sc1, ax=axes[0,0])

# PCA features
umap_pca = umap.UMAP(random_state=42).fit_transform(X_pca)
sc2 = axes[0,1].scatter(umap_pca[:,0], umap_pca[:,1], c=labels_pca, cmap='tab20', alpha=0.6, s=10)
axes[0,1].set_title('PCA Features')
plt.colorbar(sc2, ax=axes[0,1])

# Basic VAE
umap_basic = umap.UMAP(random_state=42).fit_transform(latent_basic)
sc3 = axes[0,2].scatter(umap_basic[:,0], umap_basic[:,1], c=labels_basic, cmap='tab20', alpha=0.6, s=10)
axes[0,2].set_title('Basic VAE')
plt.colorbar(sc3, ax=axes[0,2])

# Conv VAE
umap_conv = umap.UMAP(random_state=42).fit_transform(latent_conv)
sc4 = axes[1,0].scatter(umap_conv[:,0], umap_conv[:,1], c=labels_conv, cmap='tab20', alpha=0.6, s=10)
axes[1,0].set_title('Conv VAE')
plt.colorbar(sc4, ax=axes[1,0])

# Cond VAE
umap_cond = umap.UMAP(random_state=42).fit_transform(latent_cond)
sc5 = axes[1,1].scatter(umap_cond[:,0], umap_cond[:,1], c=labels_cond, cmap='tab20', alpha=0.6, s=10)
axes[1,1].set_title('Cond VAE')
plt.colorbar(sc5, ax=axes[1,1])

# Hybrid features
sc6 = axes[1,2].scatter(X_umap[:,0], X_umap[:,1], c=labels_agg, cmap='tab20', alpha=0.6, s=10)
axes[1,2].set_title('Hybrid Features (Agglomerative)')
plt.colorbar(sc6, ax=axes[1,2])

# DBSCAN on original space
sc7 = axes[2,0].scatter(X_umap[:,0], X_umap[:,1], c=labels_dbscan_original, cmap='tab20', alpha=0.6, s=10)
axes[2,0].set_title('DBSCAN (Original Space)')
plt.colorbar(sc7, ax=axes[2,0])

# DBSCAN on UMAP space
sc8 = axes[2,1].scatter(X_umap[:,0], X_umap[:,1], c=labels_dbscan_umap, cmap='tab20', alpha=0.6, s=10)
axes[2,1].set_title('DBSCAN (UMAP Space)')
plt.colorbar(sc8, ax=axes[2,1])

# Language distribution
sc9 = axes[2,2].scatter(X_umap[:,0], X_umap[:,1],
                        c=[0 if lang == 'bangla' else 1 for lang in df_final['language']],
                        cmap='coolwarm', alpha=0.6, s=10)
axes[2,2].set_title('Language (Red=Bangla, Blue=English)')
plt.colorbar(sc9, ax=axes[2,2], ticks=[0, 1])

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "results/plots/umap_visualizations.png"), dpi=300, bbox_inches='tight')
plt.show()

"""Genre Distribution"""

# ===== GENRE DISTRIBUTION VISUALIZATION =====
# Simple genre distribution for Conditional VAE
plt.figure(figsize=(10, 6))
labels = labels_cond
n_clusters = len(np.unique(labels))

# Get top 3 genres per cluster
cluster_data = []
for cluster_id in range(n_clusters):
    cluster_mask = labels == cluster_id
    if np.sum(cluster_mask) > 10:
        genre_counts = df_final[cluster_mask]['genre'].value_counts().head(3)
        for genre, count in genre_counts.items():
            cluster_data.append({
                'Cluster': f'C{cluster_id}',
                'Genre': genre,
                'Count': count
            })

# Create DataFrame and pivot
if cluster_data:
    cluster_df = pd.DataFrame(cluster_data)
    pivot_df = cluster_df.pivot(index='Cluster', columns='Genre', values='Count').fillna(0)

    # Plot
    pivot_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20c')
    plt.title('Top Genres per Cluster (Conditional VAE)')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    genre_path = os.path.join(BASE_DIR, "results/plots/genre_distribution.png")
    plt.savefig(genre_path, dpi=300, bbox_inches='tight')
    plt.show()

"""# Reconstruction Example"""

print("Generating reconstruction examples...")

# Select 3 random samples
np.random.seed(42)
sample_indices = np.random.choice(len(df_final), 3, replace=False)

fig, axes = plt.subplots(3, 4, figsize=(16, 9))
fig.suptitle('VAE Reconstruction Examples', fontsize=16, y=1.02)

for idx, sample_idx in enumerate(sample_indices):
    # Get original features
    x_original = X_audio_tensor[sample_idx:sample_idx+1].cuda()
    y_label = Y_genre_tensor[sample_idx:sample_idx+1].cuda()

    # Get reconstructions
    with torch.no_grad():
        # Basic VAE
        recon_basic, _, _ = basic_vae(x_original)

        # Conv VAE
        x_conv = X_conv_tensor[sample_idx:sample_idx+1].cuda()
        recon_conv, _, _ = conv_vae(x_conv)
        recon_conv = recon_conv.squeeze(1)

        # Conditional VAE
        recon_cond, _, _ = cond_vae(x_original, y_label)

    # Get data
    orig_features = x_original.cpu().numpy().squeeze()
    basic_features = recon_basic.cpu().numpy().squeeze()
    conv_features = recon_conv.cpu().numpy().squeeze()
    cond_features = recon_cond.cpu().numpy().squeeze()

    # Calculate reconstruction errors
    mse_basic = np.mean((orig_features - basic_features) ** 2)
    mse_conv = np.mean((orig_features - conv_features) ** 2)
    mse_cond = np.mean((orig_features - cond_features) ** 2)

    # Plot original
    axes[idx, 0].plot(orig_features, color='blue', alpha=0.8, linewidth=2)
    axes[idx, 0].set_title(f'Sample {idx+1}: Original\n{df_final.iloc[sample_idx]["genre"]} ({df_final.iloc[sample_idx]["language"]})')
    axes[idx, 0].set_xlabel('MFCC Coefficient')
    axes[idx, 0].set_ylabel('Value')
    axes[idx, 0].grid(True, alpha=0.3)

    # Plot Basic VAE reconstruction
    axes[idx, 1].plot(basic_features, color='orange', alpha=0.8)
    axes[idx, 1].plot(orig_features, color='blue', alpha=0.3, linestyle='--')
    axes[idx, 1].set_title(f'Basic VAE\nMSE: {mse_basic:.4f}')
    axes[idx, 1].set_xlabel('MFCC Coefficient')
    axes[idx, 1].grid(True, alpha=0.3)

    # Plot Conv VAE reconstruction
    axes[idx, 2].plot(conv_features, color='green', alpha=0.8)
    axes[idx, 2].plot(orig_features, color='blue', alpha=0.3, linestyle='--')
    axes[idx, 2].set_title(f'Conv VAE\nMSE: {mse_conv:.4f}')
    axes[idx, 2].set_xlabel('MFCC Coefficient')
    axes[idx, 2].grid(True, alpha=0.3)

    # Plot Conditional VAE reconstruction
    axes[idx, 3].plot(cond_features, color='red', alpha=0.8)
    axes[idx, 3].plot(orig_features, color='blue', alpha=0.3, linestyle='--')
    axes[idx, 3].set_title(f'Cond VAE\nMSE: {mse_cond:.4f}')
    axes[idx, 3].set_xlabel('MFCC Coefficient')
    axes[idx, 3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "results/plots/reconstruction_examples.png"), dpi=300, bbox_inches='tight')
plt.show()

# Calculate overall reconstruction quality
with torch.no_grad():
    # Basic VAE
    recon_all_basic, _, _ = basic_vae(X_audio_tensor.cuda())
    mse_basic_total = F.mse_loss(recon_all_basic, X_audio_tensor.cuda()).item()

    # Conv VAE
    recon_all_conv, _, _ = conv_vae(X_conv_tensor.cuda())
    recon_all_conv = recon_all_conv.squeeze(1)
    mse_conv_total = F.mse_loss(recon_all_conv, X_audio_tensor.cuda()).item()

    # Conditional VAE
    recon_all_cond, _, _ = cond_vae(X_audio_tensor.cuda(), Y_genre_tensor.cuda())
    mse_cond_total = F.mse_loss(recon_all_cond, X_audio_tensor.cuda()).item()

print("\nOverall Reconstruction Quality (MSE):")
print(f"  Basic VAE: {mse_basic_total:.6f}")
print(f"  Conv VAE: {mse_conv_total:.6f}")
print(f"  Conditional VAE: {mse_cond_total:.6f}")