"""# Model Saving & Deployment

Save All Models
"""

# ===== SAVE ALL RESULTS =====
torch.save(basic_vae.state_dict(), os.path.join(BASE_DIR, "models/basic_vae.pth"))
torch.save(conv_vae.state_dict(), os.path.join(BASE_DIR, "models/conv_vae.pth"))
torch.save(cond_vae.state_dict(), os.path.join(BASE_DIR, "models/cond_vae.pth"))
torch.save(autoencoder.state_dict(), os.path.join(BASE_DIR, "models/autoencoder.pth"))

# Save clustering labels
import pickle
all_labels = clustering_results.copy()
with open(os.path.join(BASE_DIR, "data/processed/all_clustering_labels.pkl"), 'wb') as f:
    pickle.dump(all_labels, f)

# Copy to deployment
deployment_files = [
    ("models/basic_vae.pth", "basic_vae.pth"),
    ("models/cond_vae.pth", "cond_vae.pth"),
    ("data/processed/audio_scaler.pkl", "audio_scaler.pkl"),
    ("data/processed/label_encoder.pkl", "label_encoder.pkl"),
    ("results/metrics/comprehensive_results.csv", "results_summary.csv"),
    ("results/plots/umap_visualizations.png", "umap_plot.png"),
    ("results/plots/reconstruction_examples.png", "reconstruction_examples.png"),
    ("results/plots/genre_distribution.png", "genre_distribution.png")
]

for src, dst in deployment_files:
    src_path = os.path.join(BASE_DIR, src)
    dst_path = os.path.join(BASE_DIR, "deployment", dst)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)

print("All files saved successfully")

# ===== SAVE AUDIO DESCRIPTIONS =====
# Save a sample of audio descriptions for the report
description_samples = df_final[['id', 'genre', 'language', 'audio_description']].sample(n=10, random_state=42)
description_samples.to_csv(os.path.join(BASE_DIR, "results/metrics/audio_descriptions_samples.csv"), index=False)

print("Sample audio descriptions saved:")
print(description_samples[['genre', 'audio_description']].head())
print("\nNote: Audio descriptions are derived from audio features only,")
print("not from genre information. This prevents data leakage.")