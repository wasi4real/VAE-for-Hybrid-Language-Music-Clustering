"""Feature Normalization"""

# ===== 7. FEATURE PREPARATION =====
# Convert features to arrays
X_audio = np.stack(df_final['audio_features'].values)
X_text = np.stack(df_final['text_embeddings'].values)

# Normalize features
audio_scaler = StandardScaler()
X_audio_scaled = audio_scaler.fit_transform(X_audio)

text_scaler = StandardScaler()
X_text_scaled = text_scaler.fit_transform(X_text)

# Create hybrid features (audio + text embeddings)
X_hybrid = np.hstack([X_audio_scaled, X_text_scaled])

# Save scalers
joblib.dump(audio_scaler, os.path.join(BASE_DIR, "data/processed/audio_scaler.pkl"))
joblib.dump(text_scaler, os.path.join(BASE_DIR, "data/processed/text_scaler.pkl"))

# Prepare tensors
X_audio_tensor = torch.FloatTensor(X_audio_scaled)
X_hybrid_tensor = torch.FloatTensor(X_hybrid)

print(f"Audio features: {X_audio_scaled.shape}")
print(f"Text embeddings (from audio descriptions): {X_text_scaled.shape}")
print(f"Hybrid features: {X_hybrid.shape}")
print("\nFeature sources:")
print("  - Audio: MFCC coefficients from audio files")
print("  - Text: Embeddings of audio-derived descriptions (tempo, energy, brightness, rhythm)")
print("  - NO GENRE INFORMATION in features (prevents data leakage)")

"""Label Encoding"""

# ===== 8. LABEL ENCODING (FOR EVALUATION ONLY) =====
# Encode genres for evaluation metrics (NOT used in feature generation)
genre_encoder = LabelEncoder()
df_final['genre_encoded'] = genre_encoder.fit_transform(df_final['genre'])

# One-hot encoding for conditional VAE (still uses genre, but only as condition)
genre_onehot = pd.get_dummies(df_final['genre_encoded']).values
Y_genre_tensor = torch.FloatTensor(genre_onehot)

joblib.dump(genre_encoder, os.path.join(BASE_DIR, "data/processed/label_encoder.pkl"))

print(f"Encoded {len(genre_encoder.classes_)} unique genres (for evaluation only)")
print("Genres are ONLY used for:")
print("  1. Conditional VAE conditioning (Hard Task)")
print("  2. Evaluation metrics calculation")
print("  NOT used in feature generation (prevents data leakage)")