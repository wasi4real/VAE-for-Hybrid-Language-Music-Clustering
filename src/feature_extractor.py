# ===== 5. FEATURE EXTRACTION WITH AUDIO-DERIVED TEXT =====
def extract_audio_features_with_text(file_path, n_mfcc=13, duration=3.0):
    """
    Extract MFCC features AND generate audio-derived text description
    Avoids data leakage by not using genre information
    """
    try:
        y, sr = librosa.load(file_path, duration=duration, sr=22050)

        # 1. Extract MFCC features (for audio modality)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_features = np.mean(mfccs.T, axis=0)

        # 2. Extract features for text description (avoiding data leakage)
        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo[0] if len(tempo) > 0 else 120

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        energy = np.mean(rms)

        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        brightness = np.mean(spectral_centroid)

        # Zero-crossing rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        noisiness = np.mean(zcr)

        # Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_std = np.std(chroma)  # Measure of harmonic complexity

        # Beat salience (rhythm strength)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strength = np.mean(onset_env)

        # Convert numerical features to text categories
        # Tempo categories (avoid specific BPM numbers that could correlate with genre)
        if tempo < 90:
            tempo_desc = "slow tempo"
        elif tempo < 130:
            tempo_desc = "medium tempo"
        else:
            tempo_desc = "fast tempo"

        # Energy categories
        energy_q33 = np.percentile(rms, 33)
        energy_q66 = np.percentile(rms, 66)
        if energy < energy_q33:
            energy_desc = "low energy"
        elif energy < energy_q66:
            energy_desc = "medium energy"
        else:
            energy_desc = "high energy"

        # Brightness categories
        brightness_q33 = np.percentile(spectral_centroid, 33)
        brightness_q66 = np.percentile(spectral_centroid, 66)
        if brightness < brightness_q33:
            brightness_desc = "dark sound"
        elif brightness < brightness_q66:
            brightness_desc = "balanced sound"
        else:
            brightness_desc = "bright sound"

        # Rhythm complexity
        if chroma_std < np.percentile(chroma.std(axis=0), 33):
            rhythm_desc = "simple rhythm"
        elif chroma_std < np.percentile(chroma.std(axis=0), 66):
            rhythm_desc = "moderate rhythm"
        else:
            rhythm_desc = "complex rhythm"

        # Create text description (NO GENRE INFORMATION)
        text_description = f"Audio with {tempo_desc}, {energy_desc}, {brightness_desc}, {rhythm_desc}"

        return mfcc_features, text_description

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

# Extract features for all tracks
audio_features = []
text_descriptions = []
valid_indices = []

for idx, row in df_hybrid.iterrows():
    mfcc_features, text_desc = extract_audio_features_with_text(row['file_path'])
    if mfcc_features is not None and text_desc is not None:
        audio_features.append(mfcc_features)
        text_descriptions.append(text_desc)
        valid_indices.append(idx)

df_final = df_hybrid.iloc[valid_indices].copy()
df_final['audio_features'] = audio_features
df_final['audio_description'] = text_descriptions

print(f"Successfully extracted features for {len(df_final)} tracks")
print("\nSample audio descriptions (no genre leakage):")
for i in range(min(3, len(df_final))):
    print(f"  {df_final.iloc[i]['audio_description']}")

"""Lyric/Text Feature Extraction"""

# Generate text embeddings from audio-derived descriptions only
lyric_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Use ONLY audio-derived descriptions (no genre, no title with genre info)
text_embeddings = lyric_model.encode(df_final['audio_description'].tolist(),
                                     show_progress_bar=False)

df_final['text_embeddings'] = list(text_embeddings)

print(f"Text embeddings generated from audio descriptions: {text_embeddings.shape}")
print("\nExamples of text embeddings source (no data leakage):")
for i in range(min(3, len(df_final))):
    track_info = df_final.iloc[i]
    print(f"  Track ID: {track_info['id'][:20]}...")
    print(f"  Description: {track_info['audio_description']}")
    print(f"  Genre (ground truth, NOT used in embeddings): {track_info['genre']}")
    print()
