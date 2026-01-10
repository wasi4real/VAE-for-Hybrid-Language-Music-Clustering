# 1. HYPERPARAMETER TUNING FUNCTION
def tune_vae_hyperparameters(X_train, latent_dims=[8, 16, 32], betas=[1.0, 2.0, 4.0], lrs=[1e-3, 5e-4]):
    """Grid search for optimal VAE hyperparameters"""
    best_score = -np.inf
    best_params = {}
    
    for latent_dim in latent_dims:
        for beta in betas:
            for lr in lrs:
                # Train VAE with current params
                model = BasicVAE(input_dim=X_train.shape[1], latent_dim=latent_dim).cuda()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                # Quick training (fewer epochs for tuning)
                for epoch in range(10):
                    for i in range(0, len(X_train), 64):
                        batch = X_train[i:i+64].cuda()
                        optimizer.zero_grad()
                        recon, mu, logvar = model(batch)
                        loss = vae_loss(recon, batch, mu, logvar, beta=beta)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate on validation loss
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for i in range(0, len(X_train), 64):
                        batch = X_train[i:i+64].cuda()
                        recon, mu, logvar = model(batch)
                        val_loss += vae_loss(recon, batch, mu, logvar, beta=beta).item()
                    
                    if -val_loss > best_score:  # Negative loss for maximization
                        best_score = -val_loss
                        best_params = {'latent_dim': latent_dim, 'beta': beta, 'lr': lr}
    
    return best_params

# 2. DISENTANGLEMENT METRIC FUNCTION
def compute_disentanglement_metric(model, X_sample, y_sample, n_samples=100):
    """Compute MIG (Mutual Information Gap) for disentanglement"""
    model.eval()
    with torch.no_grad():
        # Encode samples
        mu, _ = model.encode(X_sample.cuda(), y_sample.cuda())
        mu = mu.cpu().numpy()
        
        # Compute mutual information between latents and factors
        n_latents = mu.shape[1]
        n_factors = y_sample.shape[1]
        
        MI = np.zeros((n_latents, n_factors))
        for i in range(n_latents):
            for j in range(n_factors):
                # Simplified MI estimation (for real use, implement proper MI estimator)
                latent_var = np.var(mu[:, i])
                factor_var = np.var(y_sample[:, j].numpy())
                MI[i, j] = 1.0 / (1.0 + np.abs(latent_var - factor_var))  # Proxy metric
        
        # Compute MIG: gap between top two MI values for each factor
        mig_score = 0
        for j in range(n_factors):
            sorted_mi = np.sort(MI[:, j])[::-1]
            if len(sorted_mi) > 1:
                mig_score += (sorted_mi[0] - sorted_mi[1])
        
        return mig_score / n_factors

# 3. MULTI-MODAL FUSION FUNCTION
def adaptive_fusion(audio_features, text_features, temperature=0.1):
    """Learn adaptive weights for audio-text fusion"""
    # Compute similarity between modalities
    audio_norm = F.normalize(audio_features, p=2, dim=1)
    text_norm = F.normalize(text_features, p=2, dim=1)
    similarity = torch.matmul(audio_norm, text_norm.T)
    
    # Compute attention weights
    audio_weights = F.softmax(similarity.mean(dim=1) / temperature, dim=0)
    text_weights = F.softmax(similarity.mean(dim=0) / temperature, dim=0)
    
    # Apply weighted fusion
    fused_features = (audio_weights.unsqueeze(1) * audio_features + 
                     text_weights.unsqueeze(1) * text_features)
    
    return fused_features, audio_weights, text_weights


# 1. Tune hyperparameters
best_params = tune_vae_hyperparameters(X_audio_tensor[:1000])  # Use subset
print(f"Best params: {best_params}")

# 2. Measure disentanglement
sample_idx = torch.randperm(len(X_audio_tensor))[:100]
mig_score = compute_disentanglement_metric(cond_vae, 
                                          X_audio_tensor[sample_idx], 
                                          Y_genre_tensor[sample_idx])
print(f"MIG Score: {mig_score:.4f}")

# 3. Adaptive fusion
fused, audio_w, text_w = adaptive_fusion(X_audio_tensor[:100], 
                                         X_text_tensor[:100])
print(f"Fusion weights - Audio: {audio_w.mean():.3f}, Text: {text_w.mean():.3f}")