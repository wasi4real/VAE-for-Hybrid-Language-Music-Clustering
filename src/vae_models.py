"""# VAE Implementations

Easy Task: Basic VAE
"""

class BasicVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(BasicVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld

# Train Basic VAE
input_dim = X_audio_scaled.shape[1]
basic_vae = BasicVAE(input_dim).cuda()
optimizer = torch.optim.Adam(basic_vae.parameters(), lr=1e-3)

basic_vae.train()
for epoch in range(50):
    for i in range(0, len(X_audio_tensor), 64):
        batch = X_audio_tensor[i:i+64].cuda()
        optimizer.zero_grad()
        recon, mu, logvar = basic_vae(batch)
        loss = vae_loss(recon, batch, mu, logvar, beta=1.0)
        loss.backward()
        optimizer.step()

# Extract latent features
basic_vae.eval()
with torch.no_grad():
    mu, _ = basic_vae.encode(X_audio_tensor.cuda())
    latent_basic = mu.cpu().numpy()

print(f"Basic VAE trained, latent features: {latent_basic.shape}")

"""Medium Task: Convolutional VAE"""

class ConvVAE(nn.Module):
    def __init__(self, seq_length, latent_dim=32):
        super(ConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate flattened size
        with torch.no_grad():
            test_input = torch.randn(1, 1, seq_length)
            flattened_size = self.encoder(test_input).shape[1]

        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0), # Changed output_padding to 0
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, padding=1),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 32, h.shape[1] // 32)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Prepare data for Conv1D
X_conv = X_audio_scaled.reshape(-1, 1, X_audio_scaled.shape[1])
X_conv_tensor = torch.FloatTensor(X_conv)

# Train Conv VAE
conv_vae = ConvVAE(seq_length=X_audio_scaled.shape[1]).cuda()
optimizer = torch.optim.Adam(conv_vae.parameters(), lr=1e-3)

conv_vae.train()
for epoch in range(50):
    for i in range(0, len(X_conv_tensor), 64):
        batch = X_conv_tensor[i:i+64].cuda()
        optimizer.zero_grad()
        recon, mu, logvar = conv_vae(batch)
        loss = vae_loss(recon, batch, mu, logvar, beta=1.0)
        loss.backward()
        optimizer.step()

# Extract latent features
conv_vae.eval()
with torch.no_grad():
    mu, _ = conv_vae.encode(X_conv_tensor.cuda())
    latent_conv = mu.cpu().numpy()

print(f"Conv VAE trained, latent features: {latent_conv.shape}")

"""Hard Task: Conditional Beta-VAE"""

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, label_dim, latent_dim=16):
        super(ConditionalVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + label_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x, labels):
        x_cond = torch.cat([x, labels], dim=1)
        h = self.encoder(x_cond)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        z_cond = torch.cat([z, labels], dim=1)
        return self.decoder(z_cond)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

# Train Conditional Beta-VAE
label_dim = Y_genre_tensor.shape[1]
cond_vae = ConditionalVAE(input_dim, label_dim).cuda()
optimizer = torch.optim.Adam(cond_vae.parameters(), lr=1e-3)

cond_vae.train()
for epoch in range(100):
    for i in range(0, len(X_audio_tensor), 64):
        x_batch = X_audio_tensor[i:i+64].cuda()
        y_batch = Y_genre_tensor[i:i+64].cuda()
        optimizer.zero_grad()
        recon, mu, logvar = cond_vae(x_batch, y_batch)
        loss = vae_loss(recon, x_batch, mu, logvar, beta=4.0)
        loss.backward()
        optimizer.step()

# Extract latent features
cond_vae.eval()
with torch.no_grad():
    mu, _ = cond_vae.encode(X_audio_tensor.cuda(), Y_genre_tensor.cuda())
    latent_cond = mu.cpu().numpy()

print(f"Conditional Beta-VAE trained, latent features: {latent_cond.shape}")


"""Autoencoder + K-Means for Comparison"""
# ===== AUTOENCODER FOR COMPARISON =====
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Train Autoencoder
autoencoder = SimpleAutoencoder(input_dim).cuda()
ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

autoencoder.train()
for _ in range(50):
    for i in range(0, len(X_audio_tensor), 64):
        batch = X_audio_tensor[i:i+64].cuda()
        ae_optimizer.zero_grad()
        recon = autoencoder(batch)
        loss = mse_loss(recon, batch)
        loss.backward()
        ae_optimizer.step()

# Extract latent features
autoencoder.eval()
with torch.no_grad():
    latent_ae = autoencoder.encoder(X_audio_tensor.cuda()).cpu().numpy()

# K-Means on Autoencoder features
kmeans_ae = KMeans(n_clusters=8, random_state=42)
labels_ae = kmeans_ae.fit_predict(latent_ae)
