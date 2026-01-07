# ============================================================================
# CELL 1: Import Required Libraries and Load Data
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths
OUTPUT_PATH = r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\processed_data2"
RESULTS_PATH = r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\results"
os.makedirs(RESULTS_PATH, exist_ok=True)

# Load preprocessed data
print("Loading preprocessed data...")
features = np.load(os.path.join(OUTPUT_PATH, 'features_normalized.npy'))
labels = np.load(os.path.join(OUTPUT_PATH, 'labels.npy'), allow_pickle=True)
metadata = pd.read_csv(os.path.join(OUTPUT_PATH, 'metadata.csv'))

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
print(f"Data loaded: {features.shape[0]} samples, {features.shape[1]} features")

# ============================================================================
# CELL 2: Define the Model (VAE)
# ============================================================================

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], latent_dim=32):
        super(VAE, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        prev_dim = latent_dim
        for hidden_dim in hidden_dims_rev:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims_rev[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar, z
    
    def get_latent_features(self, x):
        mu, _ = self.encode(x)
        return mu

def vae_loss(reconstruction, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

# ============================================================================
# CELL 3: Run the Model
# ============================================================================

# Configuration
VAE_CONFIG = {
    'hidden_dims': [features.shape[1], 512, 256, 128],  # Suggested encoder layers
    'latent_dims_to_try': [32, 64, 128], # Suggested latent dimensions
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 100,
    'beta': 1.0,
    'patience': 15,
}

# Prepare DataLoader
X_tensor = torch.FloatTensor(features)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=VAE_CONFIG['batch_size'], shuffle=True)

best_overall_loss = float('inf')
best_vae_model_state = None
best_latent_dim = None

print("Starting training with configuration search...")

for ld in VAE_CONFIG['latent_dims_to_try']:
    print(f"\n--- Training with Latent Dim: {ld} ---")
    
    # Initialize VAE with current latent dim
    vae = VAE(
        input_dim=features.shape[1], 
        hidden_dims=VAE_CONFIG['hidden_dims'], 
        latent_dim=ld
    ).to(device)
    
    optimizer = optim.Adam(vae.parameters(), lr=VAE_CONFIG['learning_rate'])
    
    # Training variables for this config
    current_best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(VAE_CONFIG['epochs']):
        vae.train()
        epoch_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            reconstruction, mu, logvar, _ = vae(x)
            loss = vae_loss(reconstruction, x, mu, logvar, VAE_CONFIG['beta'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        
        # Early stopping check
        if avg_loss < current_best_loss:
            current_best_loss = avg_loss
            patience_counter = 0
            # Save if this is the best overall so far (or just best for this config)
            # We want to keep the best model state across ALL configs? 
            # Or picking the best config? Usually lower loss is better.
            
            # Temporarily save this model state as candidate for this config
            current_config_best_state = vae.state_dict()
        else:
            patience_counter += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{VAE_CONFIG['epochs']}] Loss: {avg_loss:.4f}")
            
        if patience_counter >= VAE_CONFIG['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    print(f"Finished training for Latent Dim {ld}. Best Loss: {current_best_loss:.4f}")
    
    # Compare with overall best
    if current_best_loss < best_overall_loss:
        best_overall_loss = current_best_loss
        best_vae_model_state = current_config_best_state
        best_latent_dim = ld

print(f"\nTraining Complete. Best Latent Dimension: {best_latent_dim} with Loss: {best_overall_loss:.4f}")

# Save and load best model
torch.save(best_vae_model_state, os.path.join(RESULTS_PATH, 'best_vae_model.pth'))

# Re-initialize best model structure to load weights
vae = VAE(
    input_dim=features.shape[1], 
    hidden_dims=VAE_CONFIG['hidden_dims'], 
    latent_dim=best_latent_dim
).to(device)
vae.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_vae_model.pth')))
vae.eval()

with torch.no_grad():
    latent_features = vae.get_latent_features(X_tensor.to(device)).cpu().numpy()
print("Latent features extracted using best model.")

# ============================================================================
# CELL 4: Baseline PCA + Kmeans vs VAE + Kmeans
# ============================================================================

print("\nComparing VAE + KMeans vs PCA + KMeans...")

# Determine optimal K using VAE features
print("Determining optimal number of clusters...")
k_range = range(2, 10)
best_k = 2
best_sil = -1

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbls = km.fit_predict(latent_features)
    score = silhouette_score(latent_features, lbls)
    if score > best_sil:
        best_sil = score
        best_k = k

print(f"Optimal K determined as: {best_k}")

# VAE + KMeans
kmeans_vae = KMeans(n_clusters=best_k, random_state=42, n_init=10)
vae_clusters = kmeans_vae.fit_predict(latent_features)
vae_sil = silhouette_score(latent_features, vae_clusters)
vae_ch = calinski_harabasz_score(latent_features, vae_clusters)

# Baseline PCA uses the best latent dim found
pca = PCA(n_components=best_latent_dim)
pca_features = pca.fit_transform(features)
kmeans_pca = KMeans(n_clusters=best_k, random_state=42, n_init=10)
pca_clusters = kmeans_pca.fit_predict(pca_features)
pca_sil = silhouette_score(pca_features, pca_clusters)
pca_ch = calinski_harabasz_score(pca_features, pca_clusters)

# Report
results_df = pd.DataFrame({
    'Metric': ['Silhouette Score', 'Calinski-Harabasz Index'],
    'VAE + KMeans': [vae_sil, vae_ch],
    'PCA + KMeans': [pca_sil, pca_ch]
})
print("\nComparison Results:")
print(results_df)

# ============================================================================
# CELL 5: Visual cluster with t-SNE
# ============================================================================

print("\nGenerating t-SNE visualizations...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(latent_features)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Features (Colored by Clusters)
scatter1 = axes[0].scatter(tsne_results[:, 0], tsne_results[:, 1], c=vae_clusters, cmap='viridis', alpha=0.6)
axes[0].set_title(f't-SNE of VAE Features (Clusters K={best_k})')
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# Plot 2: Multilingual (Colored by Language)
# map language to colors: 0=Bangla, 1=English usually
languages = metadata['language'].map({'bn': 0, 'en': 1}).values
scatter2 = axes[1].scatter(tsne_results[:, 0], tsne_results[:, 1], c=languages, cmap='coolwarm', alpha=0.6)
axes[1].set_title('t-SNE of VAE Features (Multilingual)')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
cbar = plt.colorbar(scatter2, ax=axes[1], ticks=[0, 1])
cbar.ax.set_yticklabels(['Bangla', 'English'])

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'tsne_visualization_simplified.png'))
plt.show()
print(f"Visualization saved to {os.path.join(RESULTS_PATH, 'tsne_visualization_simplified.png')}")
