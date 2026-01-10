import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder
import warnings
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


OUTPUT_PATH = r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\processed_data1"
RESULTS_PATH = r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\results\Simple_VAE"
os.makedirs(RESULTS_PATH, exist_ok=True)

print("Loading preprocessed data...")
features = np.load(os.path.join(OUTPUT_PATH, 'features_normalized.npy'))
labels = np.load(os.path.join(OUTPUT_PATH, 'labels.npy'), allow_pickle=True)
metadata = pd.read_csv(os.path.join(OUTPUT_PATH, 'metadata.csv'))

print(f"\nData loaded successfully!")
print(f"Features shape: {features.shape}")
print(f"Number of samples: {len(labels)}")
print(f"Number of unique labels: {len(np.unique(labels))}")

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# ============================================================================
# CELL 2: Model definition
# ============================================================================

class VAE(nn.Module):
    """Variational Autoencoder for music feature extraction."""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], latent_dim=64):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
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
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
    
    def get_latent_features(self, x):
        mu, _ = self.encode(x)
        return mu


def vae_loss(reconstruction, x, mu, logvar, beta=1.0):

    recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

print("VAE loss function defined!")

VAE_CONFIG = {
    'hidden_dims': [128, 64, 32],   
    'latent_dim': 32,                 
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 500,
    'beta': 0.8,                      
    'patience': 15,                   
}

print("VAE Configuration:")
for key, value in VAE_CONFIG.items():
    print(f"  {key}: {value}")
X_tensor = torch.FloatTensor(features)

dataset = TensorDataset(X_tensor)
dataloader = DataLoader(
    dataset, 
    batch_size=VAE_CONFIG['batch_size'], 
    shuffle=True
)

print(f"\nDataLoader created with {len(dataloader)} batches")

input_dim = features.shape[1]
vae = VAE(
    input_dim=input_dim,
    hidden_dims=VAE_CONFIG['hidden_dims'],
    latent_dim=VAE_CONFIG['latent_dim']
).to(device)

optimizer = optim.Adam(vae.parameters(), lr=VAE_CONFIG['learning_rate'])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15
)

print(f"\nVAE Model Summary:")
print(f"  Input dimension: {input_dim}")
print(f"  Hidden dimensions: {VAE_CONFIG['hidden_dims']}")
print(f"  Latent dimension: {VAE_CONFIG['latent_dim']}")
print(f"  Total parameters: {sum(p.numel() for p in vae.parameters())}")

print("\n" + "="*60)
print("TRAINING VAE")
print("="*60)

train_losses = []
recon_losses = []
kl_losses = []
best_loss = float('inf')
patience_counter = 0

for epoch in range(VAE_CONFIG['epochs']):
    vae.train()
    epoch_loss = 0
    epoch_recon = 0
    epoch_kl = 0
    
    for batch in dataloader:
        x = batch[0].to(device)
        
        reconstruction, mu, logvar, z = vae(x)
        
        loss, recon_loss, kl_loss = vae_loss(
            reconstruction, x, mu, logvar, VAE_CONFIG['beta']
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_recon += recon_loss.item()
        epoch_kl += kl_loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    avg_recon = epoch_recon / len(dataloader)
    avg_kl = epoch_kl / len(dataloader)
    
    train_losses.append(avg_loss)
    recon_losses.append(avg_recon)
    kl_losses.append(avg_kl)
    
    scheduler.step(avg_loss)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(vae.state_dict(), os.path.join(RESULTS_PATH, 'best_vae_model.pth'))
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{VAE_CONFIG['epochs']}] - "
              f"Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
    
    if patience_counter >= VAE_CONFIG['patience']:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print("\nTraining complete!")

print("Loading best model for evaluation...")
vae.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_vae_model.pth')))
vae.eval()

with torch.no_grad():
    latent_features = vae.get_latent_features(X_tensor.to(device)).cpu().numpy()

print(f"Latent features extracted. Shape: {latent_features.shape}")

best_latent_dim = VAE_CONFIG['latent_dim']


# ============================================================================
# CELL 3: Baseline PCA + Kmeans vs VAE + Kmeans
# ============================================================================

print("\nComparing VAE + KMeans vs PCA + KMeans...")

print("Determining optimal number of clusters...")
k_range = range(3, 10, 2)
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

kmeans_vae = KMeans(n_clusters=best_k, random_state=42, n_init=10)
vae_clusters = kmeans_vae.fit_predict(latent_features)
vae_sil = silhouette_score(latent_features, vae_clusters)
vae_ch = calinski_harabasz_score(latent_features, vae_clusters)
pca = PCA(n_components=best_latent_dim)
pca_features = pca.fit_transform(features)
kmeans_pca = KMeans(n_clusters=best_k, random_state=42, n_init=10)
pca_clusters = kmeans_pca.fit_predict(pca_features)
pca_sil = silhouette_score(pca_features, pca_clusters)
pca_ch = calinski_harabasz_score(pca_features, pca_clusters)

# Formulate results in a standardized format for the common file
df_final = pd.DataFrame({
    'Method': ['VAE + KMeans', 'PCA + KMeans'],
    'Silhouette': [vae_sil, pca_sil],
    'Calinski-Harabasz': [vae_ch, pca_ch],
    'Architecture': 'Simple VAE'
})

print("\nComparison Results:")
print(df_final)

# Common Results Path
COMMON_RESULTS_PATH = r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\results"
common_csv_path = os.path.join(COMMON_RESULTS_PATH, 'clustering_metrics.csv')

# Append to common metrics file
if os.path.exists(common_csv_path):
    try:
        df_common = pd.read_csv(common_csv_path)
        # Remove previous results for this architecture to avoid duplicates
        df_common = df_common[df_common['Architecture'] != 'Simple VAE']
        # Append new results
        df_common = pd.concat([df_common, df_final], ignore_index=True)
    except Exception as e:
        print(f"Error reading existing CSV: {e}. Creating new one.")
        df_common = df_final
else:
    df_common = df_final

df_common.to_csv(common_csv_path, index=False)
print(f"\nMetrics updated in {common_csv_path}")

# ============================================================================
# CELL 5: Visual cluster with t-SNE
# ============================================================================

print("\nGenerating t-SNE visualizations...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(latent_features)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

scatter1 = axes[0].scatter(tsne_results[:, 0], tsne_results[:, 1], c=vae_clusters, cmap='viridis', alpha=0.6)
axes[0].set_title(f't-SNE of VAE Features (Clusters K={best_k})')
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

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