
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_path': r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\processed_data2",
    'results_path': r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\results\Conditional_VAE",
    'batch_size': 32,
    'epochs': 600,  # Adjustable
    'learning_rate': 1e-4,
    'latent_dim': 64,  # Dimension of z
    'beta': 4.0,       # For Disentangled VAE (Beta-VAE)
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Ensure results directory exists
os.makedirs(CONFIG['results_path'], exist_ok=True)

print(f"Using device: {CONFIG['device']}")

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================

def load_data():
    print("-" * 60)
    print("Loading Data...")
    
    # Paths
    mel_path = os.path.join(CONFIG['data_path'], 'mel_spectrograms_normalized.npy')
    lyrics_path = os.path.join(CONFIG['data_path'], 'lyrics_embeddings.npy')
    handcrafted_path = os.path.join(CONFIG['data_path'], 'features_normalized.npy')
    metadata_path = os.path.join(CONFIG['data_path'], 'metadata.csv')
    
    # Load Files
    if not os.path.exists(mel_path):
        raise FileNotFoundError(f"File not found: {mel_path}")
        
    X_audio = np.load(mel_path)
    X_text = np.load(lyrics_path)
    X_handcrafted = np.load(handcrafted_path)
    metadata = pd.read_csv(metadata_path)
    
    # Add Channel Dimension to Audio: (N, 1, H, W)
    if len(X_audio.shape) == 3:
        X_audio = X_audio[:, np.newaxis, :, :]
    
    print(f"Audio Shape: {X_audio.shape}")
    print(f"Text Shape: {X_text.shape}")
    print(f"Handcrafted Features Shape: {X_handcrafted.shape}")
    print(f"Metadata Shape: {metadata.shape}")
    
    # Process Labels/Conditions
    # We will use 'genre' as the condition for CVAE
    # We also need 'language' for analysis
    
    # Label Encoding for Metrics (Ground Truth)
    le_genre = LabelEncoder()
    y_genre = le_genre.fit_transform(metadata['genre'])
    
    le_lang = LabelEncoder()
    y_lang = le_lang.fit_transform(metadata['language'])
    
    # One-Hot Encoding for CVAE Condition
    ohe = OneHotEncoder(sparse_output=False)
    c_genre = ohe.fit_transform(y_genre.reshape(-1, 1))
    
    dataset = {
        'audio': X_audio,
        'text': X_text,
        'handcrafted': X_handcrafted,
        'y_genre': y_genre,
        'y_lang': y_lang,
        'c_genre': c_genre,
        'genre_names': le_genre.classes_,
        'lang_names': le_lang.classes_
    }
    
    return dataset

# ============================================================================
# 2. CONDITIONAL VAE MODEL (CVAE)
# ============================================================================

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=64, text_dim=768, num_classes=10):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # --- Audio Encoder (CNN) ---
        # Input: (1, 128, 1024)
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # -> (32, 64, 512)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 32, 256)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, 16, 128)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# -> (256, 8, 64)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),# -> (512, 4, 32)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),# -> (512, 2, 16)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Flatten() # 512 * 2 * 16 = 16384
        )
        
        # --- Text Encoder (MLP) ---
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )
        
        # --- Latent Projection ---
        # Audio (16384) + Text (256) + Condition (num_classes)
        fusion_dim = 16384 + 256 + num_classes
        
        self.fc_mu = nn.Linear(fusion_dim, latent_dim)
        self.fc_logvar = nn.Linear(fusion_dim, latent_dim)
        
        # --- Decoder ---
        # Input: z (latent_dim) + Condition (num_classes)
        decoder_input_dim = latent_dim + num_classes
        
        self.decoder_fc = nn.Linear(decoder_input_dim, 16384 + 256) # Project back to split sizes
        
        # --- Text Decoder ---
        self.text_decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, text_dim)
        )
        
        # --- Audio Decoder (Transposed CNN) ---
        self.audio_unflatten = nn.Unflatten(1, (512, 2, 16))
        
        self.audio_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (512, 4, 32)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (256, 8, 64)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (128, 16, 128)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (64, 32, 256)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (32, 64, 512)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)     # -> (1, 128, 1024)
        )

    def encode(self, audio, text, condition):
        # 1. Encode modalities
        a_emb = self.audio_encoder(audio)
        t_emb = self.text_encoder(text)
        
        # 2. Concatenate with condition
        combined = torch.cat([a_emb, t_emb, condition], dim=1)
        
        # 3. Predict statistics
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        # 1. Concatenate z with condition
        combined = torch.cat([z, condition], dim=1)
        
        # 2. Project
        splits = self.decoder_fc(combined)
        
        # 3. Split
        # Audio part (16384), Text part (256)
        a_hidden = splits[:, :16384]
        t_hidden = splits[:, 16384:]
        
        # 4. Decode Audio
        a_reshaped = self.audio_unflatten(a_hidden)
        recon_audio = self.audio_decoder(a_reshaped)
        
        # 5. Decode Text
        recon_text = self.text_decoder(t_hidden)
        
        return recon_audio, recon_text

    def forward(self, audio, text, condition):
        mu, logvar = self.encode(audio, text, condition)
        z = self.reparameterize(mu, logvar)
        recon_audio, recon_text = self.decode(z, condition)
        return recon_audio, recon_text, mu, logvar

def cvae_loss_function(recon_audio, x_audio, recon_text, x_text, mu, logvar, beta=1.0):
    # Reconstruction Losses (Sum of Squared Error)
    mse_audio = nn.functional.mse_loss(recon_audio, x_audio, reduction='sum')
    mse_text = nn.functional.mse_loss(recon_text, x_text, reduction='sum')
    
    # Weight text loss higher to balance dimensions (Audio ~130k dims, Text 768 dims)
    # Ratio is approx 170. Let's use 200.
    reconstruction_loss = mse_audio + (mse_text * 200)
    
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Beta-VAE Objective
    return reconstruction_loss + beta * kld, mse_audio, mse_text, kld

# ============================================================================
# 3. BASELINE MODELS
# ============================================================================

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# ============================================================================
# 4. TRAINING & EVALUATION UTILS
# ============================================================================

def calculate_purity(y_true, y_pred):
    """
    Calculate cluster purity.
    Purity = (1/N) * sum(max(intersection(cluster_i, class_j)))
    """
    cm = confusion_matrix(y_true, y_pred)
    # For each cluster (column), find max overlap with a class (row)
    max_counts = np.amax(cm, axis=0) 
    return np.sum(max_counts) / np.sum(cm)

def evaluate_clustering(latent_vectors, y_true, name="Model"):
    print(f"\nEvaluating Clustering: {name}")
    
    # 1. K-Means
    n_clusters = len(np.unique(y_true))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(latent_vectors)
    
    # 2. Metrics
    sil = silhouette_score(latent_vectors, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    purity = calculate_purity(y_true, y_pred)
    
    print(f"  Silhouette Score: {sil:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}")
    print(f"  Purity: {purity:.4f}")
    
    return {'Silhouette': sil, 'NMI': nmi, 'ARI': ari, 'Purity': purity}

def train_cvae(model, train_loader, val_loader, epochs, beta, patience=20):
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    best_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting CVAE Training...")
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0
        for batch in train_loader:
            b_audio, b_text, b_cond = [t.to(CONFIG['device']) for t in batch]
            
            optimizer.zero_grad()
            
            recon_audio, recon_text, mu, logvar = model(b_audio, b_text, b_cond)
            loss, _, _, _ = cvae_loss_function(recon_audio, b_audio, recon_text, b_text, mu, logvar, beta=beta)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                b_audio, b_text, b_cond = [t.to(CONFIG['device']) for t in batch]
                recon_audio, recon_text, mu, logvar = model(b_audio, b_text, b_cond)
                loss, _, _, _ = cvae_loss_function(recon_audio, b_audio, recon_text, b_text, mu, logvar, beta=beta)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.2f} | Val Loss: {avg_val_loss:.2f}")
            
        # --- Early Stopping ---
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # Optional: Save best model
            # torch.save(model.state_dict(), os.path.join(CONFIG['results_path'], 'best_cvae.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    return model

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    # Load Data
    data = load_data()
    
    # Prepare Tensors
    X_audio_t = torch.FloatTensor(data['audio'])
    X_text_t = torch.FloatTensor(data['text'])
    condition_t = torch.FloatTensor(data['c_genre'])
    X_handcrafted_t = torch.FloatTensor(data['handcrafted'])
    
    dataset = TensorDataset(X_audio_t, X_text_t, condition_t)
    
    # Split into Train/Validation
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 6. TRAIN CVAE
    # ------------------
    cvae = ConditionalVAE(latent_dim=CONFIG['latent_dim'], 
                          text_dim=data['text'].shape[1], 
                          num_classes=condition_t.shape[1]).to(CONFIG['device'])
                          
    cvae = train_cvae(cvae, train_loader, val_loader, CONFIG['epochs'], CONFIG['beta'], patience=20)
    
    # Extract Latent Vectors
    cvae.eval()
    with torch.no_grad():
        mu, _ = cvae.encode(X_audio_t.to(CONFIG['device']), 
                            X_text_t.to(CONFIG['device']), 
                            condition_t.to(CONFIG['device']))
        z_cvae = mu.cpu().numpy()
        
        # Get reconstructions for 1 batch
        sample_audio, sample_text, sample_cond = next(iter(train_loader))
        rec_audio, _, _, _ = cvae(sample_audio.to(CONFIG['device']), 
                                  sample_text.to(CONFIG['device']), 
                                  sample_cond.to(CONFIG['device']))
        
    # 7. COMPARISONS & EVALUATION
    # ------------------
    results = []
    
    # A. CVAE + K-Means
    metrics_cvae = evaluate_clustering(z_cvae, data['y_genre'], "CVAE")
    metrics_cvae['Method'] = 'CVAE (Multi-Modal)'
    results.append(metrics_cvae)
    
    # B. PCA + K-Means (on Handcrafted features)
    # Using handcrafted because PCA on raw Mel is huge and usually requires flattening
    print("\nRunning PCA + K-Means...")
    pca = PCA(n_components=CONFIG['latent_dim'])
    z_pca = pca.fit_transform(data['handcrafted'])
    metrics_pca = evaluate_clustering(z_pca, data['y_genre'], "PCA (Handcrafted)")
    metrics_pca['Method'] = 'PCA + K-Means'
    results.append(metrics_pca)
    
    # C. Autoencoder + K-Means (on Handcrafted)
    print("\nRunning Autoencoder + K-Means...")
    ae = SimpleAutoencoder(input_dim=data['handcrafted'].shape[1], 
                           latent_dim=CONFIG['latent_dim']).to(CONFIG['device'])
    ae_optimizer = optim.Adam(ae.parameters(), lr=1e-3)
    
    # Simple Train Loop
    ae_loader = DataLoader(TensorDataset(X_handcrafted_t), batch_size=32, shuffle=True)
    for epoch in range(50):
        for (batch_x,) in ae_loader:
            batch_x = batch_x.to(CONFIG['device'])
            ae_optimizer.zero_grad()
            recon, z = ae(batch_x)
            loss = nn.functional.mse_loss(recon, batch_x)
            loss.backward()
            ae_optimizer.step()
            
    ae.eval()
    with torch.no_grad():
        _, z_ae = ae(X_handcrafted_t.to(CONFIG['device']))
        z_ae = z_ae.cpu().numpy()
        
    metrics_ae = evaluate_clustering(z_ae, data['y_genre'], "Autoencoder (Handcrafted)")
    metrics_ae['Method'] = 'Autoencoder + K-Means'
    results.append(metrics_ae)
    
    # D. Direct Spectral Clustering
    print("\nRunning Direct Spectral Clustering...")
    # Just run K-Means directly on normalized handcrafted features
    metrics_direct = evaluate_clustering(data['handcrafted'], data['y_genre'], "Direct Spectral")
    metrics_direct['Method'] = 'Direct Spectral'
    results.append(metrics_direct)
    
    # Save Results
    df_res = pd.DataFrame(results)
    df_res['Architecture'] = 'Conditional VAE'
    
    # Common Results Path
    COMMON_RESULTS_PATH = r"f:\BRACU\Semester 12 Final\CSE425\FInal_project\results"
    common_csv_path = os.path.join(COMMON_RESULTS_PATH, 'clustering_metrics.csv')

    # Append to common metrics file
    if os.path.exists(common_csv_path):
        try:
            df_common = pd.read_csv(common_csv_path)
            # Remove previous results for this architecture
            df_common = df_common[df_common['Architecture'] != 'Conditional VAE']
            # Append new results
            df_common = pd.concat([df_common, df_res], ignore_index=True)
        except Exception as e:
            print(f"Error reading existing CSV: {e}. Creating new one.")
            df_common = df_res
    else:
        df_common = df_res

    df_common.to_csv(common_csv_path, index=False)
    print(f"Metrics updated in {common_csv_path}")

    # Also save strictly to the cvae folder
    df_res.to_csv(os.path.join(CONFIG['results_path'], 'clustering_metrics.csv'), index=False)
    print("\nFinal Comparison:")
    print(df_res)
    
    # 8. VISUALIZATIONS
    # ------------------
    print("\nGenerating Visualizations...")
    
    # 1. Reconstruction Plot
    plt.figure(figsize=(12, 4))
    # Original
    plt.subplot(1, 2, 1)
    orig_img = sample_audio[0, 0].cpu().numpy()
    plt.imshow(orig_img, aspect='auto', origin='lower', cmap='viridis')
    plt.title("Original Mel Spectrogram")
    plt.colorbar()
    # Reconstructed
    plt.subplot(1, 2, 2)
    rec_img = rec_audio[0, 0].cpu().detach().numpy()
    plt.imshow(rec_img, aspect='auto', origin='lower', cmap='viridis')
    plt.title("CVAE Reconstruction")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['results_path'], 'reconstruction.png'))
    plt.close()
    
    # 2. Latent Space (t-SNE) for CVAE
    # We use t-SNE to project 64D z -> 2D
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    z_embedded = tsne.fit_transform(z_cvae)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=data['y_genre'], cmap='tab10', alpha=0.6, s=15)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(data['genre_names']), title="Genre")
    plt.title("CVAE Latent Space (t-SNE) by Genre")
    plt.savefig(os.path.join(CONFIG['results_path'], 'cvae_latent_tsne_genre.png'))
    plt.close()
    
    # 3. Cluster Distribution (Language)
    # Check language distribution in CVAE clusters
    kmeans_cvae = KMeans(n_clusters=len(data['genre_names']), random_state=42).fit(z_cvae)
    labels_pred = kmeans_cvae.labels_
    
    df_dist = pd.DataFrame({'Cluster': labels_pred, 'Language': data['dataset_lang_names'] if 'dataset_lang_names' in data else data['y_lang']})
    # Since y_lang is encoded, let's map back if possible or just use code
    # Actually let's use the raw dataframe for easier plotting if indices match
    
    # Quick Plot of distribution
    plt.figure(figsize=(10, 6))
    
    # Create a crosstab
    ct = pd.crosstab(labels_pred, data['y_lang'])
    # Rename columns using lang_names
    ct.columns = [data['lang_names'][i] for i in ct.columns]
    
    ct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title("Cluster Distribution by Language")
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['results_path'], 'cluster_lang_distribution.png'))
    plt.close()

    print(f"All results saved to {CONFIG['results_path']}")

if __name__ == "__main__":
    main()
