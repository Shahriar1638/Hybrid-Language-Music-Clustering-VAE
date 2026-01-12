# Hybrid Language Music Clustering VAE

This project implements and evaluates various Variational Autoencoder (VAE) architectures to cluster music data from a hybrid dataset consisting of Bangla and English songs. By leveraging multi-modal features (Audio + Lyrics), the project aims to explore latent representations that effectively distinguish different genres and languages.

## Project Structure

```
├── Datasets/                 # Raw audio datasets and metadata
├── nn_env/                   # Virtual environment (optional)
├── notebooks/                # Jupyter Notebooks for analysis
│   └── exploratory.ipynb     # Exploratory Data Analysis (EDA)
├── processed_data1/          # Output of basic preprocessing
├── processed_data2/          # Output of advanced preprocessing
├── results/                  # Generated metrics, plots, and models
│   ├── Conditional_VAE/      # Results for Conditional VAE
│   ├── Convolutional_VAE/    # Results for Convolutional VAE
│   ├── Simple_VAE/           # Results for Simple VAE
│   └── clustering_metrics.csv # Consolidated metrics for comparisons
└── src/                      # Source code for preprocessing and models
    ├── 1_preprocessing.py            # Basic feature extraction (MFCC, Spectral, etc.)
    ├── 1_preprocessing_advanced.py   # Advanced preprocessing (Hi-Res Spectrograms + Lyrics Embeddings)
    ├── Simple_VAE.py                 # Baseline MLP-based VAE implementation
    ├── Conditional_VAE.py            # Conditional VAE (CVAE) for supervised/conditioned generation
    └── Convolutional_VAE.py          # Hybrid VAE (CNN + MLP) handling Audio & Text
```

## Features

- **Multi-Modal Learning:** Combines Audio features (Mel Spectrograms, MFCCs) and Text features (Lyrics embeddings via Sentence-Transformers).
- **Advanced Preprocessing:**
    - High-Resolution Mel Spectrogram extraction.
    - Automated cleaning and feature extraction pipelines.
- **VAE Architectures:**
    - **Simple VAE:** A baseline fully connected VAE.
    - **Convolutional VAE (Hybrid):** Uses 2D CNNs for spectrograms and MLPs for text embeddings.
    - **Conditional VAE:** Conditions the latent space on Genre labels for better disentanglement.
- **Clustering Analysis:**
    - Algorithms: K-Means, Agglomerative Clustering, DBSCAN, Spectral Clustering.
    - Metrics: Silhouette Score, Davies-Bouldin Index, Adjusted Rand Index (ARI), Purity.
- **Visualization:** t-SNE projections of the latent space to visualize clusters by Genre and Language.


## Getting Started

### 1. Data Preparation
Ensure your datasets are placed in the `Datasets/` directory and update the paths in the configuration sections of the scripts if necessary.

## Dataset Access

The dataset used in this project can be downloaded from the link below:

> **[English songs dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)**

> **[Bangla songs dataset](https://www.kaggle.com/datasets/thisisjibon/banglabeats3sec)**

### 2. Preprocessing
Run the preprocessing scripts to generate feature files.

**Basic Preprocessing (for Simple VAE):**
```bash
python src/1_preprocessing.py
```
*Outputs to `processed_data1/`*

**Advanced Preprocessing (for Convolutional/Hybrid Models):**
```bash
python src/1_preprocessing_advanced.py
```
*Outputs to `processed_data2/`*

### 3. Training & Evaluation
Run any of the model scripts to train the VAE and evaluate clustering performance.

**Run Simple VAE:**
```bash
python src/Simple_VAE.py
```

**Run Convolutional (Hybrid) VAE:**
```bash
python src/Convolutional_VAE.py
```

**Run Conditional VAE:**
```bash
python src/Conditional_VAE.py
```

### 4. Viewing Results
- Check the `results/` folder for generated plots (`.png`) and metrics (`.csv`).
- The `clustering_metrics.csv` file consolidates results across different runs for easy comparison.

## key Findings
- **Latent Space Separation:** The project analyzes how well the VAE latent space separates songs based on Language (Bengali vs. English) and Genre.
- **Hybrid Features:** Combining lyrics with audio generally improves clustering performance compared to audio-only features.

## Authors
- **Shahriar1638** - *Initial Work*
