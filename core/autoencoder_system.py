"""
IntelliCloud Autoencoder System - Unified Module
Combines training, inference, and data processing for the SHERA methodology.

This module consolidates:
- autoencoder_intellicloud.py (JSON training & extraction)
- autoencoder.py (Training metrics & plotting)
- encoder.py (High-level inference API)
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from pathlib import Path
import joblib
import pickle
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. Model Architectures
# ----------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Encoder-only neural network for feature extraction
    Architecture: 8 → 64 → 4
    """
    def __init__(self, input_dim=8, latent_dim=4, hidden_dim=64):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class Autoencoder(nn.Module):
    """
    Full Autoencoder neural network for dimensionality reduction
    Architecture:
        Encoder: 8 → 64 → 4
        Decoder: 4 → 64 → 8
    """
    def __init__(self, input_dim=8, latent_dim=4, hidden_dim=64):
        super(Autoencoder, self).__init__()
        
        # Encoder: Compress 8 features to 4 latent features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder: Reconstruct 8 features from 4 latent features
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output in range [0, 1] (normalized)
        )

    def forward(self, x):
        """Forward pass through encoder and decoder"""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        """Extract latent features only"""
        return self.encoder(x)

# ----------------------------------------------------------------------------
# 2. Training Utilities
# ----------------------------------------------------------------------------

class AutoencoderTrainer:
    """
    Trainer class for the autoencoder with integrated plotting
    """
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0.0
        for batch_data in dataloader:
            inputs = batch_data[0]
            outputs = self.model(inputs)
            loss = self.criterion(outputs, inputs)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in dataloader:
                inputs = batch_data[0]
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                val_loss += loss.item()
        return val_loss / len(dataloader)

    def train(self, train_loader, val_loader, epochs=50):
        print("Starting Autoencoder Training...")
        print(f"Epochs: {epochs}")
        print("-" * 60)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        print("-" * 60)
        print("Training completed!")
        return self.train_losses, self.val_losses

    def plot_losses(self, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Autoencoder Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss plot saved to: {save_path}")
        plt.close()

# ----------------------------------------------------------------------------
# 3. Data Processing
# ----------------------------------------------------------------------------

def prepare_data_from_json(json_path, batch_size=8, val_split=0.2):
    """Load and prepare data for autoencoder training from JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 🧬 Augment tiny dataset (duplicate 30 -> 300) to ensure stable local training
    if len(data) < 100:
        data = data * 10 

    X = []
    for record in data:
        # Flexible mapping for size categories
        cat = record.get('task_size_category', 'MEDIUM')
        sz_val = 0 if 'SMALL' in cat or 'LOW' in cat else (2 if 'LARGE' in cat or 'HIGH' in cat else 1)
        
        features = [
            record.get('input_size_mb', 1.0),
            record.get('cpu_usage_cores_absolute', 0.1),
            record.get('memory_usage_mb', 50.0),
            record.get('execution_time_normalized', record.get('execution_time_seconds', 1.0)),
            record.get('instruction_count', 1000000),
            record.get('network_io_mb', 0),
            record.get('power_consumption_watts', 15.0),
            sz_val
        ]
        X.append(features)

    X = np.array(X, dtype=np.float32)
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    n_samples = len(X_normalized)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    X_train_tensor = torch.FloatTensor(X_normalized[:n_train])
    X_val_tensor = torch.FloatTensor(X_normalized[n_train:])

    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor), batch_size=batch_size, shuffle=False)

    print(f"Data prepared: {n_train} training samples, {n_val} validation samples")
    return train_loader, val_loader, scaler

# ----------------------------------------------------------------------------
# 4. Inference & Feature Extraction
# ----------------------------------------------------------------------------

class EncoderInference:
    """Wrapper class for high-level encoder inference"""
    def __init__(self, model_path, scaler_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize encoder
        self.encoder = Encoder(input_dim=8, latent_dim=4, hidden_dim=64)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        # Handle different state dict formats (full model vs encoder-only)
        if any(key.startswith('encoder.') for key in state_dict.keys()):
            # If it's a full Autoencoder state dict, extract just the encoder
            encoder_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            self.encoder.encoder.load_state_dict(encoder_dict)
        elif any(k.startswith('0.') for k in state_dict.keys()):
            # Direct Sequential state dict
            self.encoder.encoder.load_state_dict(state_dict)
        else:
            self.encoder.load_state_dict(state_dict)
            
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # Load scaler
        if str(scaler_path).endswith('.pkl'):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = joblib.load(scaler_path)
        
        print(f"✓ Encoder loaded from: {model_path}")
        print(f"✓ Scaler loaded from: {scaler_path}")

    def extract_features(self, data):
        """Extract latent features from input array"""
        single_sample = data.ndim == 1
        if single_sample:
            data = data.reshape(1, -1)
        
        data_normalized = self.scaler.transform(data)
        data_tensor = torch.FloatTensor(data_normalized).to(self.device)
        
        with torch.no_grad():
            latent_features = self.encoder(data_tensor).cpu().numpy()
        
        return latent_features[0] if single_sample else latent_features

def extract_latent_to_json(model, scaler, input_json, output_json):
    """Process JSON data and add latent features"""
    with open(input_json, 'r') as f:
        data = json.load(f)

    X = []
    for r in data:
        cat = r.get('task_size_category', 'MEDIUM')
        sz_val = 0 if 'SMALL' in cat or 'LOW' in cat else (2 if 'LARGE' in cat or 'HIGH' in cat else 1)
        
        X.append([
            r.get('input_size_mb', 1.0),
            r.get('cpu_usage_cores_absolute', 0.1),
            r.get('memory_usage_mb', 50.0),
            r.get('execution_time_normalized', r.get('execution_time_seconds', 1.0)),
            r.get('instruction_count', 1000000),
            r.get('network_io_mb', 0),
            r.get('power_consumption_watts', 15.0),
            sz_val
        ])

    X_normalized = scaler.transform(np.array(X, dtype=np.float32))
    
    model.eval()
    with torch.no_grad():
        latent = model.encode(torch.FloatTensor(X_normalized)).numpy()

    for i, record in enumerate(data):
        for j in range(4):
            record[f'latent_f{j+1}'] = float(latent[i, j])

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Enhanced dataset saved to: {output_json}")
    return data

# ----------------------------------------------------------------------------
# 5. Main Execution Flow
# ----------------------------------------------------------------------------

def main():
    """Example training workflow"""
    data_path = "dataset/task_profiles_full_training.json"
    benchmark_path = "dataset/task_profiles_clean_final.json"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Skipping high-accuracy training.")
        return

    # 🎯 Train on full 10k observation set for Better Accuracy
    train_loader, val_loader, scaler = prepare_data_from_json(data_path, batch_size=64)
    model = Autoencoder()
    trainer = AutoencoderTrainer(model)
    trainer.train(train_loader, val_loader, epochs=100)
    
    # Save
    torch.save(model.state_dict(), model_dir / "autoencoder_full.pth")
    torch.save(model.encoder.state_dict(), model_dir / "autoencoder_encoder.pth")
    joblib.dump(scaler, model_dir / "autoencoder_scaler.pkl")
    
    # 🏁 Extract latent features for the 30 Gold Benchmarks (Dashboard UI stability)
    extract_latent_to_json(model, scaler, benchmark_path, "dataset/task_profiles_12_features.json")
    
    # 📈 Extract latent features for the FULL 10k set (High-Accuracy RF Training)
    extract_latent_to_json(model, scaler, data_path, "dataset/task_profiles_full_12_features.json")
    print("✓ Full 10k latent extraction complete.")

if __name__ == "__main__":
    main()
