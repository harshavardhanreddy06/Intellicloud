"""
Autoencoder for IntelliCloud - Train on 8 features from task profiles
Based on SHERA methodology: 8 features → 4 latent features

This script adapts the autoencoder to work with our JSON dataset instead of CSV.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from pathlib import Path
import joblib


class Autoencoder(nn.Module):
    """
    Autoencoder neural network for dimensionality reduction

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
            nn.Linear(hidden_dim, latent_dim)  # Latent space (bottleneck)
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


class AutoencoderTrainer:
    """
    Trainer class for the autoencoder
    """

    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0

        for batch_data in dataloader:
            inputs = batch_data[0]

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, inputs)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def validate(self, dataloader):
        """Validate the model"""
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
        """
        Train the autoencoder

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
        """
        print("Starting Autoencoder Training...")
        print(f"Epochs: {epochs}")
        print("-" * 60)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        print("-" * 60)
        print("Training completed!")
        print(f"Final Train Loss: {self.train_losses[-1]:.6f}")
        print(f"Final Val Loss: {self.val_losses[-1]:.6f}")


def prepare_data_from_json(json_path, batch_size=32, val_split=0.2):
    """
    Load and prepare data from JSON for autoencoder training

    Args:
        json_path: Path to the task profiles JSON file
        batch_size: Batch size for training
        val_split: Validation split ratio

    Returns:
        train_loader, val_loader, scaler
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} task profiles from JSON")

    # Extract 8 features in the correct order (matching pipeline.py)
    X = []
    for record in data:
        features = [
            record['input_size_mb'],
            record['cpu_usage_cores_absolute'],
            record['memory_usage_mb'],
            record['execution_time_normalized'],
            record['instruction_count'],
            record.get('network_io_mb', 0),
            record['power_consumption_watts'],
            0 if record['task_size_category'] == 'SMALL' else
            1 if record['task_size_category'] == 'MEDIUM' else 2  # LARGE
        ]
        X.append(features)

    X = np.array(X, dtype=np.float32)
    print(f"Extracted {X.shape[1]} features for {X.shape[0]} samples")

    # Normalize to [0, 1] range
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Split into train and validation
    n_samples = len(X_normalized)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    X_train = X_normalized[:n_train]
    X_val = X_normalized[n_train:]

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data prepared: {n_train} training samples, {n_val} validation samples")

    return train_loader, val_loader, scaler


def extract_latent_features_to_json(model, scaler, json_input_path, json_output_path):
    """
    Extract latent features from JSON data and create enhanced dataset

    Args:
        model: Trained autoencoder model
        scaler: Fitted MinMaxScaler
        json_input_path: Path to input JSON file
        json_output_path: Path to save enhanced JSON with latent features
    """
    # Load JSON data
    with open(json_input_path, 'r') as f:
        data = json.load(f)

    print(f"Processing {len(data)} records for latent feature extraction...")

    # Extract 8 features for each record
    X = []
    for record in data:
        features = [
            record['input_size_mb'],
            record['cpu_usage_cores_absolute'],
            record['memory_usage_mb'],
            record['execution_time_normalized'],
            record['instruction_count'],
            record.get('network_io_mb', 0),
            record['power_consumption_watts'],
            0 if record['task_size_category'] == 'SMALL' else
            1 if record['task_size_category'] == 'MEDIUM' else 2  # LARGE
        ]
        X.append(features)

    X = np.array(X, dtype=np.float32)
    X_normalized = scaler.transform(X)

    # Extract latent features
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_normalized)
        latent_features = model.encode(X_tensor).numpy()

    # Add latent features to data
    for i, record in enumerate(data):
        record['latent_f1'] = float(latent_features[i, 0])
        record['latent_f2'] = float(latent_features[i, 1])
        record['latent_f3'] = float(latent_features[i, 2])
        record['latent_f4'] = float(latent_features[i, 3])

    # Save enhanced JSON
    with open(json_output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Enhanced dataset saved to: {json_output_path}")
    print("Added features: latent_f1, latent_f2, latent_f3, latent_f4")
    return data


def main():
    """Main training script for IntelliCloud autoencoder"""
    # Paths
    data_path = "dataset/task_profiles_clean_final.json"
    model_save_path = "models/autoencoder_encoder.pth"
    full_model_path = "models/autoencoder_full.pth"
    scaler_path = "models/autoencoder_scaler.pkl"
    enhanced_data_path = "dataset/task_profiles_12_features.json"

    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    # Prepare data
    print("=" * 60)
    print("AUTOENCODER TRAINING - IntelliCloud")
    print("=" * 60)
    train_loader, val_loader, scaler = prepare_data_from_json(data_path, batch_size=32)

    # Create model
    model = Autoencoder(input_dim=8, latent_dim=4, hidden_dim=64)
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    trainer = AutoencoderTrainer(model, learning_rate=0.001)
    trainer.train(train_loader, val_loader, epochs=50)

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.encoder.state_dict(), model_save_path)
    torch.save(model.state_dict(), full_model_path)
    print(f"\nEncoder saved to: {model_save_path}")
    print(f"Full model saved to: {full_model_path}")

    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # Extract latent features and create 12-feature dataset
    print("\n" + "=" * 60)
    print("EXTRACTING LATENT FEATURES")
    print("=" * 60)
    enhanced_data = extract_latent_features_to_json(
        model, scaler, data_path, enhanced_data_path
    )

    print("\n" + "=" * 60)
    print("AUTOENCODER TRAINING COMPLETE!")
    print("=" * 60)
    print("✓ Encoder trained and saved")
    print("✓ 12-feature dataset created")
    print("✓ Ready for Random Forest training")


if __name__ == "__main__":
    main()
