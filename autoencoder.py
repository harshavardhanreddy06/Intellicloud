"""
Autoencoder for Feature Learning (JSON Version)
Architecture: 8 → 64 → 4 (latent) → 64 → 8
Purpose: Learn compressed representations of cloud workload features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os


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
    
    def plot_losses(self, save_path=None):
        """Plot training and validation losses"""
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


def prepare_data(json_path, batch_size=64, val_split=0.2):
    """
    Load and prepare data for autoencoder training from JSON
    
    Args:
        json_path: Path to the dataset JSON file
        batch_size: Batch size for training
        val_split: Validation split ratio
    
    Returns:
        train_loader, val_loader, scaler
    """
    # Load JSON dataset
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract 8 original features from each record
    feature_data = []
    for record in data:
        features = [
            record['input_size_mb'],
            record['cpu_usage_cores_absolute'], 
            record['memory_usage_mb'],
            record['execution_time_normalized'],
            record['instruction_count'],
            record.get('network_io_mb', 0),
            record['power_consumption_watts'],
            # Encode task_size_category: SMALL=0, MEDIUM=1, LARGE=2
            0 if record['task_size_category'] == 'SMALL' else 
            1 if record['task_size_category'] == 'MEDIUM' else 2
        ]
        feature_data.append(features)
    
    X = np.array(feature_data)
    
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
    
    print(f"Data loaded: {n_train} training samples, {n_val} validation samples")
    print(f"Features: input_size_mb, cpu_usage_cores_absolute, memory_usage_mb, execution_time_normalized, instruction_count, network_io_mb, power_consumption_watts, task_size_category_encoded")
    
    return train_loader, val_loader, scaler


def extract_latent_features(model, json_path, scaler, output_path):
    """
    Extract latent features and create 12-feature dataset from JSON
    
    Args:
        model: Trained autoencoder model
        json_path: Path to original JSON dataset
        scaler: Fitted MinMaxScaler
        output_path: Path to save 12-feature dataset (JSON)
    """
    # Load original JSON dataset
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract 8 original features for latent feature extraction
    feature_data = []
    for record in data:
        features = [
            record['input_size_mb'],
            record['cpu_usage_cores_absolute'], 
            record['memory_usage_mb'],
            record['execution_time_normalized'],
            record['instruction_count'],
            record.get('network_io_mb', 0),
            record['power_consumption_watts'],
            # Encode task_size_category: SMALL=0, MEDIUM=1, LARGE=2
            0 if record['task_size_category'] == 'SMALL' else 
            1 if record['task_size_category'] == 'MEDIUM' else 2
        ]
        feature_data.append(features)
    
    X = np.array(feature_data)
    X_normalized = scaler.transform(X)
    
    # Extract latent features
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_normalized)
        latent_features = model.encode(X_tensor).numpy()
    
    # Create new dataset with 12 features (8 original + 4 latent)
    enhanced_data = []
    for i, record in enumerate(data):
        enhanced_record = record.copy()
        enhanced_record['latent_f1'] = float(latent_features[i, 0])
        enhanced_record['latent_f2'] = float(latent_features[i, 1])
        enhanced_record['latent_f3'] = float(latent_features[i, 2])
        enhanced_record['latent_f4'] = float(latent_features[i, 3])
        enhanced_data.append(enhanced_record)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    print(f"12-feature dataset saved to: {output_path}")
    print(f"New features: latent_f1, latent_f2, latent_f3, latent_f4")
    print(f"Total features: 8 original + 4 latent = 12 features")
    
    return enhanced_data


def main():
    """Main training script"""
    # Paths (JSON format)
    data_path = 'dataset/task_profiles_clean_final.json'  # JSON dataset
    model_save_path = 'models/autoencoder_encoder.pth'
    full_model_path = 'models/autoencoder_full.pth'
    scaler_path = 'models/autoencoder_scaler.pkl'
    loss_plot_path = 'results/autoencoder_loss.png'
    output_12f_path = 'dataset/task_profiles_12_features.json'  # JSON output
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please ensure the JSON dataset exists!")
        return
    
    # Prepare data
    print("=" * 60)
    print("AUTOENCODER TRAINING (JSON)")
    print("=" * 60)
    train_loader, val_loader, scaler = prepare_data(data_path, batch_size=64)
    
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
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    
    # Plot losses
    os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
    trainer.plot_losses(save_path=loss_plot_path)
    
    # Extract latent features
    print("\n" + "=" * 60)
    print("EXTRACTING LATENT FEATURES")
    print("=" * 60)
    enhanced_data = extract_latent_features(model, data_path, scaler, output_12f_path)
    
    print("\n" + "=" * 60)
    print("AUTOENCODER TRAINING COMPLETE!")
    print("=" * 60)
    print(f"✓ Encoder saved")
    print(f"✓ 12-feature dataset created (JSON)")
    print(f"✓ Ready for Random Forest training")


if __name__ == "__main__":
    main()
