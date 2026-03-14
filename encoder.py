"""
Encoder-Only Model for Feature Learning
Architecture: 8 → 64 → 4 (latent features)
Purpose: Extract compressed representations of cloud workload features
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class Encoder(nn.Module):
    """
    Encoder neural network for feature extraction
    
    Architecture:
        Input: 8 features → Hidden: 64 neurons → Output: 4 latent features
    """
    
    def __init__(self, input_dim=8, latent_dim=4, hidden_dim=64):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        """Extract latent features from input"""
        return self.encoder(x)


class EncoderInference:
    """
    Wrapper class for easy encoder inference
    Handles loading, preprocessing, and feature extraction
    """
    
    def __init__(self, model_path, scaler_path):
        """
        Initialize encoder for inference
        
        Args:
            model_path: Path to trained encoder weights (.pth file)
            scaler_path: Path to fitted scaler (.pkl file)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load encoder
        self.encoder = Encoder(input_dim=8, latent_dim=4, hidden_dim=64)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Handle different state dict formats
        # If keys start with numbers (e.g., '0.weight'), load directly into encoder.encoder
        if any(key.startswith('0.') for key in state_dict.keys()):
            self.encoder.encoder.load_state_dict(state_dict)
        else:
            self.encoder.load_state_dict(state_dict)
        
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✓ Encoder loaded from: {model_path}")
        print(f"✓ Scaler loaded from: {scaler_path}")
        print(f"✓ Device: {self.device}")
    
    def extract_features(self, data):
        """
        Extract latent features from input data
        
        Args:
            data: numpy array of shape (n_samples, 8) or (8,)
        
        Returns:
            latent_features: numpy array of shape (n_samples, 4) or (4,)
        """
        # Handle single sample
        single_sample = False
        if data.ndim == 1:
            data = data.reshape(1, -1)
            single_sample = True
        
        # Normalize
        data_normalized = self.scaler.transform(data)
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(data_normalized).to(self.device)
        
        # Extract features
        with torch.no_grad():
            latent_features = self.encoder(data_tensor).cpu().numpy()
        
        # Return single sample if input was single sample
        if single_sample:
            return latent_features[0]
        
        return latent_features
    
    def process_dataframe(self, df, feature_columns=None):
        """
        Process a pandas DataFrame and add latent features
        
        Args:
            df: pandas DataFrame containing the 8 input features
            feature_columns: list of column names (default: standard 8 features)
        
        Returns:
            df_enhanced: DataFrame with 4 additional latent feature columns
        """
        if feature_columns is None:
            feature_columns = [
                'cpu_usage', 'memory_usage', 'network_traffic', 
                'power_consumption', 'execution_time', 'task_type', 
                'vm_type', 'historical_efficiency'
            ]
        
        # Extract features
        X = df[feature_columns].values
        latent_features = self.extract_features(X)
        
        # Add to dataframe
        df_enhanced = df.copy()
        df_enhanced['learned_f1'] = latent_features[:, 0]
        df_enhanced['learned_f2'] = latent_features[:, 1]
        df_enhanced['learned_f3'] = latent_features[:, 2]
        df_enhanced['learned_f4'] = latent_features[:, 3]
        
        return df_enhanced
    
    def process_csv(self, input_path, output_path, feature_columns=None):
        """
        Process a CSV file and save enhanced version with latent features
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save enhanced CSV file
            feature_columns: list of column names (default: standard 8 features)
        """
        print(f"Processing: {input_path}")
        
        # Load data
        df = pd.read_csv(input_path)
        print(f"  Loaded {len(df)} samples")
        
        # Process
        df_enhanced = self.process_dataframe(df, feature_columns)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_enhanced.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
        print(f"  Added features: learned_f1, learned_f2, learned_f3, learned_f4")
        
        return df_enhanced


def load_encoder(model_path='../../models/encoder_only.pth',
                 scaler_path='../../models/autoencoder_scaler.pkl'):
    """
    Convenience function to load encoder for inference
    
    Args:
        model_path: Path to encoder weights
        scaler_path: Path to scaler
    
    Returns:
        encoder_inference: EncoderInference object ready for use
    """
    return EncoderInference(model_path, scaler_path)


def main():
    """Example usage of encoder-only model"""
    print("=" * 60)
    print("ENCODER-ONLY FEATURE EXTRACTION")
    print("=" * 60)
    
    # Paths
    model_path = '../../models/encoder_only.pth'
    scaler_path = '../../models/autoencoder_scaler.pkl'
    input_csv = '../../data/cloud_workload_10k.csv'
    output_csv = '../../data/cloud_workload_12f_encoder.csv'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the autoencoder first!")
        return
    
    # Load encoder
    encoder_inf = load_encoder(model_path, scaler_path)
    
    # Example 1: Single sample
    print("\n" + "-" * 60)
    print("Example 1: Single Sample Feature Extraction")
    print("-" * 60)
    
    sample = np.array([
        75.5,  # cpu_usage
        60.2,  # memory_usage
        150.0, # network_traffic
        180.5, # power_consumption
        120.0, # execution_time
        0,     # task_type
        2,     # vm_type
        4      # historical_efficiency
    ])
    
    latent = encoder_inf.extract_features(sample)
    print(f"Input shape: {sample.shape}")
    print(f"Latent features shape: {latent.shape}")
    print(f"Latent features: {latent}")
    
    # Example 2: Batch processing
    print("\n" + "-" * 60)
    print("Example 2: Batch Processing")
    print("-" * 60)
    
    batch = np.random.rand(100, 8) * 100  # 100 random samples
    latent_batch = encoder_inf.extract_features(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Latent features shape: {latent_batch.shape}")
    
    # Example 3: Process CSV file
    if os.path.exists(input_csv):
        print("\n" + "-" * 60)
        print("Example 3: CSV File Processing")
        print("-" * 60)
        
        df_enhanced = encoder_inf.process_csv(input_csv, output_csv)
        
        print("\nEnhanced dataset preview:")
        print(df_enhanced[['cpu_usage', 'learned_f1', 'learned_f2', 
                          'learned_f3', 'learned_f4']].head())
    
    print("\n" + "=" * 60)
    print("ENCODER FEATURE EXTRACTION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
