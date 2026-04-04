# Autoencoder

The Autoencoder is the core dimensionality reduction component of the SHERA (SHAP-Enhanced Resource Allocation) pipeline.

## 1. Architecture
The Autoencoder is implemented in `core/autoencoder_system.py` as a PyTorch neural network.

### Encoder (8 → 64 → 4)
- **Input (8 Features)**: Receives the telemetry vector from the feature extractor.
- **Hidden Layer (64 Neurons)**: High-dimensional non-linear transformation with ReLU activation.
- **Latent Layer (4 Neurons)**: Compresses the 8 features into 4 "latent" dimensions.

### Decoder (4 → 64 → 8)
- **Latent Input (4 Features)**: Input from the latent space.
- **Hidden Layer (64 Neurons)**: Reconstruction layer with ReLU activation.
- **Output (8 Features)**: Reconstructs the original features using a Sigmoid activation (for normalized outputs).

## 2. Training Workflow
The model is trained on `dataset/task_profiles_full_training.json` (10,000+ records).

1. **Min-Max Scaling**: Normalizes all 8 base features in the [0, 1] range.
2. **Loss Function**: Mean Squared Error (MSE) measures the difference between reconstructed and original features.
3. **Optimizer**: Adam optimizer with 0.001 learning rate.
4. **Epochs**: Trained for 100 epochs until the validation loss stabilizes.

## 3. How Latent Features are Formed
Latent features are the "hidden" representations learned by the encoder after passing through the bottleneck. These features often capture:
- **Latent F1**: Computation intensity vs Input size.
- **Latent F2**: Memory bottleneck vs Execution time.
- **Latent F3**: Energy efficiency per instruction.
- **Latent F4**: I/O and Network overhead patterns.

## 4. Integration in SHERA
The 4 latent features are combined with the 8 original features to create a **12-feature vector**.
- This 12-feature vector is used to train the **Random Forest Energy Efficiency Classifier**.
- Research indicates that using latent features significantly improves the accuracy of classifying energy efficiency.

## 5. Summary of Model Files
- `models/autoencoder_full.pth`: Full weights for both Encoder and Decoder.
- `models/autoencoder_encoder.pth`: Compressed weights for just the Encoder (used during live inference).
- `models/autoencoder_scaler.pkl`: The MinMaxScaler used to normalize live task inputs to the encoder.
