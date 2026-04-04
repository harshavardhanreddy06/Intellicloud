# About Datasets

The IntelliCloud project relies on a high-fidelity dataset of cloud task profiles to train its machine learning models (Autoencoder, Random Forest, and DQN).

## 1. Dataset Overview
The primary dataset is stored in `dataset/task_profiles_full_12_features.json`, which contains over 10,000 observations. Each observation represents a historical execution of a specific cloud task (e.g., image resizing, video compression, PDF parsing).

## 2. Feature Structure (8 Base Features)
Each task profile includes the following benchmarked metrics:

| Feature | Description | Unit |
|---------|-------------|------|
| `input_size_mb` | Total size of the input payload | Megabytes (MB) |
| `cpu_usage_cores_absolute` | Normalized CPU consumption | Cores (0.0 - 1.0) |
| `memory_usage_mb` | Peak memory footprint during execution | Megabytes (MB) |
| `execution_time_normalized` | Time taken to process the task | Seconds (s) |
| `instruction_count` | Total CPU instructions retired | Count |
| `network_io_mb` | Network traffic generated | Megabytes (MB) |
| `power_consumption_watts` | Real-time power draw during task | Watts (W) |
| `task_size_category` | Ordinal classification (SMALL, MEDIUM, LARGE) | Category |

## 3. Enhanced Features (4 Latent Features)
The "12-feature" dataset adds four latent dimensions processed through the **Autoencoder**:
- `latent_f1` to `latent_f4`: Compressed state representations used to capture non-linear relationships between the 8 base features.

## 4. Task Types
The dataset covers 30 diverse task types across 4 major domains:
- **Image Processing**: Resize, Compression, Background Removal, Watermarking, etc.
- **Video Processing**: Trimming, GIFs, Subtitles, Frame Extraction, etc.
- **Audio Processing**: Noise Reduction, Normalization, Split Track, etc.
- **Office/PDF**: PDF Merging, Split, Office Conversion, Text Extraction.

## 5. Metadata
Each record also contains:
- `id`: Unique task identifier.
- `task_signature`: A unique hash/string identifier (e.g., `GOLD_vid_gif_creation`).
- `priority`: User-defined priority (low, medium, high, critical).
- `complexity`: Computational complexity (low, medium, high).
- `timestamp`: Execution time record.

## 6. How it's Used
- **Feature Extraction**: Provides the historical baseline for estimating costs of new, unseen tasks.
- **Autoencoder Training**: Used to learn the latent 12-feature space for better prediction.
- **Random Forest Training**: Labels (Energy Class 1-5) are generated from these metrics to train the energy efficiency classifier.
- **DQN Training**: The state space for the RL scheduler is derived from these task profiles.
