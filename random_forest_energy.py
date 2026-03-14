"""
Random Forest for Energy Efficiency Prediction
Based on SHERA methodology: 12 features (8 original + 4 latent) → Energy Efficiency Class (1-5)

Current implementation uses 8 features due to encoder loading issues.
Latent features from autoencoder need to be integrated once encoder is properly loaded.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

def compute_energy_efficiency_class(cpu_usage, memory_usage, power_consumption, min_cpu, max_cpu, min_memory, max_memory, min_power, max_power):
    """
    Compute energy efficiency percentage using composite score from CPU, memory, and power consumption.
    Higher efficiency = lower resource consumption across all metrics.

    Args:
        cpu_usage: CPU usage in cores
        memory_usage: Memory usage in MB
        power_consumption: Power in watts
        min_cpu, max_cpu: Min/max CPU values in dataset
        min_memory, max_memory: Min/max memory values in dataset
        min_power, max_power: Min/max power values in dataset

    Returns:
        efficiency_class: 1-5 (1=lowest efficiency, 5=highest efficiency)
    """
    # Normalize each metric to 0-1 scale (higher values = lower efficiency)
    if max_cpu > min_cpu:
        cpu_normalized = (cpu_usage - min_cpu) / (max_cpu - min_cpu)
    else:
        cpu_normalized = 0.5  # Default if no variation

    if max_memory > min_memory:
        memory_normalized = (memory_usage - min_memory) / (max_memory - min_memory)
    else:
        memory_normalized = 0.5  # Default if no variation

    if max_power > min_power:
        power_normalized = (power_consumption - min_power) / (max_power - min_power)
    else:
        power_normalized = 0.5  # Default if no variation

    # Compute composite efficiency score (weighted average)
    # Power gets higher weight (50%) as it's fundamental to energy efficiency
    # CPU: 30%, Memory: 20%, Power: 50%
    composite_score = (cpu_normalized * 0.3) + (memory_normalized * 0.2) + (power_normalized * 0.5)

    # Convert to efficiency percentage (100% = lowest resource usage, 0% = highest resource usage)
    efficiency_percentage = 100 * (1 - composite_score)

    # Bin into 5 classes (0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
    if efficiency_percentage <= 20:
        return 1
    elif efficiency_percentage <= 40:
        return 2
    elif efficiency_percentage <= 60:
        return 3
    elif efficiency_percentage <= 80:
        return 4
    else:
        return 5

def prepare_data(json_path):
    """
    Load data and prepare features + labels for Random Forest training.
    Now uses 12 features: 8 original + 4 latent from autoencoder.

    Returns:
        X: Feature matrix (12 features)
        y: Energy efficiency classes (1-5)
        feature_names: List of feature names
    """
    # Load dataset (now with latent features)
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} task profiles with 12 features")

    # Extract CPU, memory, and power consumption for efficiency calculation
    cpu_values = [record['cpu_usage_cores_absolute'] for record in data]
    memory_values = [record['memory_usage_mb'] for record in data]
    power_values = [record['power_consumption_watts'] for record in data]

    min_cpu, max_cpu = min(cpu_values), max(cpu_values)
    min_memory, max_memory = min(memory_values), max(memory_values)
    min_power, max_power = min(power_values), max(power_values)

    print(".2f")
    print(".2f")
    print(".2f")

    # Prepare features and labels
    X = []
    y = []
    feature_names = [
        'input_size_mb', 'cpu_usage_cores_absolute', 'memory_usage_mb',
        'execution_time_normalized', 'instruction_count', 'network_io_mb',
        'power_consumption_watts', 'task_size_category_encoded',
        'latent_f1', 'latent_f2', 'latent_f3', 'latent_f4'  # 4 latent features
    ]

    # Encode task_size_category
    size_encoder = LabelEncoder()
    size_categories = [record['task_size_category'] for record in data]
    size_encoded = size_encoder.fit_transform(size_categories)

    for i, record in enumerate(data):
        # 12 features: 8 original + 4 latent
        features = [
            record['input_size_mb'],
            record['cpu_usage_cores_absolute'],
            record['memory_usage_mb'],
            record['execution_time_normalized'],
            record['instruction_count'],
            record.get('network_io_mb', 0),
            record['power_consumption_watts'],
            size_encoded[i],
            record['latent_f1'],  # Latent features from autoencoder
            record['latent_f2'],
            record['latent_f3'],
            record['latent_f4']
        ]

        X.append(features)

        # Compute energy efficiency class using composite score
        efficiency_class = compute_energy_efficiency_class(
            record['cpu_usage_cores_absolute'],
            record['memory_usage_mb'],
            record['power_consumption_watts'],
            min_cpu, max_cpu, min_memory, max_memory, min_power, max_power
        )
        y.append(efficiency_class)

    X = np.array(X)
    y = np.array(y)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)} (classes 1-5)")

    return X, y, feature_names

def train_random_forest(X, y, feature_names):
    """
    Train Random Forest classifier with hyperparameter tuning.
    """
    print("\n" + "="*60)
    print("RANDOM FOREST TRAINING")
    print("="*60)

    # Split data (remove stratify due to small dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # Reduced test size for small dataset
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Scale features (as per SHERA methodology)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest with hyperparameters similar to SHERA
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Default, can be tuned
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    rf_model.fit(X_train_scaled, y_train)

    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
    print(".4f")

    # Test predictions
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Feature importance
    feature_importances = rf_model.feature_importances_
    print("\nFeature Importance:")
    for name, importance in zip(feature_names, feature_importances):
        print(f"{name:30}: {importance:.4f}")

    return rf_model, scaler, accuracy

def save_model(model, scaler, model_path, scaler_path):
    """Save trained model and scaler."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

def main():
    """Main training function."""
    # Paths
    data_path = "dataset/task_profiles_12_features.json"  # Now using 12-feature dataset
    model_path = "models/random_forest_energy_efficiency.pkl"
    scaler_path = "models/rf_scaler.pkl"

    # Prepare data
    print("Preparing data...")
    X, y, feature_names = prepare_data(data_path)

    # Train model
    model, scaler, accuracy = train_random_forest(X, y, feature_names)

    # Save model
    save_model(model, scaler, model_path, scaler_path)

    print(".4f")
    print("\n✅ Random Forest trained on 12 features (8 original + 4 latent)")
    print("✅ Ready for energy efficiency prediction as per SHERA methodology!")

if __name__ == "__main__":
    main()
