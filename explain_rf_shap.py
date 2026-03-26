import joblib
import json
import numpy as np
import pandas as pd
import shap
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pathlib import Path

def prepare_data(json_path):
    """
    Load data and prepare features for SHAP explanation.
    Consistent with random_forest_energy.py
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Prepare features
    feature_names = [
        'input_size_mb', 'cpu_usage_cores_absolute', 'memory_usage_mb',
        'execution_time_normalized', 'instruction_count', 'network_io_mb',
        'power_consumption_watts', 'task_size_category_encoded',
        'latent_f1', 'latent_f2', 'latent_f3', 'latent_f4'
    ]

    # Encode task_size_category
    size_encoder = LabelEncoder()
    size_categories = [record['task_size_category'] for record in data]
    size_encoded = size_encoder.fit_transform(size_categories)

    X = []
    for i, record in enumerate(data):
        features = [
            record['input_size_mb'],
            record['cpu_usage_cores_absolute'],
            record['memory_usage_mb'],
            record['execution_time_normalized'],
            record['instruction_count'],
            record.get('network_io_mb', 0),
            record['power_consumption_watts'],
            size_encoded[i],
            record['latent_f1'],
            record['latent_f2'],
            record['latent_f3'],
            record['latent_f4']
        ]
        X.append(features)

    return np.array(X), feature_names

def generate_sample_explanation(model, scaler, X_scaled, X_df, shap_values, explainer, sample_idx, output_dir):
    """
    Generate a comparison plot showing why the predicted class was chosen over the runner-up.
    """
    # Get all probabilities
    probs = model.predict_proba(X_scaled[sample_idx:sample_idx+1])[0]
    classes = model.classes_
    
    # Sort classes by probability
    sorted_idx = np.argsort(probs)[::-1]
    top_class_idx = sorted_idx[0]
    second_class_idx = sorted_idx[1]
    
    pred_class = classes[top_class_idx]
    second_class = classes[second_class_idx]
    
    print(f"Generating comparison for sample {sample_idx}: Class {pred_class} (Conf: {probs[top_class_idx]:.2f}) vs Class {second_class} (Conf: {probs[second_class_idx]:.2f})")
    
    # SHAP values for top two classes
    shap_top = shap_values[sample_idx, :, top_class_idx]
    shap_second = shap_values[sample_idx, :, second_class_idx]
    
    feature_names = X_df.columns
    
    # Sort features by importance for the chosen class
    sort_idx = np.argsort(np.abs(shap_top))
    
    names_sorted = feature_names[sort_idx]
    top_sorted = shap_top[sort_idx]
    second_sorted = shap_second[sort_idx]
    
    # Labels with values
    sample_values = X_df.iloc[sample_idx].values[sort_idx]
    labels = [f"{n} ({v:.2f})" for n, v in zip(names_sorted, sample_values)]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y = np.arange(len(labels))
    height = 0.35
    
    ax.barh(y + height/2, top_sorted, height, label=f'Chosen: Class {pred_class} ({probs[top_class_idx]*100:.1f}%)', color='#ff0051')
    ax.barh(y - height/2, second_sorted, height, label=f'Runner-up: Class {second_class} ({probs[second_class_idx]*100:.1f}%)', color='#008bfb', alpha=0.6)
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('SHAP Value (Feature Influence)')
    ax.set_title(f'SHAP Comparison: Why RF chosen Class {pred_class} over Class {second_class}\nSample {sample_idx}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = output_dir / f"sample_{sample_idx}_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Comparison plot saved to: {output_path}")

def main():
    # Paths
    data_path = "dataset/task_profiles_12_features.json"
    model_path = "models/random_forest_energy_efficiency.pkl"
    scaler_path = "models/rf_scaler.pkl"
    output_dir = Path("shap_explanations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Allow passing sample index via command line
    sample_to_explain = 0
    if len(sys.argv) > 1:
        try:
            sample_to_explain = int(sys.argv[1])
        except ValueError:
            print("Usage: python3 explain_rf_shap.py [sample_index]")
            sys.exit(1)

    # Load model and scaler
    print("Loading model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Prepare data
    print("Preparing data...")
    X, feature_names = prepare_data(data_path)
    
    if sample_to_explain >= len(X):
        print(f"Error: Sample index {sample_to_explain} out of range (max {len(X)-1})")
        sys.exit(1)

    # Scale features
    X_scaled = scaler.transform(X)
    X_df = pd.DataFrame(X_scaled, columns=feature_names)

    # Initialize SHAP explainer
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for the specific sample or all if needed
    # To be fast, we only compute for all once, or just the one needed
    # But usually, global plots need all.
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_scaled)
    
    classes = model.classes_

    # Generate Global Insights (only once)
    if not (output_dir / "global_summary_plot.png").exists():
        print("Generating global insights...")
        plt.figure(figsize=(10, 6))
        shap_values_list = [shap_values[:, :, i] for i in range(len(classes))]
        shap.summary_plot(shap_values_list, X_df, class_names=list(classes), show=False)
        plt.title("Overall SHAP Feature Importance across all Classes")
        plt.tight_layout()
        plt.savefig(output_dir / "global_summary_plot.png")
        plt.close()

    # Generate explanation for the requested sample
    generate_sample_explanation(model, scaler, X_scaled, X_df, shap_values, explainer, sample_to_explain, output_dir)

    print(f"\n✅ SHAP explanation complete for sample {sample_to_explain}")

if __name__ == "__main__":
    main()
