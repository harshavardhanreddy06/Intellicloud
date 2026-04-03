import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
import torch
import shap
from pathlib import Path
from datetime import datetime

# Local imports
from feature_extractor import TaskFeatureExtractor
from autoencoder_system import Encoder

# Container definitions (also moved here for consistency)
CONTAINER_PROFILES = [
    {"id": 0, "name": "Tiny",   "cpu_cores": 0.25, "memory_mb": 256,  "label": "0.25 core / 256 MB"},
    {"id": 1, "name": "Medium", "cpu_cores": 0.50, "memory_mb": 512,  "label": "0.50 core / 512 MB"},
    {"id": 2, "name": "Large",  "cpu_cores": 1.00, "memory_mb": 1024, "label": "1.00 core / 1024 MB"},
]

TASK_REQUEST_MAP = {
    "SMALL":  {"cpu": 0.15, "memory": 128},
    "MEDIUM": {"cpu": 0.40, "memory": 384},
    "LARGE":  {"cpu": 0.80, "memory": 768},
}

class IntelliCloudPredictor:
    """Orchestrator for SHERA + DQN Pipeline"""
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine workspace root
        root = Path(__file__).resolve().parent
        history_path = str(root / "dataset" / "task_profiles_clean_final.json")
        self.extractor = TaskFeatureExtractor(history_path=history_path)
        
        # 1. Load Autoencoder
        self.encoder = Encoder(input_dim=8, latent_dim=4)
        encoder_path = root / "models" / "autoencoder_encoder.pth"
        if encoder_path.exists():
            state_dict = torch.load(encoder_path, map_location=self.device)
            # Handle sequential vs class-wrapped keys
            if any(k.startswith("0.") for k in state_dict.keys()):
                self.encoder.encoder.load_state_dict(state_dict)
            else:
                self.encoder.load_state_dict(state_dict)
            self.encoder.to(self.device).eval()
            print(f"   ✓ Encoder loaded from {encoder_path}")
        
        # 2. Load Random Forest
        rf_path = root / "models" / "random_forest_energy_efficiency.pkl"
        if rf_path.exists():
            self.rf_model = joblib.load(rf_path)
            print(f"   ✓ RF Energy model loaded from {rf_path}")
            # SHAP Explainer
            self.explainer = shap.TreeExplainer(self.rf_model)
        else:
            self.rf_model = None
            print(f"   ⚠️ RF model not found at {rf_path}")

        # 3. Load Scalers
        scaler_path = self.model_dir / "feature_scaler.pkl"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        
        # 4. Load DQN Agent (for integration demo only, real logic is in start.py)
        # Note: We only import DQNAgent here if needed for internal prediction
        from src.rl_scheduler.dqn_agent import DQNAgent
        self.dqn_agent = DQNAgent(state_dim=13, action_dim=3)
        dqn_path = self.model_dir / "dqn_scheduler.pth"
        if dqn_path.exists():
            self.dqn_agent.load(str(dqn_path))
        
        # Load DQN State Scaler
        dqn_scaler_path = self.model_dir / "dqn_state_scaler.pkl"
        self.dqn_state_scaler = joblib.load(dqn_scaler_path) if dqn_scaler_path.exists() else None

    def predict_energy_efficiency(self, task_data, include_shap=True):
        """Full SHERA Prediction Flow for a single task."""
        # A. Feature Extraction (8 features)
        features_8_dict = self.extractor.extract_features(task_data)
        features_8_list = [
            features_8_dict['input_size_mb'], features_8_dict['cpu_usage_cores_absolute'],
            features_8_dict['memory_usage_mb'], features_8_dict['execution_time_normalized'],
            features_8_dict['instruction_count'], features_8_dict['network_io_mb'],
            features_8_dict['power_consumption_watts'],
            0 if features_8_dict['task_size_category'] == 'SMALL' else 1 if features_8_dict['task_size_category'] == 'MEDIUM' else 2
        ]
        
        # Scale for encoder
        x_scaled = self.scaler.transform([features_8_list]) if self.scaler else [features_8_list]
        x_tensor = torch.FloatTensor(x_scaled).to(self.device)
        
        # B. Latent Feature Extraction (4 features)
        with torch.no_grad():
            latent = self.encoder(x_tensor).cpu().numpy()[0]
        
        # C. Random Forest Prediction (12 features)
        state_12 = features_8_list + list(latent)
        prediction = int(self.rf_model.predict([state_12])[0]) if self.rf_model else 3
        probs = self.rf_model.predict_proba([state_12])[0] if self.rf_model else [0.2]*5
        
        # Efficiency Level Mapping
        levels = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}
        
        # SHAP EXPLANATION
        shap_path = "Not created"
        if include_shap and self.rf_model:
            try:
                task_id = f"{task_data.get('task_type', 'task')}_{task_data.get('input_size_mb', 0):.2f}MB"
                shap_path = self.generate_shap_explanation(state_12, prediction, task_id)
            except Exception as e:
                shap_path = f"Error: {e}"

        result = {
            "task_info": task_data,
            "features": {**features_8_dict, "latent_f1": float(latent[0]), "latent_f2": float(latent[1]), "latent_f3": float(latent[2]), "latent_f4": float(latent[3])},
            "prediction": {
                "energy_efficiency_class": prediction,
                "efficiency_level": f"{levels.get(prediction, 'Unknown')} ({20*(prediction-1)}-{20*prediction}%)",
                "confidence": float(max(probs)),
                "explanation_image": shap_path
            }
        }
        
        return result

    def generate_shap_explanation(self, features_12, prediction, task_id):
        import matplotlib.pyplot as plt
        
        # We need to scale the features using the RF Scaler. The Random Forest pipeline previously used `features_scaled`.
        # However, earlier in this code we don't have an RF specific scaler initialized, we just passed `state_12`. 
        # Wait, if RF was trained on `state_12` unscaled, then we pass `state_12`. We'll just pass `state_12` as a 2D array.
        X_df = pd.DataFrame([features_12], columns=[
            'Input Size MB', 'CPU Cores Abs', 'Memory MB', 'Exec Time Norm',
            'Instructions', 'Network IO', 'Power Watts', 'Size Class',
            'Latent F1', 'Latent F2', 'Latent F3', 'Latent F4'
        ])
        
        shap_values = self.explainer.shap_values(X_df)
        
        plt.figure(figsize=(10, 6))
        
        # Extract correct SHAP values for the specific prediction class
        if isinstance(shap_values, list):
            sv = shap_values[prediction - 1] if (prediction - 1) < len(shap_values) else shap_values[0]
            sv_1d = sv[0] if len(sv.shape) == 2 else sv
        else:
            sv_1d = shap_values[0,:,prediction-1] if len(shap_values.shape) == 3 else shap_values[0]
            
        bv = self.explainer.expected_value
        if isinstance(bv, (list, np.ndarray)):
            bv = float(bv[prediction - 1]) if (prediction - 1) < len(bv) else float(bv[0])
        else:
            bv = float(bv)
            
        shap.plots.waterfall(shap.Explanation(values=sv_1d, base_values=bv, data=X_df.iloc[0]), show=False)
        
        plt.title(f"SHAP Explanation - {task_id}")
        plt.tight_layout()
        
        output_dir = Path("shap_explanations")
        output_dir.mkdir(exist_ok=True)
        filename = f"shap_{task_id.replace('.','_').replace(' ', '')}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=120)
        plt.close()
        
        return str(filepath)

# --- Global Predictor Singleton ---
_predictor_instance = None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = IntelliCloudPredictor()
    return _predictor_instance
