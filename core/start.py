"""
IntelliCloud Complete System
SHERA Methodology: Task → 8 Features → Autoencoder → 12 Features → Random Forest → Energy Efficiency

Combined file with:
- TaskPipeline: Feature processing pipeline
- IntelliCloudPredictor: Production API for predictions
- demonstrate_complete_pipeline(): Demo and testing functionality

Usage:
    python intellicloud_complete.py  # Runs demo
    python -c "from intellicloud_complete import predict_task_energy; print(predict_task_energy({...}))"
"""

import json
import numpy as np
import torch
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import sys

# Ensure core directory is in the search path for neighboring modules
CORE_DIR = Path(__file__).resolve().parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

# Add project root to path for src imports
ROOT_DIR = CORE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rl_scheduler.dqn_agent import DQNAgent
import shap
import matplotlib.pyplot as plt
from feature_extractor import TaskFeatureExtractor
from autoencoder_system import Encoder

# ----------------------------------------------------------------------------
# Feature ordering — maps feature_extractor output → encoder input vector
# ----------------------------------------------------------------------------
# The encoder was trained on 8 numeric columns in this exact order.
# task_size_category is ordinal-encoded: SMALL=0, MEDIUM=1, LARGE=2
FEATURE_ORDER = [
    "input_size_mb",               # position 0
    "cpu_usage_cores_absolute",    # position 1
    "memory_usage_mb",             # position 2
    "execution_time_normalized",   # position 3
    "instruction_count",           # position 4
    "network_io_mb",               # position 5
    "power_consumption_watts",     # position 6
    "task_size_category",          # position 7  (encoded below)
]

SIZE_CATEGORY_MAP = {"SMALL": 0, "MEDIUM": 1, "LARGE": 2}

# ----------------------------------------------------------------------------
# Main Pipeline class
# ----------------------------------------------------------------------------
class TaskPipeline:
    """
    Connects TaskFeatureExtractor to the trained Encoder.

    Usage:
        pipeline = TaskPipeline()
        result = pipeline.run(incoming_task)
        # result contains the original 8 features + latent_f1..f4
    """

    def __init__(
        self,
        model_path: str = None,
        scaler_path: str = None,
        history_path: str = None,
    ):
        """
        Args:
            model_path:   Path to encoder_only.pth  (auto-resolved if None)
            scaler_path:  Path to autoencoder_scaler.pkl (auto-resolved if None)
            history_path: Path to task_profiles_clean_final.json (auto-resolved if None)
        """
        root = Path(__file__).resolve().parent.parent
        self.model_path   = Path(model_path) if model_path else root / "models" / "autoencoder_encoder.pth"
        history_path      = history_path or str(root / "dataset" / "task_profiles_clean_final.json")

        # 1. Feature extractor
        self.extractor = TaskFeatureExtractor(history_path=history_path)

        # 2. Scaler: fit from historical task profiles
        self._fit_scaler()

        # 3. Encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_encoder()

        print(f"✓ Pipeline ready  |  device={self.device}")

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _fit_scaler(self):
        """
        Fit a MinMaxScaler on the 8 numeric features from the historical
        task profiles JSON.  This guarantees the scaler domain exactly
        matches the feature_extractor output domain.
        """
        history_path = Path(self.extractor.history_path)
        with open(history_path) as f:
            records = json.load(f)

        rows = []
        for r in records:
            size_cat = SIZE_CATEGORY_MAP.get(r.get("task_size_category", "MEDIUM"), 1)
            rows.append([
                float(r["input_size_mb"]),
                float(r["cpu_usage_cores_absolute"]),
                float(r["memory_usage_mb"]),
                float(r["execution_time_normalized"]),
                float(r["instruction_count"]),
                float(r.get("network_io_mb", 0)),
                float(r["power_consumption_watts"]),
                float(size_cat),
            ])

        X = np.array(rows, dtype=np.float32)
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        print(f"✓ Scaler fitted   |  {len(records)} historical task profiles")

    def _load_encoder(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Encoder model not found: {self.model_path}")

        self.encoder = Encoder(input_dim=8, latent_dim=4, hidden_dim=64)
        state_dict = torch.load(self.model_path, map_location=self.device)

        # Handle both "0.weight" style and "encoder.0.weight" style keys
        if any(k.startswith("0.") for k in state_dict.keys()):
            self.encoder.encoder.load_state_dict(state_dict)
        else:
            self.encoder.load_state_dict(state_dict)

        self.encoder.to(self.device)
        self.encoder.eval()
        print(f"✓ Encoder loaded  |  {self.model_path.name}")

    # ------------------------------------------------------------------
    # Feature dict → ordered numpy array
    # ------------------------------------------------------------------
    def _to_array(self, features: dict) -> np.ndarray:
        """Convert feature_extractor output dict to ordered (8,) numpy array."""
        row = []
        for field in FEATURE_ORDER:
            val = features[field]
            if field == "task_size_category":
                val = SIZE_CATEGORY_MAP.get(val, 1)   # default MEDIUM=1
            row.append(float(val))
        return np.array(row, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, incoming_task: dict) -> dict:
        """
        Full pipeline: task dict → features + 4 latent values.

        Args:
            incoming_task: dict with keys like task_type, input_size_mb,
                           complexity, priority, application.

        Returns:
            dict with:
              - all 8 numeric features from the extractor
              - latent_f1, latent_f2, latent_f3, latent_f4
              - task metadata fields (task_type, task_category, priority, etc.)

        Returns None if feature extraction fails (e.g. input_size_mb <= 0).
        """
        # Step 1: Extract features
        features = self.extractor.extract_features(incoming_task)
        if features is None:
            print("⚠️  Feature extraction returned None — check input_size_mb > 0")
            return None

        # Step 2: Build ordered numpy array
        raw_array = self._to_array(features)          # shape (8,)

        # Step 3: MinMaxScale to [0, 1] using scaler fitted on same feature domain
        normalized = self.scaler.transform(raw_array.reshape(1, -1))    # (1, 8)

        # Step 4: Encode → 4 latent features
        tensor = torch.FloatTensor(normalized).to(self.device)
        with torch.no_grad():
            latent = self.encoder(tensor).cpu().numpy()[0]              # (4,)

        # Step 5: Build result
        result = {
            # --- 8 extracted features ---
            "input_size_mb":             round(features["input_size_mb"], 4),
            "cpu_usage_cores_absolute":  round(features["cpu_usage_cores_absolute"], 4),
            "memory_usage_mb":           round(features["memory_usage_mb"], 4),
            "execution_time_normalized": round(features["execution_time_normalized"], 4),
            "instruction_count":         int(features["instruction_count"]),
            "network_io_mb":             round(features["network_io_mb"], 4),
            "power_consumption_watts":   round(features["power_consumption_watts"], 4),
            "task_size_category":        features["task_size_category"],
            # --- 4 latent features from encoder ---
            "latent_f1": round(float(latent[0]), 6),
            "latent_f2": round(float(latent[1]), 6),
            "latent_f3": round(float(latent[2]), 6),
            "latent_f4": round(float(latent[3]), 6),
            # --- metadata ---
            "task_type":     features["task_type"],
            "task_category": features["task_category"],
            "priority":      features["priority"],
            "complexity":    features["complexity"],
            "application":   features["application"],
            "exact_match_found": features["exact_match_found"],  # Boolean flag for exact match vs estimation
            "is_scaled": features["is_scaled"],                   # Boolean flag for scaling usage
            "is_having_history": features["is_having_history"]   # Boolean flag for history availability
        }
        return result

# ----------------------------------------------------------------------------
# Production API Predictor
# ----------------------------------------------------------------------------
class IntelliCloudPredictor:
    """Production-ready predictor for IntelliCloud energy efficiency."""

    def __init__(self):
        """Initialize with trained models."""
        print("Loading IntelliCloud models...")
        self.pipeline = TaskPipeline()
        self.rf_model = joblib.load(ROOT_DIR / "models" / "random_forest_energy_efficiency.pkl")
        self.rf_scaler = joblib.load(ROOT_DIR / "models" / "rf_scaler.pkl")
        
        # Load DQN Agent
        print("   → Loading DQN Scheduler...")
        self.dqn_agent = DQNAgent(state_dim=13, action_dim=3)
        dqn_path = ROOT_DIR / "models" / "dqn_scheduler.pth"
        if dqn_path.exists():
            self.dqn_agent.load(str(dqn_path))
            print("      ✓ DQN weights loaded")
        
        # Load DQN State Scaler
        scaler_p = ROOT_DIR / "models" / "dqn_state_scaler.pkl"
        if scaler_p.exists():
            self.dqn_state_scaler = joblib.load(str(scaler_p))
            print("      ✓ DQN state scaler loaded")
        else:
            self.dqn_state_scaler = None

        # Container naming
        self.containers = ["Tiny (0.25c/256MB)", "Medium (0.5c/512MB)", "Large (1c/1024MB)"]
        
        # Initialize SHAP explainer (caching it for performance)
        print("   → Initializing SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.rf_model)
        
        print("✓ All models loaded successfully")

    def predict_energy_efficiency(self, task_dict, include_shap=True):
        """
        Predict energy efficiency for a task.

        Args:
            task_dict: Dictionary with task information
            include_shap: Whether to generate a SHAP explanation image

        Returns:
            JSON response with all 12 features and prediction
        """
        try:
            # Extract 8 features and get latent features
            result_12 = self.pipeline.run(task_dict)
            if result_12 is None:
                return {
                    "error": "Feature extraction failed",
                    "task_info": task_dict
                }

            # Prepare 12 features for Random Forest
            features_12 = [
                result_12['input_size_mb'],
                result_12['cpu_usage_cores_absolute'],
                result_12['memory_usage_mb'],
                result_12['execution_time_normalized'],
                result_12['instruction_count'],
                result_12['network_io_mb'],
                result_12['power_consumption_watts'],
                0 if result_12['task_size_category'] == 'SMALL' else
                1 if result_12['task_size_category'] == 'MEDIUM' else 2,
                result_12['latent_f1'],
                result_12['latent_f2'],
                result_12['latent_f3'],
                result_12['latent_f4']
            ]

            # Random Forest prediction
            features_scaled = self.rf_scaler.transform(np.array(features_12).reshape(1, -1))
            prediction = self.rf_model.predict(features_scaled)[0]
            probabilities = self.rf_model.predict_proba(features_scaled)[0]

            # Efficiency levels mapping
            efficiency_levels = {
                1: "Very Low (0-20%)",
                2: "Low (20-40%)",
                3: "Medium (40-60%)",
                4: "High (60-80%)",
                5: "Very High (80-100%)"
            }

            # VM scheduling recommendations
            vm_recommendations = {
                1: "Schedule on high-performance VM (needs optimization)",
                2: "Schedule on standard VM with monitoring",
                3: "Schedule on balanced VM",
                4: "Schedule on efficient VM",
                5: "Schedule on eco-optimized VM (excellent efficiency)"
            }

            # --- DQN CONTAINER SELECTION ---
            dqn_action = None
            selected_container = "None"
            
            if self.dqn_agent and self.dqn_state_scaler:
                # Construct 13-dim state vector (12 SHERA features + RF class)
                state_raw = np.array(features_12 + [int(prediction)], dtype=np.float32).reshape(1, -1)
                state_scaled = self.dqn_state_scaler.transform(state_raw).flatten()
                
                # Extract raw Q-values 
                state_tensor = torch.FloatTensor(state_scaled).unsqueeze(0).to(self.dqn_agent.device)
                with torch.no_grad():
                    q_values = self.dqn_agent.policy_net(state_tensor).squeeze().cpu().numpy()
                
                # DQN Decides with natively learned Data-Driven policies!
                dqn_action = int(np.argmax(q_values))
                selected_container = self.containers[dqn_action]

            # --- GENERATE SHAP EXPLANATION ---
            shap_path = "skipped"
            if include_shap:
                try:
                    # Pass a unique identifier (task_type + index or similar)
                    task_id = f"{task_dict.get('task_type', 'task')}_size{task_dict.get('input_size_mb', '0')}"
                    shap_path = self.generate_shap_explanation(features_scaled, prediction, task_id)
                except Exception as shap_e:
                    print(f"   ⚠️  SHAP generation failed: {shap_e}")
                    shap_path = "failed"

            # Create complete JSON response
            response = {
                "task_info": task_dict,
                "features": {
                    "input_size_mb": float(result_12['input_size_mb']),
                    "cpu_usage_cores_absolute": float(result_12['cpu_usage_cores_absolute']),
                    "memory_usage_mb": float(result_12['memory_usage_mb']),
                    "execution_time_normalized": float(result_12['execution_time_normalized']),
                    "instruction_count": int(result_12['instruction_count']),
                    "network_io_mb": float(result_12['network_io_mb']),
                    "power_consumption_watts": float(result_12['power_consumption_watts']),
                    "task_size_category": result_12['task_size_category'],
                    "latent_f1": float(result_12['latent_f1']),
                    "latent_f2": float(result_12['latent_f2']),
                    "latent_f3": float(result_12['latent_f3']),
                    "latent_f4": float(result_12['latent_f4']),
                    "exact_match_found": result_12['exact_match_found'],
                    "is_scaled": result_12['is_scaled'],
                    "is_having_history": result_12['is_having_history']
                },
                "prediction": {
                    "energy_efficiency_class": int(prediction),
                    "efficiency_level": efficiency_levels[prediction],
                    "confidence": float(probabilities[prediction-1]),
                    "all_probabilities": {
                        "class_1": float(probabilities[0]),
                        "class_2": float(probabilities[1]),
                        "class_3": float(probabilities[2]),
                        "class_4": float(probabilities[3]),
                        "class_5": float(probabilities[4])
                    },
                    "explanation_image": shap_path
                },
                "vm_scheduling": {
                    "recommendation": vm_recommendations[prediction],
                    "selected_container": selected_container,
                    "dqn_action_id": dqn_action,
                    "priority": "high" if prediction <= 2 else "medium" if prediction == 3 else "low"
                },
                "processing_timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                "model_version": "SHERA-v1.0",
                "status": "success"
            }

            return response

        except Exception as e:
            return {
                "error": str(e),
                "task_info": task_dict,
                "status": "failed"
            }

    def generate_shap_explanation(self, features_scaled, prediction, task_id):
        """Generate a complete, self-explanatory SHAP report image."""
        shap_values = self.explainer.shap_values(features_scaled)
        
        probs = self.rf_model.predict_proba(features_scaled)[0]
        classes = self.rf_model.classes_
        sorted_idx = np.argsort(probs)[::-1]
        top_idx, second_idx = sorted_idx[0], (sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0])
        
        pred_class, second_class = classes[top_idx], classes[second_idx]
        shap_top, shap_second = shap_values[0, :, top_idx], shap_values[0, :, second_idx]
        
        feature_names = [
            'Input Size', 'CPU Focus', 'Memory', 'Time', 'Complexity', 'Network', 'Power', 'Size Class', 'AE-F1', 'AE-F2', 'AE-F3', 'AE-F4'
        ]
        
        # Focus on top features for clarity
        abs_top = np.abs(shap_top)
        top_indices = np.argsort(abs_top)[-10:]
        names_sorted = [feature_names[i] for i in top_indices]
        top_sorted = shap_top[top_indices]
        second_sorted = shap_second[top_indices]
        
        # 1. Create figure with extra width for the sidebar
        fig, ax = plt.subplots(figsize=(16, 10))
        y = np.arange(len(names_sorted))
        
        # Main Bars
        b1 = ax.barh(y + 0.18, top_sorted, 0.35, label=f'Impact on Class {pred_class} (Chosen)', color='#1abc9c', edgecolor='#16a085')
        b2 = ax.barh(y - 0.18, second_sorted, 0.35, label=f'Impact on Class {second_class} (Runner-up)', color='#e67e22', edgecolor='#d35400', alpha=0.5)
        
        # 2. Add value labels
        for rect in b1:
            width = rect.get_width()
            ax.text(width + (0.005 if width >= 0 else -0.005), rect.get_y() + rect.get_height()/2, 
                    f'{width:+.2f}', va='center', ha='left' if width >= 0 else 'right', fontsize=10, fontweight='bold', color='#16a085')

        ax.set_yticks(y)
        ax.set_yticklabels(names_sorted, fontsize=12, fontweight='bold')
        ax.grid(axis='x', linestyle=':', alpha=0.5)
        ax.axvline(0, color='black', linewidth=1.5)
        
        # 3. Headers and Labels
        ax.set_title(f'INTELLICLOUD EXPLANATION REPORT: {task_id.upper()}', fontsize=20, fontweight='bold', color='#2c3e50', pad=50)
        ax.set_xlabel('SHAP VALUE (Influence on AI Decision)\n[Positive = Pushes AI toward this Class | Negative = Pushes AI AWAY]', fontsize=12, labelpad=20)
        
        # 4. Comprehensive Sidebar (Guide for Beginners)
        guide_text = (
            "[?] HOW TO READ THIS REPORT\n"
            "------------------------------------\n"
            f"[*] CHOSEN: Class {pred_class} ({probs[top_idx]:.1%} confidence)\n"
            f"[-] RUNNER-UP: Class {second_class} ({probs[second_idx]:.1%} confidence)\n\n"
            "[INFO] WHAT ARE THE BARS?\n"
            "Each bar shows how much a specific feature\n"
            "(like CPU or Power) influenced the AI's choice.\n\n"
            "[+] POSITIVE (Bars to the RIGHT):\n"
            "This feature STRONGLY SUGGESTS the AI \n"
            "should choose this class.\n\n"
            "[-] NEGATIVE (Bars to the LEFT):\n"
            "This feature implies this class is less likely.\n\n"
            "[*] COLOR GUIDE:\n"
            "TEAL   = Support for the Chosen Class.\n"
            "ORANGE = Support for the Runner-up Class.\n\n"
            "[!] BIGGEST FACTOR:\n"
            f"The feature '{names_sorted[-1].upper()}' was the\n"
            "main reason for this classification."
        )
        
        # Using a text box on the right side
        plt.figtext(0.80, 0.5, guide_text, fontsize=11, family='monospace', 
                    bbox=dict(facecolor='#f7faff', alpha=1.0, boxstyle='round,pad=1.5', edgecolor='#3498db', linewidth=2),
                    va='center')

        # 5. Move Legend
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.1), ncol=2, fontsize=12, frameon=True, shadow=True)
        
        # Final layout adjustments
        plt.subplots_adjust(right=0.78, bottom=0.15, top=0.90)
        
        output_dir = Path("shap_explanations")
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_id = task_id.replace(' ', '_').replace('.', '_')
        img_path = output_dir / f"explanation_{safe_id}.png"
        
        plt.savefig(img_path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"   ✓ SHAP explanation report generated: {img_path}")
        return str(img_path)

# ----------------------------------------------------------------------------
# Global predictor instance for API usage
# ----------------------------------------------------------------------------
predictor = None

def get_predictor():
    """Get or create the global predictor instance."""
    global predictor
    if predictor is None:
        predictor = IntelliCloudPredictor()
    return predictor

def predict_task_energy(task_dict):
    """
    Convenience function for predicting energy efficiency.

    Args:
        task_dict: Task information dictionary

    Returns:
        JSON response with all 12 features and prediction
    """
    pred = get_predictor()
    return pred.predict_energy_efficiency(task_dict)

# ----------------------------------------------------------------------------
# Demo and Testing Functionality
# ----------------------------------------------------------------------------
def demonstrate_complete_pipeline(tasks=None):
    """
    Demonstrate the complete IntelliCloud pipeline workflow.
    """
    print("=" * 80)
    print("INTELLICLOUD PIPELINE DEMO - SHERA METHODOLOGY")
    print("=" * 80)

    # 1. Initialize Predictor (which handles features, RF, and SHAP)
    print("\n1. Initializing IntelliCloud Predictor...")
    predictor = get_predictor()
    print("   ✓ All models (Encoder, RF, SHAP) ready")

    # 3. Example incoming tasks (now passed as argument)
    if tasks is None:
        print("   ⚠️  No tasks provided for demonstration.")
        return

    print("\n3. Processing Incoming Tasks...")
    print("-" * 50)

    for i, task in enumerate(tasks, 1):
        print(f"\n📋 Task {i}: {task['task_type']} ({task['input_size_mb']} MB)")

        # Full Prediction (including SHAP)
        print("   → Running full SHERA Pipeline...")
        prediction_result = predictor.predict_energy_efficiency(task)
        
        if "error" in prediction_result:
            print(f"   ⚠️  Prediction failed: {prediction_result['error']}")
            continue

        prediction = prediction_result['prediction']['energy_efficiency_class']
        efficiency_level = prediction_result['prediction']['efficiency_level']
        confidence = prediction_result['prediction']['confidence']
        
        print(f"   🎯 Energy Efficiency Prediction: Class {prediction}")
        print(f"      Level: {efficiency_level}")
        print(f"      Confidence: {confidence:.3f}")
        
        # New Output: Selected Container
        selected_vm = prediction_result['vm_scheduling']['selected_container']
        print(f"      🚀 Selected Container: {selected_vm}")

        print("\n  ── 8 Extracted Features ──────────────────────────")
        f = prediction_result['features']
        print(f"    input_size_mb             : {f['input_size_mb']}")
        print(f"    cpu_usage_cores_absolute  : {f['cpu_usage_cores_absolute']}")
        print(f"    memory_usage_mb           : {f['memory_usage_mb']}")
        print(f"    execution_time_normalized : {f['execution_time_normalized']}")
        print(f"    instruction_count         : {f['instruction_count']:,}")
        print(f"    network_io_mb             : {f['network_io_mb']}")
        print(f"    power_consumption_watts   : {f['power_consumption_watts']}")
        print(f"    task_size_category        : {f['task_size_category']}")
        print(f"    exact_match_found         : {f['exact_match_found']} {'✅' if f['exact_match_found'] else '⚠️ (estimated)'}")
        print(f"    is_scaled                 : {f['is_scaled']} {'🔧' if f['is_scaled'] else '📏'}")
        print(f"    is_having_history         : {f['is_having_history']} {'📚' if f['is_having_history'] else '❓'}")

        print("\n  ── 4 Latent Features (Encoder Output) ────────────")
        print(f"    latent_f1 : {f['latent_f1']}")
        print(f"    latent_f2 : {f['latent_f2']}")
        print(f"    latent_f3 : {f['latent_f3']}")
        print(f"    latent_f4 : {f['latent_f4']}")

        print(f"   📄 Complete JSON Result:")
        print(json.dumps(prediction_result, indent=4))

    print("\n" + "=" * 80)
    print("PIPELINE DEMO COMPLETE")
    print("=" * 80)
    print("\n✅ Workflow: Task → 8 Features → Autoencoder → 12 Features → RF → Efficiency")
    print("✅ Implements SHERA methodology for cloud energy optimization")
    print("✅ Ready for production VM scheduling decisions!")

# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Load tasks from tasks.json
    tasks_file = "tasks.json"
    try:
        with open(tasks_file, 'r') as f:
            tasks_data = json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading {tasks_file}: {e}")
        tasks_data = []

    # Run demo with tasks from JSON
    demonstrate_complete_pipeline(tasks_data)