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
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

import torch
import torch.nn as nn

import joblib

from feature_extractor import TaskFeatureExtractor

# ----------------------------------------------------------------------------
# Encoder architecture (must match the trained model)
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    """
    Architecture: 8 → 64 → 4
    Mirrors src/ml_models/encoder.py so we don't need a cross-folder import.
    """
    def __init__(self, input_dim=8, latent_dim=4, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

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
        root = Path(__file__).resolve().parent  # → /Users/harshareddy/Desktop/intellicloud

        self.model_path   = Path(model_path) if model_path else root / "models" / "autoencoder_encoder.pth"
        history_path      = history_path or str(Path(__file__).parent / "dataset" / "task_profiles_clean_final.json")

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
        self.rf_model = joblib.load("models/random_forest_energy_efficiency.pkl")
        self.rf_scaler = joblib.load("models/rf_scaler.pkl")
        print("✓ All models loaded successfully")

    def predict_energy_efficiency(self, task_dict):
        """
        Predict energy efficiency for a task.

        Args:
            task_dict: Dictionary with task information

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
                    "exact_match_found": result_12['exact_match_found'],  # Boolean flag for exact match vs estimation
                    "is_scaled": result_12['is_scaled'],                   # Boolean flag for scaling usage
                    "is_having_history": result_12['is_having_history']   # Boolean flag for history availability
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
                    }
                },
                "vm_scheduling": {
                    "recommendation": vm_recommendations[prediction],
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
def demonstrate_complete_pipeline():
    """
    Demonstrate the complete IntelliCloud pipeline workflow.
    """
    print("=" * 80)
    print("INTELLICLOUD PIPELINE DEMO - SHERA METHODOLOGY")
    print("=" * 80)

    # 1. Initialize Pipeline (loads feature extractor, scaler, and trained autoencoder)
    print("\n1. Initializing Pipeline...")
    pipeline = TaskPipeline()
    print("   ✓ Pipeline ready with trained autoencoder")

    # 2. Load Random Forest model
    print("\n2. Loading Random Forest model...")
    rf_model = joblib.load("models/random_forest_energy_efficiency.pkl")
    rf_scaler = joblib.load("models/rf_scaler.pkl")
    print("   ✓ Random Forest loaded (trained on 12 features)")

    # 3. Example incoming tasks
    incoming_tasks = [
        {
            "task_type": "matrix_multiplication",
            "input_size_mb": 4.0,  # Unusual size that won't match exactly
            "complexity": "high",
            "priority": "critical",
            "application": "scientific_computing"
        }
    ]

    print("\n3. Processing Incoming Tasks...")
    print("-" * 50)

    for i, task in enumerate(incoming_tasks, 1):
        print(f"\n📋 Task {i}: {task['task_type']} ({task['input_size_mb']} MB)")

        # Step 1: Extract 8 features using historical data
        print("   → Extracting 8 features from historical profiles...")
        features_8 = pipeline.extractor.extract_features(task)

        if features_8 is None:
            print("   ⚠️  Feature extraction failed (invalid input_size_mb)")
            continue

        print("   ✓ 8 features extracted:")
        cpu_val = features_8['cpu_usage_cores_absolute']
        mem_val = features_8['memory_usage_mb']
        power_val = features_8['power_consumption_watts']
        print(f"     CPU: {cpu_val:.3f} cores")
        print(f"     Memory: {mem_val:.1f} MB")
        print(f"     Power: {power_val:.1f} W")

        # Step 2: Get 4 latent features from autoencoder
        print("   → Generating 4 latent features via autoencoder...")
        result_12 = pipeline.run(task)

        if result_12 is None:
            print("   ⚠️  Pipeline processing failed")
            continue

        print("   ✓ 4 latent features generated:")
        f1 = result_12['latent_f1']
        f2 = result_12['latent_f2']
        f3 = result_12['latent_f3']
        f4 = result_12['latent_f4']
        print(f"     Latent F1: {f1:.6f}")
        print(f"     Latent F2: {f2:.6f}")
        print(f"     Latent F3: {f3:.6f}")
        print(f"     Latent F4: {f4:.6f}")

        # Step 3: Prepare 12 features for Random Forest
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

        # Step 4: Random Forest prediction
        print("   → Predicting energy efficiency with Random Forest...")
        features_scaled = rf_scaler.transform(np.array(features_12).reshape(1, -1))
        prediction = rf_model.predict(features_scaled)[0]
        probabilities = rf_model.predict_proba(features_scaled)[0]

        efficiency_levels = {
            1: "Very Low (0-20%)",
            2: "Low (20-40%)",
            3: "Medium (40-60%)",
            4: "High (60-80%)",
            5: "Very High (80-100%)"
        }

        print(f"   🎯 Energy Efficiency Prediction: Class {prediction}")
        print(f"      Level: {efficiency_levels[prediction]}")
        print(f"      Confidence: {probabilities[prediction-1]:.3f}")

        # Step 5: VM Scheduling Recommendation (based on efficiency)
        vm_recommendations = {
            1: "Schedule on high-performance VM (needs optimization)",
            2: "Schedule on standard VM with monitoring",
            3: "Schedule on balanced VM",
            4: "Schedule on efficient VM",
            5: "Schedule on eco-optimized VM (excellent efficiency)"
        }

        # Step 5: Create complete JSON output
        prediction_result = {
            "task_info": task,
            "features": {
                "input_size_mb": result_12['input_size_mb'],
                "cpu_usage_cores_absolute": result_12['cpu_usage_cores_absolute'],
                "memory_usage_mb": result_12['memory_usage_mb'],
                "execution_time_normalized": result_12['execution_time_normalized'],
                "instruction_count": result_12['instruction_count'],
                "network_io_mb": result_12['network_io_mb'],
                "power_consumption_watts": result_12['power_consumption_watts'],
                "task_size_category": result_12['task_size_category'],
                "latent_f1": result_12['latent_f1'],
                "latent_f2": result_12['latent_f2'],
                "latent_f3": result_12['latent_f3'],
                "latent_f4": result_12['latent_f4'],
                "exact_match_found": result_12['exact_match_found'],  # NEW: Boolean flag
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
                }
            },
            "vm_scheduling": {
                "recommendation": vm_recommendations[prediction],
                "priority": "high" if prediction <= 2 else "medium" if prediction == 3 else "low"
            },
            "processing_timestamp": "2025-02-23T22:55:00Z"  # Would be dynamic in production
        }

        print("\n  ── 8 Extracted Features ──────────────────────────")
        print(f"    input_size_mb             : {result_12['input_size_mb']}")
        print(f"    cpu_usage_cores_absolute  : {result_12['cpu_usage_cores_absolute']}")
        print(f"    memory_usage_mb           : {result_12['memory_usage_mb']}")
        print(f"    execution_time_normalized : {result_12['execution_time_normalized']}")
        print(f"    instruction_count         : {result_12['instruction_count']:,}")
        print(f"    network_io_mb             : {result_12['network_io_mb']}")
        print(f"    power_consumption_watts   : {result_12['power_consumption_watts']}")
        print(f"    task_size_category        : {result_12['task_size_category']}")
        print(f"    exact_match_found         : {result_12['exact_match_found']} {'✅' if result_12['exact_match_found'] else '⚠️ (estimated)'}")
        print(f"    is_scaled                 : {result_12['is_scaled']} {'🔧' if result_12['is_scaled'] else '📏'}")
        print(f"    is_having_history         : {result_12['is_having_history']} {'📚' if result_12['is_having_history'] else '❓'}")

        print("\n  ── 4 Latent Features (Encoder Output) ────────────")
        print(f"    latent_f1 : {result_12['latent_f1']}")
        print(f"    latent_f2 : {result_12['latent_f2']}")
        print(f"    latent_f3 : {result_12['latent_f3']}")
        print(f"    latent_f4 : {result_12['latent_f4']}")

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
    # Run demo by default
    demonstrate_complete_pipeline()

    # Example of API usage (uncomment to test)
    print("\n" + "="*60)
    print("API USAGE EXAMPLE:")
    print("="*60)

    test_task = {
        "task_type": "matrix_multiplication",
        "input_size_mb": 50.0,
        "complexity": "high",
        "priority": "critical",
        "application": "scientific_computing"
    }

    result = predict_task_energy(test_task)
    print("API Result:")
    print(json.dumps(result, indent=2))
