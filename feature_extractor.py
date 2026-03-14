import json
import numpy as np
import math
from pathlib import Path
from sklearn.linear_model import LinearRegression

class TaskFeatureExtractor:
    def __init__(self, history_path='data/task_profiles_clean_final.json'):
        # Ensure path is absolute relative to project root (one level up from this script)
        path_obj = Path(history_path)
        if not path_obj.is_absolute():
            # This script is in /featuresextraction/, data is in /data/
            # So go up one level and then into the data folder
            root_dir = Path(__file__).parent.parent
            self.history_path = root_dir / history_path
        else:
            self.history_path = path_obj
            
        self.history_data = self._load_history()
        
    def _load_history(self):
        if not self.history_path.exists():
            print(f"⚠️ Warning: History file {self.history_path} not found.")
            return []
        with open(self.history_path, 'r') as f:
            return json.load(f)

    def _get_size_category(self, size_mb):
        if size_mb <= 20:
            return "SMALL"
        elif size_mb <= 60:
            return "MEDIUM"
        else:
            return "LARGE"

    def _infer_task_category(self, task_type):
        """Maps task types to broad categories like Compute, Analysis, I/O, etc."""
        mapping = {
            'matrix_multiplication': 'compute',
            'monte_carlo_simulation': 'compute',
            'statistical_analysis': 'compute',
            'csv_correlation_analysis': 'analysis',
            'log_parsing': 'analysis',
            'log_pattern_matching': 'analysis',
            'csv_aggregation': 'analysis',
            'csv_groupby': 'analysis',
            'csv_merge_operations': 'io_heavy',
            'data_deduplication': 'io_heavy',
            'log_aggregation': 'io_heavy',
            'text_search_replace': 'io_heavy',
            'text_tokenization': 'io_heavy',
            'text_word_count': 'io_heavy',
            'image_compression': 'media',
            'image_filter_application': 'media',
            'image_resize': 'media',
            'thumbnail_generation': 'media'
        }
        return mapping.get(task_type, 'general')

    def extract_features(self, incoming_task):
        """
        Extract features for an incoming task based on historical performance data.
        Corrected and stabilized version.
        """
        task_type = incoming_task.get('task_type')
        input_size = float(incoming_task.get('input_size_mb', 0))
        complexity = incoming_task.get('complexity', 'medium').lower()
        priority = incoming_task.get('priority', 'medium').lower()
        application = incoming_task.get('application', 'unknown')
        
        # Determine category from input or infer it
        task_category = incoming_task.get('task_category') or self._infer_task_category(task_type)

        if input_size <= 0:
            return None

        # STEP 1: Filter History
        history_relevant = [r for r in self.history_data if r.get('task_type') == task_type]
        
        # Initialize estimation flag
        exact_match_found = True  # Assume exact match initially
        is_having_history = True   # Assume we have some history initially
        is_scaled = False          # Assume no scaling initially
        
        # --- Handle Unknown Task Types ---
        if not history_relevant:
            # Fallback: Filter by category if task_type is unknown
            history_relevant = [r for r in self.history_data if r.get('task_category') == task_category]
            
            if history_relevant:
                # Found category-based history
                is_having_history = True
            else:
                # If still nothing (extreme unknown), use the entire global dataset
                history_relevant = self.history_data
                is_having_history = False  # No specific history found
            
            # Flag that this is a synthetic estimate based on peers
            exact_match_found = False  # Using fallback estimation

        # STEP 2: Check for Same Size Match (within 5% tolerance)
        exact_matches = [
            r for r in history_relevant 
            if abs(r.get('input_size_mb', 0) - input_size) <= (0.05 * input_size)
        ]

        if exact_matches:
            # CASE A: Use median of exact matches
            extracted = self._aggregate_matches(exact_matches)
            is_scaled = False  # Exact matches found, no scaling needed
        else:
            # CASE B: Scaling Required
            extracted = self._scale_from_history(history_relevant, input_size)
            exact_match_found = False  # Scaling means estimation, not exact match
            is_scaled = True  # Scaling was required due to size mismatch

        # STEP 4: Attribute Adjustments (Apply after scaling)
        extracted = self._apply_adjustments(extracted, complexity, priority)

        # STEP 3 Revision: Core Power Consumption Logic
        # Power = 40 + 40 * cpu_usage_cores_absolute
        extracted['power_consumption_watts'] = 40.0 + (40.0 * extracted['cpu_usage_cores_absolute'])

        # STEP 5: Consistency Constraints
        extracted = self._enforce_constraints(extracted, input_size, history_relevant)

        # STEP 6: Final Output Format
        final_output = {
            "input_size_mb": round(input_size, 2),
            "cpu_usage_cores_absolute": round(extracted['cpu_usage_cores_absolute'], 4),
            "memory_usage_mb": round(extracted['memory_usage_mb'], 2),
            "execution_time_normalized": round(extracted['execution_time_normalized'], 4),
            "instruction_count": int(extracted['instruction_count']),
            "network_io_mb": round(extracted['network_io_mb'], 4),
            "power_consumption_watts": round(extracted['power_consumption_watts'], 2),
            "task_type": task_type,
            "task_category": task_category,
            "priority": priority,
            "complexity": complexity,
            "application": application,
            "task_size_category": self._get_size_category(input_size),
            "input_size_log": round(math.log(input_size), 4),
            "execution_time_log": round(math.log(extracted['execution_time_normalized']), 4),
            "instruction_count_log": round(math.log(extracted['instruction_count']), 4),
            "exact_match_found": exact_match_found,  # Boolean flag for exact match vs estimation
            "is_scaled": is_scaled,                    # Boolean flag for scaling usage
            "is_having_history": is_having_history    # Boolean flag for history availability
        }
        
        return final_output

    def _aggregate_matches(self, matches):
        """Returns median metrics from matches to avoid noise."""
        metrics = [
            'cpu_usage_cores_absolute', 'memory_usage_mb', 
            'execution_time_normalized', 'instruction_count', 
            'network_io_mb'
        ]
        result = {}
        for m in metrics:
            vals = [match.get(m, 0) for match in matches if match.get(m, 0) > 0]
            result[m] = float(np.median(vals)) if vals else 0.1
        return result

    def _scale_from_history(self, history, target_size):
        """Learns scaling exponents and estimates metrics using log-log regression."""
        # Find closest reference task by input_size
        closest_ref = min(history, key=lambda x: abs(x.get('input_size_mb', 0) - target_size))
        ref_size = float(closest_ref.get('input_size_mb', 1))
        size_ratio = target_size / ref_size
        
        result = {}
        
        # Scaling Metrics
        scaling_metrics = [
            'memory_usage_mb', 'execution_time_normalized', 'instruction_count'
        ]
        
        # 1. Regression Consistency Fix: Only use records where size and metric > 0
        for metric in scaling_metrics:
            valid_stats = [
                (math.log(h['input_size_mb']), math.log(h[metric]))
                for h in history 
                if h.get('input_size_mb', 0) > 0 and h.get(metric, 0) > 0
            ]
            
            if len(valid_stats) > 1:
                X = np.array([s[0] for s in valid_stats]).reshape(-1, 1)
                y = np.array([s[1] for s in valid_stats]).reshape(-1, 1)
                model = LinearRegression().fit(X, y)
                scaling_exp = float(model.coef_[0][0])
            else:
                scaling_exp = 1.0 # Default to linear
                
            ref_val = float(closest_ref.get(metric, 0.1))
            result[metric] = ref_val * (size_ratio ** scaling_exp)
            
            # 2. CPU Scaling Correction: Sub-linear growth (approx. 0.3 * exp)
            if metric == 'execution_time_normalized':
                ref_cpu = float(closest_ref.get('cpu_usage_cores_absolute', 0.5))
                result['cpu_usage_cores_absolute'] = ref_cpu * (size_ratio ** (0.3 * scaling_exp))

        # Network IO: Use median historical value
        net_vals = [h.get('network_io_mb', 0) for h in history]
        result['network_io_mb'] = float(np.median(net_vals)) if net_vals else 0.0
        
        return result

    def _apply_adjustments(self, metrics, complexity, priority):
        """Applies complexity and priority factors after scaling."""
        # Complexity factor
        comp_map = {'low': 0.9, 'medium': 1.0, 'high': 1.15}
        c_factor = comp_map.get(complexity, 1.0)
        
        metrics['execution_time_normalized'] *= c_factor
        metrics['instruction_count'] *= c_factor
        
        # Priority factor (applied only to execution time)
        prio_map = {'low': 1.05, 'medium': 1.0, 'high': 0.95, 'critical': 0.9}
        p_factor = prio_map.get(priority, 1.0)
        
        metrics['execution_time_normalized'] *= p_factor
        
        return metrics

    def _enforce_constraints(self, metrics, input_size, history):
        """Ensures physical bounds and monotonicity constraints."""
        # CPU constrained between [0.1, 1.0]
        metrics['cpu_usage_cores_absolute'] = max(0.1, min(1.0, metrics['cpu_usage_cores_absolute']))
        
        # Memory Sanity: Proportion of input size
        metrics['memory_usage_mb'] = max(input_size * 0.02, metrics['memory_usage_mb'])
        
        # Monotonicity Fix: Ensure larger tasks don't have smaller footprints than ref
        closest_ref = min(history, key=lambda x: abs(x.get('input_size_mb', 0) - input_size))
        ref_size = closest_ref['input_size_mb']
        
        for field in ['execution_time_normalized', 'instruction_count']:
            ref_val = closest_ref.get(field, 0)
            if input_size > ref_size and metrics[field] < ref_val:
                metrics[field] = ref_val * (input_size / ref_size)
            elif input_size < ref_size and metrics[field] > ref_val:
                metrics[field] = ref_val * (input_size / ref_size)

        # No negative values
        for k in metrics:
            if isinstance(metrics[k], (int, float)):
                metrics[k] = max(1e-6, metrics[k])
                
        return metrics

if __name__ == '__main__':
    # Test script
    extractor = TaskFeatureExtractor()
    test_task = {
     "task_type": "deep_learning_inference_v2", 
        "task_category": "compute", # User provides category
        "input_size_mb": 120.0,
        "priority": "critical",
        "complexity": "high",
        "application": "cloud_storage"
    }
    
    
    features = extractor.extract_features(test_task)
    if features:
        print("✅ Corrected Features for Incoming Task:")
        print(json.dumps(features, indent=2))
