import json
import math
import statistics
from pathlib import Path
from collections import defaultdict, Counter

def get_mode(items):
    if not items:
        return None
    counts = Counter(items)
    return counts.most_common(1)[0][0]

def finalize_dataset():
    # Input files (the 10k profiles)
    files = {
        'medium': '/Users/harshareddy/Desktop/intellicloud/dataset/medium_vm_profiles.json',
        'small': '/Users/harshareddy/Desktop/intellicloud/dataset/small_vm_profiles.json',
        'tiny': '/Users/harshareddy/Desktop/intellicloud/dataset/tiny_vm_profiles.json'
    }
    
    # Scaling factors used during generation
    time_factors = {
        'medium': 1.0,
        'small': 1.8,
        'tiny': 3.5
    }

    all_records = []
    
    # Load and normalize
    for tier, path in files.items():
        if not Path(path).exists():
            print(f"⚠️ {path} not found.")
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            for rec in data:
                # Normalize execution time back to medium-equivalent
                rec['execution_time_normalized'] = rec['execution_time_sec'] / time_factors[tier]
                all_records.append(rec)

    print(f"📖 Loaded {len(all_records)} task records.")

    # --- STEP 1: Aggregation by (task_signature, task_size_category) ---
    groups = defaultdict(list)
    for rec in all_records:
        key = (rec.get('task_signature'), rec.get('task_size_category'))
        groups[key].append(rec)

    aggregated_profiles = []

    for (sig, size_cat), members in groups.items():
        base_record = {}
        count = len(members)

        # Numerical Fields (Mean)
        num_fields = [
            'input_size_mb', 'cpu_usage_cores_absolute', 'memory_usage_mb',
            'execution_time_normalized', 'instruction_count', 
            'network_io_mb', 'power_consumption_watts'
        ]
        
        for field in num_fields:
            vals = [m.get(field, 0) for m in members]
            base_record[field] = sum(vals) / count

        # Categorical Fields (Mode)
        cat_fields = ['task_type', 'task_category', 'priority', 'complexity', 'application']
        for field in cat_fields:
            vals = [m.get(field) for m in members if m.get(field)]
            base_record[field] = get_mode(vals) if vals else 'unknown'

        # Set Keys
        base_record['task_signature'] = sig
        base_record['task_size_category'] = size_cat

        # Log Transformations
        base_record['input_size_log'] = round(math.log(base_record['input_size_mb'] + 1), 4)
        base_record['execution_time_log'] = round(math.log(base_record['execution_time_normalized'] + 1), 4)
        base_record['instruction_count_log'] = round(math.log(base_record['instruction_count'] + 1), 4)
        
        # Rounding for cleanliness
        base_record['input_size_mb'] = round(base_record['input_size_mb'], 2)
        base_record['cpu_usage_cores_absolute'] = round(base_record['cpu_usage_cores_absolute'], 4)
        base_record['memory_usage_mb'] = round(base_record['memory_usage_mb'], 2)
        base_record['execution_time_normalized'] = round(base_record['execution_time_normalized'], 4)
        base_record['instruction_count'] = int(base_record['instruction_count'])
        base_record['network_io_mb'] = round(base_record['network_io_mb'], 4)
        base_record['power_consumption_watts'] = round(base_record['power_consumption_watts'], 2)

        aggregated_profiles.append(base_record)

    # Save to file
    output_file = '/Users/harshareddy/Desktop/intellicloud/dataset/task_profiles_clean_final.json'
    with open(output_file, 'w') as f:
        json.dump(aggregated_profiles, f, indent=2)

    print(f"\n✨ Cleanup complete.")
    print(f"✅ Final Dataset Updated: {output_file}")
    print(f"📊 Unique Task Profiles: {len(aggregated_profiles)}")

if __name__ == '__main__':
    finalize_dataset()
