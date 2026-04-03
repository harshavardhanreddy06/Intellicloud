#!/usr/bin/env python3
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

def clean_and_finalize_dataset():
    input_file = Path('data/task_profiles_vm_independent.json')
    output_file = Path('data/task_profiles_clean_final.json')
    
    if not input_file.exists():
        print(f"❌ Input file {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        all_records = json.load(f)

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
            base_record[field] = get_mode(vals)

        # Set Keys
        base_record['task_signature'] = sig
        base_record['task_size_category'] = size_cat

        # --- STEP 2: Memory Sanity Rule ---
        # memory_usage_mb >= input_size_mb * 0.02
        if base_record['memory_usage_mb'] < base_record['input_size_mb'] * 0.02:
            print(f"⚠️ Memory sanity violation for {sig} ({size_cat}). Fixing with median.")
            mem_vals = [m.get('memory_usage_mb', 0) for m in members]
            base_record['memory_usage_mb'] = statistics.median(mem_vals)
            # Re-check sanity - if still fails (e.g. all were low), use a floor
            if base_record['memory_usage_mb'] < base_record['input_size_mb'] * 0.02:
                 base_record['memory_usage_mb'] = base_record['input_size_mb'] * 0.05 # Conservative floor

        aggregated_profiles.append(base_record)

    # --- STEP 3: Scaling Consistency ---
    # Group by task_type to compare sizes
    type_groups = defaultdict(list)
    for p in aggregated_profiles:
        type_groups[p['task_type']].append(p)

    scale_order = {'SMALL': 1, 'MEDIUM': 2, 'LARGE': 3}
    
    for t_type, profiles in type_groups.items():
        # Sort by size category for comparison
        sorted_profiles = sorted([p for p in profiles if p['task_size_category'] in scale_order], 
                                key=lambda x: scale_order[x['task_size_category']])
        
        check_fields = ['execution_time_normalized', 'instruction_count', 'cpu_usage_cores_absolute']
        
        for i in range(len(sorted_profiles) - 1):
            smaller = sorted_profiles[i]
            larger = sorted_profiles[i+1]
            
            for field in check_fields:
                s_val = smaller[field]
                l_val = larger[field]
                
                # Rule: Smaller > Larger * 2 is an anomaly
                if s_val > (l_val * 2.0) and l_val > 0:
                    print(f"❗ Scaling anomaly: {t_type} {smaller['task_size_category']} {field} ({s_val}) is > 2x {larger['task_size_category']} ({l_val})")
                    # Replace with group mean for that task type/field
                    all_vals_for_type = [p[field] for p in profiles]
                    shared_mean = sum(all_vals_for_type) / len(all_vals_for_type)
                    smaller[field] = shared_mean

    # --- FINAL LOG RECOMPUTATION ---
    for p in aggregated_profiles:
        p['input_size_log'] = round(math.log(p['input_size_mb'] + 1), 4)
        p['execution_time_log'] = round(math.log(p['execution_time_normalized'] + 1), 4)
        p['instruction_count_log'] = round(math.log(p['instruction_count'] + 1), 4)
        
        # Round other numerical values for cleanliness
        p['input_size_mb'] = round(p['input_size_mb'], 2)
        p['cpu_usage_cores_absolute'] = round(p['cpu_usage_cores_absolute'], 4)
        p['memory_usage_mb'] = round(p['memory_usage_mb'], 2)
        p['execution_time_normalized'] = round(p['execution_time_normalized'], 4)
        p['instruction_count'] = int(p['instruction_count'])
        p['network_io_mb'] = round(p['network_io_mb'], 4)
        p['power_consumption_watts'] = round(p['power_consumption_watts'], 2)

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(aggregated_profiles, f, indent=2)

    print(f"\n✨ Cleanup complete.")
    print(f"✅ Final Dataset Created: {output_file}")
    print(f"📊 Unique Task Profiles: {len(aggregated_profiles)}")

if __name__ == '__main__':
    clean_and_finalize_dataset()
