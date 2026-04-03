#!/usr/bin/env python3
import json
import math
from pathlib import Path
from collections import defaultdict

def create_independent_dataset():
    input_dir = Path('data/profiles_postprocessed')
    output_file = Path('data/task_profiles_vm_independent.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = [
        'tiny_vm_profiles.json',
        'small_vm_profiles.json',
        'medium_vm_profiles.json'
    ]

    all_records = []
    for f_name in files:
        f_path = input_dir / f_name
        if f_path.exists():
            with open(f_path, 'r') as f:
                all_records.extend(json.load(f))
        else:
            print(f"⚠️ Warning: {f_path} not found.")

    if not all_records:
        print("❌ No records found to process.")
        return

    # Grouping and Aggregation
    # Group key: (task_signature, task_type, task_size_category, complexity)
    groups = defaultdict(list)
    for rec in all_records:
        key = (
            rec.get('task_signature'),
            rec.get('task_type'),
            rec.get('task_size_category'),
            rec.get('complexity')
        )
        groups[key].append(rec)

    independent_profiles = []

    print(f"⚙️ Aggregating {len(all_records)} records into unique task profiles...")

    for key, members in groups.items():
        # Baseline task info from the first member
        base = members[0]
        
        # Aggregation
        count = len(members)
        avg_input_size = sum(m.get('input_size_mb', 0) for m in members) / count
        avg_cpu_abs = sum(m.get('cpu_usage_cores_absolute', 0) for m in members) / count
        avg_memory = sum(m.get('memory_usage_mb', 0) for m in members) / count
        avg_instructions = sum(m.get('instruction_count', 0) for m in members) / count
        avg_network = sum(m.get('network_io_mb', 0) for m in members) / count
        avg_power = sum(m.get('power_consumption_watts', 0) for m in members) / count
        
        # Normalization of execution time to 1.0 core reference
        # Formula: mean(time * vm_cpu_cores)
        # This represents the "core-seconds" consumed, which is the time if it had 1.0 core.
        norm_times = []
        for m in members:
            time_raw = m.get('execution_time_sec', 0)
            cores = m.get('vm_cpu_cores', 1.0)
            # Normalized to 1.0 core: if it takes 4s on 0.25 core, it takes 1s on 1.0 core.
            norm_times.append(time_raw * cores)
        
        avg_norm_time = sum(norm_times) / count

        # Log factors
        input_size_log = math.log(avg_input_size + 1)
        instruction_count_log = math.log(avg_instructions + 1)
        execution_time_log = math.log(avg_norm_time + 1)

        profile = {
            "task_signature": base.get('task_signature'),
            "task_type": base.get('task_type'),
            "task_category": base.get('task_category', 'real'),
            "application": base.get('application'),
            "priority": base.get('priority'),
            "complexity": base.get('complexity'),
            "task_size_category": base.get('task_size_category'),
            
            "input_size_mb": round(avg_input_size, 2),
            "input_size_log": round(input_size_log, 4),
            "cpu_usage_cores_absolute": round(avg_cpu_abs, 4),
            "memory_usage_mb": round(avg_memory, 2),
            "execution_time_normalized": round(avg_norm_time, 4),
            "execution_time_log": round(execution_time_log, 4),
            "instruction_count": int(avg_instructions),
            "instruction_count_log": round(instruction_count_log, 4),
            "network_io_mb": round(avg_network, 4),
            "power_consumption_watts": round(avg_power, 2)
        }
        
        independent_profiles.append(profile)

    # Save output
    with open(output_file, 'w') as f:
        json.dump(independent_profiles, f, indent=2)

    print(f"✅ Successfully created VM-independent dataset with {len(independent_profiles)} unique tasks.")
    print(f"📂 Location: {output_file}")

if __name__ == '__main__':
    create_independent_dataset()
