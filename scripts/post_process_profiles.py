#!/usr/bin/env python3
"""
Post-processing of task profiles to normalize CPU metrics.
Converts relative VM saturation into absolute compute demand.
"""

import json
import os
from pathlib import Path

def post_process():
    input_file = Path('data/real_profiles/real_task_profiles_3k.json')
    output_dir = Path('data/profiles_postprocessed')
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"❌ Input file {input_file} not found.")
        return

    print(f"📖 Reading profiles from {input_file}...")
    with open(input_file, 'r') as f:
        try:
            profiles = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ JSON decode error. The file might still be being written to or is truncated.")
            # Try to fix truncated JSON if needed, but let's assume it's valid for now
            return

    processed_count = 0
    tiny_profiles = []
    small_profiles = []
    medium_profiles = []

    print(f"⚙️ Processing {len(profiles)} profiles...")

    for profile in profiles:
        # 1. Original field
        rel_cpu = profile.get('cpu_usage_percent', 0.0)
        vm_cores = profile.get('vm_cpu_cores', 1.0)
        vm_tier = profile.get('vm_tier', 'unknown').lower()

        # 2. Add cpu_usage_percent_relative
        profile['cpu_usage_percent_relative'] = rel_cpu

        # 3. Add cpu_usage_cores_absolute
        # Formula: (Relative % / 100) * VM Cores
        abs_cores = (rel_cpu / 100.0) * vm_cores
        profile['cpu_usage_cores_absolute'] = round(abs_cores, 4)

        # 4. Add cpu_usage_percent_absolute
        # Formula: Absolute Cores * 100 (This is the % of one full physical core)
        profile['cpu_usage_percent_absolute'] = round(abs_cores * 100.0, 2)

        # Categorize by tier
        if 'tiny' in vm_tier:
            tiny_profiles.append(profile)
        elif 'small' in vm_tier:
            small_profiles.append(profile)
        elif 'medium' in vm_tier:
            medium_profiles.append(profile)
        
        processed_count += 1

    # Save to respective files
    files_map = {
        'tiny_vm_profiles.json': tiny_profiles,
        'small_vm_profiles.json': small_profiles,
        'medium_vm_profiles.json': medium_profiles
    }

    for filename, data in files_map.items():
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {len(data)} profiles to {output_path}")

    print(f"\n✨ Post-processing complete. Total processed: {processed_count}")

if __name__ == '__main__':
    post_process()
