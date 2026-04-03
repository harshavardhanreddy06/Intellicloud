import json
import random
import os
import numpy as np
from datetime import datetime, timedelta

def scale_multi_modal_profiles():
    files = {
        'medium': '/Users/harshareddy/Desktop/intellicloud/dataset/medium_vm_profiles.json',
        'small': '/Users/harshareddy/Desktop/intellicloud/dataset/small_vm_profiles.json',
        'tiny': '/Users/harshareddy/Desktop/intellicloud/dataset/tiny_vm_profiles.json'
    }

    print("🚀 Generating 10k Multi-Modal Profiles for Multi-Tier Clouds (30MB Cap)...")
    
    # ── Task Definitions (30+ Categories) ──────────────────────────────────
    tasks_config = {
        # IMAGE (10)
        'img_resize': {'cat': 'image', 'cpu': 30, 'mem': 1.8},
        'img_cropping': {'cat': 'image', 'cpu': 25, 'mem': 2.0},
        'img_compression': {'cat': 'image', 'cpu': 60, 'mem': 1.5},
        'img_format_conv': {'cat': 'image', 'cpu': 35, 'mem': 1.2},
        'img_watermark': {'cat': 'image', 'cpu': 40, 'mem': 1.8},
        'img_puzzle_split': {'cat': 'image', 'cpu': 50, 'mem': 2.5},
        'img_color_corr': {'cat': 'image', 'cpu': 45, 'mem': 2.2},
        'img_bg_removal': {'cat': 'image', 'cpu': 90, 'mem': 4.5}, # Heavy
        'img_annotation': {'cat': 'image', 'cpu': 20, 'mem': 1.5},
        'img_batch_rename': {'cat': 'image', 'cpu': 5,  'mem': 0.5},
        
        # VIDEO (10) - Heavy Resource Use
        'vid_cropping': {'cat': 'video', 'cpu': 85, 'mem': 5.0},
        'vid_trimming': {'cat': 'video', 'cpu': 60, 'mem': 3.0},
        'vid_compression': {'cat': 'video', 'cpu': 98, 'mem': 6.5}, # Maximum
        'vid_remove_audio': {'cat': 'video', 'cpu': 30, 'mem': 2.0},
        'vid_add_subtitles': {'cat': 'video', 'cpu': 75, 'mem': 4.0},
        'vid_format_conv': {'cat': 'video', 'cpu': 92, 'mem': 5.5},
        'vid_frame_extraction': {'cat': 'video', 'cpu': 70, 'mem': 3.5},
        'vid_gif_creation': {'cat': 'video', 'cpu': 80, 'mem': 4.5},
        'vid_watermarking': {'cat': 'video', 'cpu': 85, 'mem': 4.2},
        'vid_split_segments': {'cat': 'video', 'cpu': 50, 'mem': 3.0},
        
        # AUDIO (5)
        'aud_noise_red': {'cat': 'audio', 'cpu': 70, 'mem': 2.5},
        'aud_format_conv': {'cat': 'audio', 'cpu': 25, 'mem': 1.0},
        'aud_trimming': {'cat': 'audio', 'cpu': 15, 'mem': 0.8},
        'aud_normalization': {'cat': 'audio', 'cpu': 30, 'mem': 1.2},
        'aud_split_track': {'cat': 'audio', 'cpu': 20, 'mem': 1.0},
        
        # PDF (5)
        'pdf_merge': {'cat': 'pdf', 'cpu': 15, 'mem': 3.5},
        'pdf_split': {'cat': 'pdf', 'cpu': 10, 'mem': 2.5},
        'pdf_to_office': {'cat': 'pdf', 'cpu': 65, 'mem': 4.5},
        'pdf_password': {'cat': 'pdf', 'cpu': 20, 'mem': 1.0},
        'pdf_extraction': {'cat': 'pdf', 'cpu': 30, 'mem': 1.5}
    }
    
    task_types = list(tasks_config.keys())
    applications = ['cloud_orchestrator', 'mobile_edge', 'serverless_fn', 'batch_processor']
    priorities = ['low', 'medium', 'high', 'critical']
    
    # Tier configurations relative to Medium (1.0 CPU, 1024MB)
    tier_configs = {
        'medium': { 'cpu_cores': 1.0, 'mem_total': 1024, 'time_factor': 1.0, 'cpu_factor': 1.0, 'count': 4000 },
        'small':  { 'cpu_cores': 0.5, 'mem_total': 512,  'time_factor': 2.5, 'cpu_factor': 1.8, 'count': 3500 },
        'tiny':   { 'cpu_cores': 0.25, 'mem_total': 256, 'time_factor': 5.5, 'cpu_factor': 3.5, 'count': 2500 }
    }

    # 1. Create exactly 30 unique base task templates (one per type)
    templates = []
    for t_type in task_types:
        cfg = tasks_config[t_type]
        input_sz = 15.0 # Constant scale for Gold Reference
        cat_base_time = {'image': 1.5, 'video': 18.0, 'audio': 4.0, 'pdf': 2.0}[cfg['cat']]
        base_mem = (input_sz * cfg['mem']) + 30.0
        
        templates.append({
            'task_signature': f"GOLD_{t_type}",
            'task_type': t_type,
            'application': 'standard_benchmark',
            'priority': 'medium',
            'complexity': 'high' if cfg['cpu'] > 80 else 'medium',
            'task_size_category': 'HIGH_15',
            'input_size_mb': input_sz,
            'm_cpu': cfg['cpu'],
            'm_mem': base_mem,
            'm_time': cat_base_time,
            'instructions': int(input_sz * cfg['cpu'] * 6e5)
        })

    print(f"✅ Generated 30 GOLD reference templates.")
    start_date = datetime.now() - timedelta(days=20)
    
    for tier, config in tier_configs.items():
        new_profiles = []
        for i in range(config['count']):
            t = random.choice(templates)
            noise = lambda: random.uniform(0.95, 1.05)
            
            # Simulated Variety in inputs for TRAINING observations
            obs_sz = random.uniform(0.1, 15.0)
            sz_ratio = obs_sz / 15.0
            
            cpu_usage = min(100.0, t['m_cpu'] * config['cpu_factor'] * noise())
            exec_time = t['m_time'] * config['time_factor'] * sz_ratio * noise()
            mem_usage = t['m_mem'] * sz_ratio * noise()
            
            p_static = 12.0 if tier == 'tiny' else (32.0 if tier == 'small' else 55.0)
            power = p_static + (40.0 * (cpu_usage / 100.0))
            
            p = {
                "id": f"TASK_{tier.upper()}_{i}",
                "task_signature": t['task_signature'],
                "task_type": t['task_type'],
                "application": t['application'],
                "priority": random.choice(priorities),
                "complexity": t['complexity'],
                "input_size_mb": round(obs_sz, 2),
                "task_size_category": 'HIGH_15' if obs_sz > 10 else ('MED_8' if obs_sz > 5 else 'LOW_3'),
                "cpu_usage_cores_absolute": (cpu_usage / 100.0) * config['cpu_cores'],
                "memory_usage_mb": mem_usage,
                "execution_time_normalized": exec_time,
                "power_consumption_watts": power,
                "energy_consumption_joules": power * exec_time,
                "instruction_count": int(t['instructions'] * sz_ratio * noise()),
                "timestamp": (start_date + timedelta(minutes=random.randint(0, 28800))).isoformat()
            }
            new_profiles.append(p)
            
        os.makedirs(os.path.dirname(files[tier]), exist_ok=True)
        with open(files[tier], 'w') as f:
            json.dump(new_profiles, f, indent=2)
        print(f"✅ Generated {config['count']} observations for {tier} tier.")

    # 3. Create the UNIQUE CLEAN final dataset (Exactly 30 GOLD rows)
    all_unique_gold = []
    for t in templates:
        all_unique_gold.append({
            "task_signature": t['task_signature'],
            "task_type": t['task_type'],
            "application": t['application'],
            "priority": t['priority'],
            "complexity": t['complexity'],
            "input_size_mb": t['input_size_mb'],
            "task_size_category": t['task_size_category'],
            "cpu_usage_cores_absolute": (t['m_cpu'] / 100.0),
            "memory_usage_mb": t['m_mem'],
            "execution_time_normalized": t['m_time'],
            "power_consumption_watts": 55.0 + (40.0 * (t['m_cpu'] / 100.0)),
            "instruction_count": t['instructions']
        })
    
    unique_file = '/Users/harshareddy/Desktop/intellicloud/dataset/task_profiles_clean_final.json'
    with open(unique_file, 'w') as f:
        json.dump(all_unique_gold, f, indent=2)
    print(f"✅ Generated 30 UNIQUE GOLD benchmark profiles at {unique_file}")

if __name__ == "__main__":
    scale_multi_modal_profiles()
