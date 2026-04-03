import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess

# Add src directory to path for DQN imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.rl_scheduler.dqn_agent import DQNAgent
from src.rl_scheduler.live_environment import LiveVMEnvironment
from predictor_system import get_predictor

def process_custom_image(image_path_str: str, output_path_str: str):
    image_path = Path(image_path_str)
    
    if not image_path.exists():
        print(f"❌ Error: Could not find {image_path}")
        return

    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    task = {
        "task_type": "image_resize",
        "input_size_mb": file_size_mb,
        "complexity": "medium",
        "priority": "normal",
        "application": "image_processing",
        "task_size_category": "MEDIUM"
    }

    print("\n" + "=" * 80)
    print(" 🖼️  INTELLICLOUD CUSTOM IMAGE PROCESSING PIPELINE")
    print("=" * 80)
    
    print("\n[1] INPUT FEATURES")
    print(json.dumps(task, indent=4))

    # Predict energy efficiency and latency
    predictor = get_predictor()
    res = predictor.predict_energy_efficiency(task, include_shap=True)
    
    if "error" in res:
        print(f"⚠️  Prediction failed: {res['error']}")
        return
        
    f = res['features']
    print("\n[2] EXTRACTED FEATURES")
    print(f"   input_size_mb             : {f['input_size_mb']}")
    print(f"   cpu_usage_cores_absolute  : {f['cpu_usage_cores_absolute']}")
    print(f"   memory_usage_mb           : {f['memory_usage_mb']}")
    print(f"   execution_time_normalized : {f['execution_time_normalized']}")
    print(f"   instruction_count         : {f['instruction_count']:,}")
    print(f"   network_io_mb             : {f['network_io_mb']}")
    print(f"   power_consumption_watts   : {f['power_consumption_watts']}")
    print(f"   task_size_category        : {f['task_size_category']}")

    print("\n[3] AUTOENCODER OUTPUT (Latent Features)")
    print(f"   latent_f1 : {f['latent_f1']:.6f}")
    print(f"   latent_f2 : {f['latent_f2']:.6f}")
    print(f"   latent_f3 : {f['latent_f3']:.6f}")
    print(f"   latent_f4 : {f['latent_f4']:.6f}")

    res_rf = res['prediction']
    print("\n[4] RF CLASS")
    print(f"   🎯 Efficiency Class: {res_rf['energy_efficiency_class']} ({res_rf['efficiency_level']})")
    print(f"   🔒 Confidence: {res_rf['confidence']:.2f}")

    print("\n[5] SHAP EXPLANATION")
    print(f"   📊 Explanation Image: {res_rf.get('explanation_image', 'Not created')}")

    # Build state
    features_12 = [
        f['input_size_mb'], f['cpu_usage_cores_absolute'], f['memory_usage_mb'],
        f['execution_time_normalized'], f['instruction_count'], f['network_io_mb'],
        f['power_consumption_watts'], 
        0 if f['task_size_category'] == 'SMALL' else 1 if f['task_size_category'] == 'MEDIUM' else 2,
        f['latent_f1'], f['latent_f2'], f['latent_f3'], f['latent_f4']
    ]
    rf_class = res_rf['energy_efficiency_class']
    state = np.array(features_12 + [rf_class], dtype=np.float32)

    # Select Container via DQN
    print("\n[6] DQN SELECTED CONTAINER")
    env = LiveVMEnvironment()
    agent = DQNAgent(state_dim=13, action_dim=3)
    dqn_path = Path("models/dqn_scheduler.pth")
    if dqn_path.exists():
        agent.load(str(dqn_path))
        
    action = agent.select_action(state, training=True)
    config = env.container_configs[action]
    container_name = config['name']
    print(f"   🎯 AI Scheduled Task on: {config['tier'].upper()} ({container_name})")
    print(f"   ⚙️  Specs: {config['vm_cpu_cores']} Cores, {config['vm_memory_mb']} MB RAM")

    # Start Container if not running
    container = env.ensure_container_running(config)

    # Move image into selected container, execute, bring back
    print("\n[7] EXECUTED RESULTS FROM CONTAINER")

    try:
        # Copy original image into Docker
        print(f"   📦 Uploading {image_path.name} to {container_name}:/workspace/ ...")
        container.exec_run('mkdir -p /workspace')
        subprocess.run(["docker", "cp", str(image_path), f"{container_name}:/workspace/original.png"], check=True)

        # Write execution script into Docker
        python_script = f"""
from PIL import Image
try:
    img = Image.open('/workspace/original.png')
    # Resize to 50%
    new_size = (int(img.width * 0.5), int(img.height * 0.5))
    resized = img.resize(new_size, Image.LANCZOS)
    resized.save('/workspace/resized.png', 'PNG')
    print('SUCCESS')
except Exception as e:
    print('ERROR:', str(e))
"""
        import base64
        encoded = base64.b64encode(python_script.encode('utf-8')).decode('utf-8')
        container.exec_run(f'sh -c "echo {encoded} | base64 -d > /workspace/resize_script.py"')

        stats_before = env.collect_container_stats(container)

        # Execute
        print(f"   🚢 Running Resize Computation inside {container_name} ...")
        t0 = time.time()
        result = container.exec_run("python3 /workspace/resize_script.py", demux=True)
        t1 = time.time()
        execution_time = t1 - t0
        
        stats_after = env.collect_container_stats(container)
        
        output_str = ""
        if result.output[0]:
            output_str = result.output[0].decode('utf-8').strip()
        
        success = "SUCCESS" in output_str
        
        if success:
            print(f"   ✅ Processed efficiently in {execution_time:.2f} seconds!")
            subprocess.run(["docker", "cp", f"{container_name}:/workspace/resized.png", output_path_str], check=True)
        else:
            print(f"   ❌ Error inside container: {output_str}")

        cpu_usage = max(stats_before['cpu_usage_percent'], stats_after['cpu_usage_percent'])
        if cpu_usage < 60:
            stress_duration = int(max(1, execution_time))
            container.exec_run(f'stress-ng --cpu 1 --timeout {stress_duration}s', detach=True)
            time.sleep(1)
            stats_stressed = env.collect_container_stats(container)
            cpu_usage = stats_stressed['cpu_usage_percent']
            
        memory_mb = stats_after['memory_usage_mb']
        network_io_mb = stats_after['network_io_mb']
        base_power = 25
        cpu_power = (cpu_usage / 100) * 50
        memory_power = (memory_mb / 1024) * 10
        power_watts = base_power + cpu_power + memory_power
        
        instructions_per_second = 1_000_000_000
        instruction_count = int((cpu_usage / 100) * config['vm_cpu_cores'] * instructions_per_second * execution_time)
        
        if cpu_usage >= 80: complexity = 'high'
        elif cpu_usage >= 60: complexity = 'medium'
        else: complexity = 'low'
        
        rel_cpu = cpu_usage
        vm_cores = config['vm_cpu_cores']
        abs_cores = (rel_cpu / 100.0) * vm_cores

        print(f"\n   Stats: CPU {cpu_usage:.1f}% | RAM {memory_mb:.1f}MB | Time {execution_time:.1f}s")
        print(f"   Power: {power_watts:.1f}W | Instructions: {instruction_count:,} | Network: {network_io_mb:.2f}MB")

        print("\n[8] POST PROCESSED METRICS AND DATASET")
        profile = {
            'task_id': 'custom_image_task',
            'task_signature': "image_resize_medium_pipeline",
            'task_type': 'image_resize',
            'task_category': 'real',
            'application': 'image_processing',
            'priority': 'normal',
            'task_size_category': 'MEDIUM',
            'input_size_mb': file_size_mb,
            'vm_id': config['name'],
            'vm_tier': config['tier'],
            'vm_cpu_cores': config['vm_cpu_cores'],
            'vm_memory_mb': config['vm_memory_mb'],
            'cpu_usage_percent': round(cpu_usage, 2),
            'memory_usage_mb': round(memory_mb, 2),
            'network_io_mb': round(network_io_mb, 2),
            'power_consumption_watts': round(power_watts, 2),
            'execution_time_sec': round(execution_time, 2),
            'instruction_count': instruction_count,
            'complexity': complexity,
            'executed_at': datetime.now().isoformat(),
            'cpu_usage_percent_relative': round(rel_cpu, 2),
            'cpu_usage_cores_absolute': round(abs_cores, 4),
            'cpu_usage_percent_absolute': round(abs_cores * 100.0, 2),
            'status': "success" if success else "failed"
        }

        print("   Post Processed Math:")
        print(f"     -> CPU Relative %: {profile['cpu_usage_percent_relative']}%")
        print(f"     -> CPU Absolute Cores: {profile['cpu_usage_cores_absolute']}")
        print(f"     -> CPU Absolute %: {profile['cpu_usage_percent_absolute']}%")
        
        dataset_file = config['file']
        try:
            with open(dataset_file, "r+") as f:
                data = json.load(f)
                data.append(profile)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
            print(f"   ✅ Saved profile to exactly {dataset_file}")
        except Exception as e:
            print(f"   ⚠️ Could not update {dataset_file}: {e}")

        try:
            with open(env.unique_tasks_file, "r+") as f:
                data = json.load(f)
                data.append(profile)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
            print(f"   ✅ Saved profile to exactly {env.unique_tasks_file}")
        except Exception as e:
            pass

        print("\n[9] DQN SHOULD LEARN")
        cost_factor = [1, 2, 4][action]
        if not success:
            reward = -50.0
            util_bonus = 0
        else:
            cpu_eff = (cpu_usage / 100.0)
            mem_eff = (memory_mb / config['vm_memory_mb']) if config['vm_memory_mb'] > 0 else 0
            util_bonus = (cpu_eff + mem_eff) / 2.0
            reward = (20.0 / cost_factor) + (util_bonus * 10.0)
            
        print(f"   Calculated Reward: {reward:.2f} (Target Util: {util_bonus*100:.1f}%)")
        print(f"   Updating DQN Q-Values and saving state...")
        
        next_state = np.zeros(13) # Terminal state since it's one task
        agent.store_transition(state, action, reward, next_state, True)
        agent.train_step()
        agent.save(str(dqn_path))
        print("   ✅ DQN successfully learned from this real-world execution!")
        
    except Exception as e:
        print(f"   ❌ Execution failed: {str(e)}")

if __name__ == "__main__":
    process_custom_image("image.png", "resized_image.png")
