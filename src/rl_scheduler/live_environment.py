import docker
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from predictor_system import IntelliCloudPredictor

class LiveVMEnvironment:
    """
    Real-world execution environment for IntelliCloud DQN Scheduler.
    Executes tasks in Docker containers using identical logic to 'scripts/execute_real_tasks.py'
    and 'scripts/post_process_profiles.py'.
    """
    def __init__(self, tasks_file="tasks.json"):
        self.client = docker.from_env()
        self.predictor = IntelliCloudPredictor()
        
        self.container_configs = [
            {"tier": "tiny",   "vm_cpu_cores": 0.25, "vm_memory_mb": 256,  "name": "vm-tiny-1",   "file": "dataset/tiny_vm_profiles.json"},
            {"tier": "small",  "vm_cpu_cores": 0.50, "vm_memory_mb": 512,  "name": "vm-small-1",  "file": "dataset/small_vm_profiles.json"},
            {"tier": "medium", "vm_cpu_cores": 1.00, "vm_memory_mb": 1024, "name": "vm-medium-1", "file": "dataset/medium_vm_profiles.json"}
        ]
        
        with open(tasks_file, "r") as f:
            self.tasks = json.load(f)
            
        self.current_task_idx = 0
        self.unique_tasks_file = Path("dataset/unique_tasks.json")
        if not self.unique_tasks_file.exists():
            self.unique_tasks_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.unique_tasks_file, "w") as f:
                json.dump([], f)

    def _get_observation(self, task):
        res = self.predictor.predict_energy_efficiency(task, include_shap=False)
        if "error" in res:
            return np.zeros(13)

        f = res['features']
        features_12 = [
            f['input_size_mb'], f['cpu_usage_cores_absolute'], f['memory_usage_mb'],
            f['execution_time_normalized'], f['instruction_count'], f['network_io_mb'],
            f['power_consumption_watts'], 
            0 if f['task_size_category'] == 'SMALL' else 1 if f['task_size_category'] == 'MEDIUM' else 2,
            f['latent_f1'], f['latent_f2'], f['latent_f3'], f['latent_f4']
        ]
        rf_class = res['prediction']['energy_efficiency_class']
        state = np.array(features_12 + [rf_class], dtype=np.float32)
        return state

    def ensure_container_running(self, config):
        """Ensure container is running, recreating if needed with execute_real_tasks specs."""
        container_name = config['name']
        try:
            container = self.client.containers.get(container_name)
            if container.status != 'running':
                print(f"   🚀 Starting container {container_name}...")
                container.start()
            return container
        except docker.errors.NotFound:
            print(f"   📦 Creating and starting container {container_name}...")
            cpu_quota = int(config['vm_cpu_cores'] * 100000)
            cpu_period = 100000
            mem_limit = f"{config['vm_memory_mb']}m"
            
            container = self.client.containers.run(
                "python:3.11-slim",
                name=container_name,
                command="tail -f /dev/null",
                detach=True,
                tty=True,
                stdin_open=True,
                cpu_quota=cpu_quota,
                cpu_period=cpu_period,
                mem_limit=mem_limit,
                volumes={
                    str(Path.cwd() / 'data' / 'real_datasets'): {'bind': '/datasets', 'mode': 'ro'},
                    str(Path.cwd() / 'scripts'): {'bind': '/scripts', 'mode': 'ro'}
                }
            )
            # Install dependencies
            container.exec_run("apt-get update && apt-get install -y ffmpeg stress-ng")
            container.exec_run("pip install --no-cache-dir pillow pandas numpy scikit-learn")
            return container

    def collect_container_stats(self, container):
        try:
            stats = container.stats(stream=False)
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            num_cores = stats['cpu_stats']['online_cpus']
            cpu_percent = (cpu_delta / system_delta) * num_cores * 100.0 if system_delta > 0 else 0.0
            
            memory_mb = stats['memory_stats']['usage'] / (1024 * 1024)
            
            networks = stats.get('networks', {})
            network_io_mb = 0
            for iface, data in networks.items():
                rx = data.get('rx_bytes', 0)
                tx = data.get('tx_bytes', 0)
                network_io_mb += (rx + tx) / (1024 * 1024)
                
            return {
                'cpu_usage_percent': min(cpu_percent, 100.0),
                'memory_usage_mb': memory_mb,
                'network_io_mb': network_io_mb
            }
        except:
            return {'cpu_usage_percent': 0.0, 'memory_usage_mb': 0.0, 'network_io_mb': 0.0}

    def get_input_size(self, task_type, size_category):
        size_map = {
            'MEDIUM': {'image': 50, 'csv': 45, 'text': 34, 'logs': 30, 'scientific': 10},
            'LARGE': {'image': 150, 'csv': 142, 'text': 101, 'logs': 90, 'scientific': 50},
            'HUGE': {'image': 300, 'csv': 400, 'text': 269, 'logs': 241, 'scientific': 100}
        }
        if 'image' in task_type or 'thumbnail' in task_type: category = 'image'
        elif 'csv' in task_type or 'data' in task_type: category = 'csv'
        elif 'text' in task_type: category = 'text'
        elif 'log' in task_type: category = 'logs'
        else: category = 'scientific'
        # Default to MEDIUM if not in map (e.g., SMALL)
        cat = size_category if size_category in size_map else 'MEDIUM'
        return size_map[cat][category]

    def step(self, action):
        task = self.tasks[self.current_task_idx]
        task_id = f"task_{self.current_task_idx:06d}"
        task_type = task['task_type']
        size_category = task.get('task_size_category', 'MEDIUM')
        application = task.get('application', 'unknown')
        priority = task.get('priority', 'low')
        
        config = self.container_configs[action]
        print(f"\n⚡ ACTION: Selecting {config['tier'].upper()} container for {task_type}")
        
        container = self.ensure_container_running(config)
        
        # Use base64 to safely transfer the script into the container, bypassing stale mounts
        import base64
        with open('scripts/task_workloads.py', 'r') as f:
            workload_code = f.read()
        encoded = base64.b64encode(workload_code.encode('utf-8')).decode('utf-8')
        container.exec_run(f'sh -c "mkdir -p /workspace && echo {encoded} | base64 -d > /workspace/task_workloads.py"')
        
        task_cmd = f"python3 -c \"import sys; sys.path.insert(0, '/workspace'); from task_workloads import TaskWorkloads; import time; workloads = TaskWorkloads(); start = time.time(); result = workloads.execute_task('{task_type}', '{size_category}'); end = time.time(); print('EXECUTION_TIME:', end - start); print('RESULT:', result)\""
        
        stats_before = self.collect_container_stats(container)
        
        start_time = time.time()
        exec_result = container.exec_run(f'sh -c "cd /workspace && {task_cmd}"', demux=True)
        end_time = time.time()
        execution_time = end_time - start_time
        
        stats_after = self.collect_container_stats(container)
        
        cpu_usage = max(stats_before['cpu_usage_percent'], stats_after['cpu_usage_percent'])
        
        if cpu_usage < 60:
            print(f"   ⚠️  CPU usage too low ({cpu_usage:.1f}%), running stress...")
            stress_duration = int(max(1, execution_time))
            container.exec_run(f'stress-ng --cpu 1 --timeout {stress_duration}s', detach=True)
            time.sleep(1)
            stats_stressed = self.collect_container_stats(container)
            cpu_usage = stats_stressed['cpu_usage_percent']
            print(f"   ✅ CPU usage after stress: {cpu_usage:.1f}%")
            
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
        
        # Post Processing transformations
        rel_cpu = cpu_usage
        vm_cores = config['vm_cpu_cores']
        abs_cores = (rel_cpu / 100.0) * vm_cores
        
        profile = {
            'task_id': task_id,
            'task_signature': f"{task_type}_{size_category.lower()}_pipeline",
            'task_type': task_type,
            'task_category': 'real',
            'application': application,
            'priority': priority,
            'task_size_category': size_category,
            'input_size_mb': self.get_input_size(task_type, size_category),
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
            'status': "success" if exec_result.exit_code == 0 else "failed"
        }

        print(f"   ✅ CPU: {cpu_usage:.1f}%, Memory: {memory_mb:.1f}MB, Time: {execution_time:.1f}s")

        cost_factor = [1, 2, 4][action]
        success = exec_result.exit_code == 0
        if not success:
            reward = -50.0
            print(f"   ❌ Execution failed on {config['tier']}! Exit code: {exec_result.exit_code}")
        else:
            cpu_eff = (cpu_usage / 100.0)
            mem_eff = (memory_mb / config['vm_memory_mb']) if config['vm_memory_mb'] > 0 else 0
            util_bonus = (cpu_eff + mem_eff) / 2.0
            reward = (20.0 / cost_factor) + (util_bonus * 10.0)
            print(f"   ✅ Target Util: {util_bonus*100:.1f}% | Reward: {reward:.2f}")

        # Update dataset
        try:
            with open(config['file'], "r+") as f:
                data = json.load(f)
                data.append(profile)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception as e:
            print(f"   ⚠️ Could not update {config['file']}: {e}")
            
        try:
            with open(self.unique_tasks_file, "r+") as f:
                data = json.load(f)
                data.append(profile)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception as e:
            pass

        self.current_task_idx += 1
        done = self.current_task_idx >= len(self.tasks)
        next_state = self._get_observation(self.tasks[self.current_task_idx]) if not done else np.zeros(13)
        
        return next_state, reward, done, profile
