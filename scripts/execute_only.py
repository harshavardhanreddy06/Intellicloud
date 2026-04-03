#!/usr/bin/env python3
"""
Parallel Real Task Executor
Executes tasks across all 10 VMs simultaneously without creating/deleting VMs.
Collects ACCURATE CPU usage using delta calculation.
"""

import docker
import json
import time
import argparse
import random
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Lock for thread-safe file writing
write_lock = threading.Lock()

class ParallelTaskExecutor:
    def __init__(self, output_file='data/real_profiles/real_task_profiles_3k.json'):
        self.client = docker.from_env()
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.profiles = []
        self.failed_tasks = []
        
        # Load VM inventory
        with open('data/vm_inventory.json', 'r') as f:
            self.vms = json.load(f)
        
        print(f"✅ Loaded {len(self.vms)} VMs from inventory")

        # Define all task types
        self.task_types = {
            'image_processing': [
                'image_resize', 'image_compression', 'thumbnail_generation',
                'image_filter_application'
            ],
            'csv_processing': [
                'csv_aggregation', 'csv_groupby', 'csv_correlation_analysis',
                'csv_merge_operations', 'data_deduplication'
            ],
            'nlp': [
                'text_tokenization', 'text_word_count', 'text_search_replace'
            ],
            'log_processing': [
                'log_parsing', 'log_pattern_matching', 'log_aggregation'
            ],
            'scientific': [
                'matrix_multiplication', 'monte_carlo_simulation', 
                'statistical_analysis'
            ]
        }
        
        self.all_task_types = [
            task for category in self.task_types.values() 
            for task in category
        ]
        
        self.size_categories = ['MEDIUM', 'LARGE', 'HUGE']
        self.priorities = ['low', 'medium', 'high', 'critical']
        self.applications = ['data_pipeline', 'web_service', 'batch_processing', 'analytics', 'ml_training', 'etl_pipeline']

    def generate_task_plan(self, total_tasks=3000):
        """Generate plan for 3000 tasks"""
        print(f"📋 Generating task plan for {total_tasks} tasks...")
        
        task_plan = []
        task_id_counter = 0
        
        # Distribute tasks across sizes and VMs
        tasks_per_size = total_tasks // 3
        vm_tiers = ['tiny', 'small', 'medium']
        
        for size_category in self.size_categories:
            size_task_count = 0
            while size_task_count < tasks_per_size:
                task_type = random.choice(self.all_task_types)
                vm_tier = vm_tiers[size_task_count % 3]
                
                # Select specific VM from tier
                vm_options = [vm for vm in self.vms if vm['vm_tier'] == vm_tier]
                vm = random.choice(vm_options)
                
                task = {
                    'task_id': f'task_{task_id_counter:06d}',
                    'task_type': task_type,
                    'task_size_category': size_category,
                    'vm_id': vm['vm_id'],
                    'vm_tier': vm['vm_tier'],
                    'vm_cpu_cores': vm['vm_cpu_cores'],
                    'vm_memory_mb': vm['vm_memory_mb'],
                    'priority': random.choice(self.priorities),
                    'application': random.choice(self.applications)
                }
                
                task_plan.append(task)
                task_id_counter += 1
                size_task_count += 1
        
        return task_plan

    def get_container_cpu_stats(self, container):
        """Get raw CPU stats snapshot"""
        try:
            stats = container.stats(stream=False)
            cpu_usage = stats['cpu_stats']['cpu_usage']['total_usage']
            system_usage = stats['cpu_stats']['system_cpu_usage']
            online_cpus = stats['cpu_stats'].get('online_cpus', 1)
            # Memory usage (current)
            memory_usage = stats['memory_stats'].get('usage', 0)
            # Network stats
            networks = stats.get('networks', {})
            network_io = sum(d.get('rx_bytes', 0) + d.get('tx_bytes', 0) for d in networks.values())
            
            return {
                'cpu_usage': cpu_usage,
                'system_usage': system_usage,
                'online_cpus': online_cpus,
                'memory_usage': memory_usage,
                'network_io': network_io
            }
        except Exception as e:
            # Fallback
            return {'cpu_usage': 0, 'system_usage': 0, 'online_cpus': 1, 'memory_usage': 0, 'network_io': 0}

    def worker_execute_tasks(self, vm_id, tasks):
        """Worker function to execute a list of tasks on a specific VM"""
        print(f"🚀 Worker started for {vm_id} with {len(tasks)} tasks")
        
        try:
            container = self.client.containers.get(vm_id)
        except Exception as e:
            print(f"❌ Failed to connect to {vm_id}: {e}")
            return

        completed_count = 0
        
        for task in tasks:
            task_id = task['task_id']
            task_type = task['task_type']
            size_category = task['task_size_category']

            # Create task script
            script_content = f"""
import sys
sys.path.append('/scripts')
from task_workloads import TaskWorkloads
import time

try:
    workloads = TaskWorkloads()
    start = time.time()
    result = workloads.execute_task('{task_type}', '{size_category}')
    end = time.time()
    print('EXECUTION_TIME:', end - start)
    print('RESULT:', result)
except Exception as e:
    print('ERROR:', e)
"""
            # Write script to container
            script_path = f"/tmp/{task_id}.py"
            container.exec_run(
                f"sh -c \"cat > {script_path} << 'EOF'\n{script_content}\nEOF\""
            )

            # Capture stats BEFORE execution
            stats_before = self.get_container_cpu_stats(container)
            
            # Execute task
            start_time = time.time()
            exec_result = container.exec_run(
                f"python3 {script_path}",
                demux=True,
                environment={'PYTHONPATH': '/scripts'}
            )
            
            # Capture stats AFTER execution
            stats_after = self.get_container_cpu_stats(container)
            
            # Clean up
            container.exec_run(f"rm {script_path}")
            
            # Parse execution result
            output = ""
            if exec_result.output:
                stdout, stderr = exec_result.output
                if stdout: output += stdout.decode('utf-8')
                if stderr: output += stderr.decode('utf-8')
            
            # Extract execution time from python output if available
            execution_time = 0.0
            for line in output.splitlines():
                if line.startswith('EXECUTION_TIME:'):
                    try:
                        execution_time = float(line.split(':')[1].strip())
                    except: pass
            
            if execution_time == 0.0:
                 execution_time = time.time() - start_time # Fallback to wall clock
            
            # Calculate CPU usage over execution interval
            cpu_delta = stats_after['cpu_usage'] - stats_before['cpu_usage']
            system_delta = stats_after['system_usage'] - stats_before['system_usage']
            num_cores = stats_after['online_cpus']
            
            if system_delta > 0 and cpu_delta > 0:
                raw_cpu_percent = (cpu_delta / system_delta) * num_cores * 100.0
                # Normalize relative to VM limit (e.g. 25% raw on 0.25 core VM = 100% load)
                cpu_percent = (raw_cpu_percent / task['vm_cpu_cores'])
            else:
                cpu_percent = 0.0
            
            # Memory & Network
            memory_mb = stats_after['memory_usage'] / (1024 * 1024)
            network_io_mb = stats_after['network_io'] / (1024 * 1024)
            
            # Power estimate
            base_power = 25
            cpu_power = (cpu_percent / 100) * 50
            memory_power = (memory_mb / 1024) * 10
            power_watts = base_power + cpu_power + memory_power
            
            # Instruction count estimate
            instructions_per_second = 1_000_000_000
            instruction_count = int((cpu_percent / 100) * task['vm_cpu_cores'] * instructions_per_second * execution_time)
            
            if cpu_percent >= 80:
                complexity = 'high'
            elif cpu_percent >= 60:
                complexity = 'medium'
            else:
                complexity = 'low'

            profile = {
                'task_id': task_id,
                'task_signature': f"{task_type}_{size_category.lower()}_pipeline",
                'task_type': task_type,
                'task_category': 'real',
                'application': task['application'],
                'priority': task['priority'],
                'task_size_category': size_category,
                'input_size_mb': self.get_input_size(task_type, size_category),
                'vm_id': vm_id,
                'vm_tier': task['vm_tier'],
                'vm_cpu_cores': task['vm_cpu_cores'],
                'vm_memory_mb': task['vm_memory_mb'],
                'cpu_usage_percent': round(cpu_percent, 2),
                'memory_usage_mb': round(memory_mb, 2),
                'network_io_mb': round(network_io_mb, 2),
                'power_consumption_watts': round(power_watts, 2),
                'execution_time_sec': round(execution_time, 2),
                'instruction_count': instruction_count,
                'complexity': complexity,
                'executed_at': datetime.now().isoformat()
            }
            
            with write_lock:
                self.profiles.append(profile)
                if len(self.profiles) % 10 == 0:
                    self.save_profiles()
                    
            completed_count += 1
            print(f"[{vm_id}] ✅ {task_id} ({task_type}): {cpu_percent:.1f}% CPU, {execution_time:.1f}s")
            
            # Small delay to allow counters to settle if needed, but parallelism uses waiting tasks anyway
            # time.sleep(0.1)

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
        return size_map[size_category][category]

    def save_profiles(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.profiles, f, indent=2)

    def run(self, total_tasks=3000):
        task_plan = self.generate_task_plan(total_tasks)
        
        # Group tasks by VM
        tasks_by_vm = {vm['vm_id']: [] for vm in self.vms}
        for task in task_plan:
            tasks_by_vm[task['vm_id']].append(task)
            
        print(f"\n🚀 Starting parallel execution on {len(self.vms)} VMs...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=len(self.vms)) as executor:
            futures = []
            for vm_id, tasks in tasks_by_vm.items():
                if tasks:
                    futures.append(executor.submit(self.worker_execute_tasks, vm_id, tasks))
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ Worker exception: {e}")

        total_time = time.time() - start_time
        self.save_profiles()
        print(f"\n✅ All tasks completed in {total_time/60:.1f} minutes")
        print(f"📄 Profiles saved to {self.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=3000)
    args = parser.parse_args()
    
    executor = ParallelTaskExecutor()
    executor.run(args.count)
