#!/usr/bin/env python3
"""
Create Docker VMs for Real Task Execution
Creates tiny, small, and medium VMs with proper resource constraints
"""

import docker
import json
from pathlib import Path

class VMManager:
    def __init__(self):
        self.client = docker.from_env()
        self.vms = []
        
    def create_vm_config(self):
        """Define VM configurations"""
        return {
            'tiny': {
                'count': 3,
                'cpu_quota': 25000,  # 0.25 cores (25% of 100000)
                'cpu_period': 100000,
                'memory': '256m',
                'cores': 0.25,
                'memory_mb': 256
            },
            'small': {
                'count': 4,
                'cpu_quota': 50000,  # 0.5 cores (50% of 100000)
                'cpu_period': 100000,
                'memory': '512m',
                'cores': 0.5,
                'memory_mb': 512
            },
            'medium': {
                'count': 3,
                'cpu_quota': 100000,  # 1.0 cores (100% of 100000)
                'cpu_period': 100000,
                'memory': '1g',
                'cores': 1.0,
                'memory_mb': 1024
            }
        }
    
    def create_vms(self):
        """Create all VMs"""
        print("🚀 Creating Docker VMs...")
        print("=" * 60)
        
        config = self.create_vm_config()
        vm_inventory = []
        
        for tier, spec in config.items():
            print(f"\n📦 Creating {spec['count']} {tier.upper()} VMs...")
            
            for i in range(spec['count']):
                vm_id = f"vm-{tier}-{i+1}"
                
                try:
                    # Create container
                    container = self.client.containers.run(
                        image='python:3.11-slim',
                        name=vm_id,
                        detach=True,
                        tty=True,
                        stdin_open=True,
                        cpu_quota=spec['cpu_quota'],
                        cpu_period=spec['cpu_period'],
                        mem_limit=spec['memory'],
                        volumes={
                            str(Path.cwd() / 'data' / 'real_datasets'): {
                                'bind': '/datasets',
                                'mode': 'ro'
                            },
                            str(Path.cwd() / 'scripts'): {
                                'bind': '/scripts',
                                'mode': 'ro'
                            }
                        },
                        command='tail -f /dev/null'  # Keep container running
                    )
                    
                    print(f"   ✅ Created {vm_id} ({spec['cores']} cores, {spec['memory_mb']}MB RAM)")
                    
                    # Install dependencies
                    print(f"      Installing dependencies...")
                    
                    # Update package list
                    container.exec_run('apt-get update', demux=False)
                    
                    # Install system packages
                    container.exec_run(
                        'apt-get install -y ffmpeg stress-ng',
                        demux=False
                    )
                    
                    # Install Python packages
                    container.exec_run(
                        'pip install --no-cache-dir pillow pandas numpy scikit-learn',
                        demux=False
                    )
                    
                    print(f"      ✅ Dependencies installed")
                    
                    # Store VM info
                    vm_info = {
                        'vm_id': vm_id,
                        'vm_tier': tier,
                        'vm_cpu_cores': spec['cores'],
                        'vm_memory_mb': spec['memory_mb'],
                        'container_id': container.id,
                        'status': 'ready'
                    }
                    
                    vm_inventory.append(vm_info)
                    self.vms.append(container)
                    
                except Exception as e:
                    print(f"   ❌ Failed to create {vm_id}: {e}")
        
        # Save VM inventory
        inventory_file = Path('data/vm_inventory.json')
        with open(inventory_file, 'w') as f:
            json.dump(vm_inventory, f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"✅ Created {len(vm_inventory)} VMs")
        print(f"📄 Inventory saved to: {inventory_file}")
        print("=" * 60)
        
        return vm_inventory
    
    def cleanup_existing_vms(self):
        """Remove any existing VMs"""
        print("🧹 Cleaning up existing VMs...")
        
        vm_prefixes = ['vm-tiny-', 'vm-small-', 'vm-medium-']
        removed_count = 0
        
        for container in self.client.containers.list(all=True):
            if any(container.name.startswith(prefix) for prefix in vm_prefixes):
                try:
                    container.stop()
                    container.remove()
                    print(f"   ✅ Removed {container.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"   ⚠️  Failed to remove {container.name}: {e}")
        
        if removed_count > 0:
            print(f"✅ Removed {removed_count} existing VMs\n")
        else:
            print("   No existing VMs found\n")
    
    def test_vms(self):
        """Test that VMs are working"""
        print("\n🧪 Testing VMs...")
        print("=" * 60)
        
        for container in self.vms[:3]:  # Test first 3 VMs
            vm_name = container.name
            
            print(f"\n📋 Testing {vm_name}...")
            
            # Test Python
            result = container.exec_run('python3 --version')
            print(f"   Python: {result.output.decode().strip()}")
            
            # Test Pillow
            result = container.exec_run('python3 -c "import PIL; print(PIL.__version__)"')
            print(f"   Pillow: {result.output.decode().strip()}")
            
            # Test pandas
            result = container.exec_run('python3 -c "import pandas; print(pandas.__version__)"')
            print(f"   Pandas: {result.output.decode().strip()}")
            
            # Test dataset access
            result = container.exec_run('ls -lh /datasets')
            if result.exit_code == 0:
                print(f"   ✅ Datasets accessible")
            else:
                print(f"   ❌ Datasets not accessible")
        
        print("\n" + "=" * 60)
        print("✅ VM testing complete")
        print("=" * 60)


if __name__ == '__main__':
    manager = VMManager()
    
    # Cleanup existing VMs
    manager.cleanup_existing_vms()
    
    # Create new VMs
    vm_inventory = manager.create_vms()
    
    # Test VMs
    manager.test_vms()
    
    print("\n🎉 VMs are ready for task execution!")
