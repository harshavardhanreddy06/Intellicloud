#!/bin/bash
# Monitor VM creation progress

echo "📊 VM Creation Progress Monitor"
echo "================================"
echo ""

while true; do
    clear
    echo "📊 VM Creation Progress Monitor"
    echo "================================"
    echo "Time: $(date '+%H:%M:%S')"
    echo ""
    
    # Count VMs
    total_vms=$(docker ps -a | grep -c "vm-" || echo "0")
    running_vms=$(docker ps | grep -c "vm-" || echo "0")
    
    echo "VMs Created: $total_vms / 10"
    echo "VMs Running: $running_vms / 10"
    echo ""
    
    # Show VM status
    echo "VM Status:"
    echo "----------"
    docker ps --format "{{.Names}}\t{{.Status}}" | grep "vm-" | sort || echo "No VMs yet"
    
    echo ""
    
    # Estimate progress
    if [ "$total_vms" -gt 0 ]; then
        progress=$((total_vms * 10))
        echo -n "Progress: ["
        for i in {1..10}; do
            if [ $i -le $total_vms ]; then
                echo -n "█"
            else
                echo -n "░"
            fi
        done
        echo "] $progress%"
    fi
    
    echo ""
    
    # Check if complete
    if [ "$total_vms" -eq 10 ]; then
        echo "✅ All VMs created!"
        echo ""
        echo "Next: Task execution will start automatically"
        break
    fi
    
    sleep 5
done
