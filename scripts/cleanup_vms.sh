#!/bin/bash
# Cleanup VMs - ONLY RUN WHEN PROJECT IS COMPLETELY DONE

echo "⚠️  WARNING: This will DELETE all VMs!"
echo "==========================================="
echo ""
echo "This script will remove:"
echo "  - vm-tiny-1, vm-tiny-2, vm-tiny-3"
echo "  - vm-small-1, vm-small-2, vm-small-3, vm-small-4"
echo "  - vm-medium-1, vm-medium-2, vm-medium-3"
echo ""
echo "You will need to recreate them (~50 min) if you delete them."
echo ""

# Show current VMs
vm_count=$(docker ps -a | grep -c "vm-" || echo "0")
echo "Current VMs found: $vm_count"
echo ""

if [ "$vm_count" -eq 0 ]; then
    echo "✅ No VMs to delete"
    exit 0
fi

# Double confirmation
read -p "Are you SURE you want to DELETE all VMs? (type 'DELETE' to confirm): " confirm

if [ "$confirm" != "DELETE" ]; then
    echo "❌ Cancelled - VMs are safe"
    exit 0
fi

echo ""
echo "🗑️  Deleting VMs..."
echo ""

# Stop and remove all VMs
docker ps -aq --filter "name=vm-tiny-" --filter "name=vm-small-" --filter "name=vm-medium-" | while read container; do
    vm_name=$(docker inspect --format='{{.Name}}' $container | cut -c 2-)
    echo "   Removing $vm_name..."
    docker rm -f $container > /dev/null 2>&1
    echo "   ✅ Deleted $vm_name"
done

echo ""
echo "✅ All VMs deleted"
echo ""
echo "To recreate VMs, run:"
echo "  python3 scripts/create_vms.py"
echo ""
