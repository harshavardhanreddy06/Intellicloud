#!/bin/bash
# Monitor task execution progress

echo "📊 Real Task Execution Monitor"
echo "=============================="

while true; do
    clear
    echo "📊 Real Task Execution Monitor"
    echo "=============================="
    echo "Time: $(date '+%H:%M:%S')"
    echo ""
    
    # Count profiles
    if [ -f "data/real_profiles/real_task_profiles_3k.json" ]; then
        count=$(grep -o "task_id" "data/real_profiles/real_task_profiles_3k.json" | wc -l)
    else
        count=0
    fi
    
    total=3000
    percent=$((count * 100 / total))
    
    # Progress bar
    bar_len=$((percent / 2))
    printf "Progress: ["
    for ((i=0; i<bar_len; i++)); do printf "█"; done
    for ((i=bar_len; i<50; i++)); do printf "░"; done
    printf "] %d%% (%d/%d)\n" "$percent" "$count" "$total"
    
    echo ""
    
    # Estimate ETA (assume 30 tasks/min = 0.5 tasks/sec)
    # Better: calculate speed based on elapsed time? Hard without start time.
    # Just show recent profiles
    
    echo "Recent Profiles:"
    echo "----------------"
    if [ -f "data/real_profiles/real_task_profiles_3k.json" ]; then
        tail -n 20 "data/real_profiles/real_task_profiles_3k.json" | grep "task_type" -B 1 -A 5 | grep -v "\-\-" | sed 's/  //g' | head -n 5
    else
        echo "No profiles yet..."
    fi
    
    echo ""
    echo "Refresh in 5s..."
    sleep 5
done
