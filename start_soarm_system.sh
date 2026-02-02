#!/bin/bash
# Complete system startup for SoArm 100 with flexible composition
# Usage: ./start_soarm_system.sh [--task MODEL] [--action MODE]

# Default composition
TASK_MODEL="simple_mlp"
ACTION_MODE="frozen_policy"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --task) TASK_MODEL="$2"; shift ;;
        --action) ACTION_MODE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=== SoArm 100 Unified Pipeline Startup ==="
echo "Composition: Task=$TASK_MODEL, Action=$ACTION_MODE"
echo ""

# Check if in hive_mind directory
if [ ! -d "services" ]; then
    echo "ERROR: Must run from /home/mecha/hive_mind directory"
    exit 1
fi

# Create tmux session
SESSION="soarm_pipeline"

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null

# Start new session
tmux new-session -d -s $SESSION -n "driver"

# Window 0: Hardware Driver
tmux send-keys -t $SESSION:driver "cd ~/ros2_ws" C-m
tmux send-keys -t $SESSION:driver "source /opt/ros/jazzy/setup.bash" C-m
tmux send-keys -t $SESSION:driver "source install/setup.bash" C-m
tmux send-keys -t $SESSION:driver "source ~/ros_venv/bin/activate" C-m
tmux send-keys -t $SESSION:driver "ros2 launch so_100_arm hardware.launch.py" C-m

# Window 1: Perception Service
tmux new-window -t $SESSION -n "perception"
tmux send-keys -t $SESSION:perception "cd /home/mecha/hive_mind" C-m
tmux send-keys -t $SESSION:perception "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:perception "uvicorn services.perception.app:app --host 0.0.0.0 --port 8001" C-m

# Window 2: Task Service  
tmux new-window -t $SESSION -n "task"
tmux send-keys -t $SESSION:task "cd /home/mecha/hive_mind" C-m
tmux send-keys -t $SESSION:task "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:task "TASK_MODEL=$TASK_MODEL uvicorn services.task.app:app --host 0.0.0.0 --port 8003" C-m

# Window 3: Action Service
tmux new-window -t $SESSION -n "action"
tmux send-keys -t $SESSION:action "cd /home/mecha/hive_mind" C-m
tmux send-keys -t $SESSION:action "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:action "ACTION_MODE=$ACTION_MODE uvicorn services.action.app:app --host 0.0.0.0 --port 8002" C-m

# Window 4: Pipeline
tmux new-window -t $SESSION -n "pipeline"
tmux send-keys -t $SESSION:pipeline "cd /home/mecha/hive_mind" C-m
tmux send-keys -t $SESSION:pipeline "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:pipeline "uvicorn services.pipeline.app:app --host 0.0.0.0 --port 8000" C-m

# Window 5: Camera Publisher
tmux new-window -t $SESSION -n "camera"
tmux send-keys -t $SESSION:camera "cd /home/mecha/hive_mind/hardware/ros_adapter" C-m
tmux send-keys -t $SESSION:camera "source /opt/ros/jazzy/setup.bash" C-m
tmux send-keys -t $SESSION:camera "export PYTHONPATH=/home/mecha/hive_mind:\$PYTHONPATH" C-m
tmux send-keys -t $SESSION:camera "source ~/ros_venv/bin/activate" C-m
tmux send-keys -t $SESSION:camera "./start_camera.sh" C-m

# Window 6: ROS Adapter
tmux new-window -t $SESSION -n "adapter"
tmux send-keys -t $SESSION:adapter "cd /home/mecha/hive_mind/hardware/ros_adapter" C-m
tmux send-keys -t $SESSION:adapter "source /opt/ros/jazzy/setup.bash" C-m
tmux send-keys -t $SESSION:adapter "export PYTHONPATH=/home/mecha/hive_mind:\$PYTHONPATH" C-m
tmux send-keys -t $SESSION:adapter "source ~/ros_venv/bin/activate" C-m
tmux send-keys -t $SESSION:adapter "python3 ros_node.py" C-m

# Attach to session
echo ""
echo "✅ All services started in tmux session '$SESSION'"
echo "To attach: tmux attach -t $SESSION"
echo ""

# tmux attach -t $SESSION
