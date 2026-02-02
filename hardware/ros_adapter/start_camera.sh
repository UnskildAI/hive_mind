#!/bin/bash
# Quick start script for testing ROS adapter with camera

echo "=== ROS Adapter Camera Publisher Test ==="
echo ""
echo "This script will start the camera publisher."
echo "Make sure you have:"
echo "  1. ROS2 sourced: source /opt/ros/jazzy/setup.bash"
echo "  2. Camera connected at /dev/video0"
echo "  3. All pipeline services running (Perception, Task, Action, Pipeline)"
echo ""

# Check if ROS2 is sourced
if ! command -v ros2 &> /dev/null; then
    echo "ERROR: ROS2 not found. Please source ROS2 setup:"
    echo "  source /opt/ros/jazzy/setup.bash"
    exit 1
fi

# Check if camera exists
if [ ! -e /dev/video0 ]; then
    echo "WARNING: /dev/video0 not found. Camera may not be connected."
    echo "Available video devices:"
    ls -la /dev/video* 2>/dev/null || echo "  No video devices found"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Starting camera publisher..."
echo "Press Ctrl+C to stop"
echo ""

cd "$(dirname "$0")"
export PYTHONPATH=/home/mecha/hive_mind:$PYTHONPATH

python3 camera_publisher.py --device 0 --fps 20
