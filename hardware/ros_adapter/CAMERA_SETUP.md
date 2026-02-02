# Camera Publisher for ROS Adapter

## Quick Start

```bash
# Terminal 1: Start camera publisher
cd /home/mecha/hive_mind/hardware/ros_adapter
source /opt/ros/jazzy/setup.bash
./start_camera.sh
```

## Manual Usage

```bash
# Basic usage (default /dev/video0 at 20 FPS)
python3 camera_publisher.py

# Specify device and frame rate
python3 camera_publisher.py --device 0 --fps 30

# Use different camera
python3 camera_publisher.py --device 2
```

## Verify Camera is Publishing

```bash
# Check if topic exists
ros2 topic list | grep camera

# See camera info
ros2 topic info /camera/image_raw

# Echo messages (will show headers, not image data)
ros2 topic echo /camera/image_raw --no-arr

# Check publishing rate
ros2 topic hz /camera/image_raw
```

## Troubleshooting

### Camera not found
```bash
# List available cameras
ls -la /dev/video*

# Check camera permissions
sudo usermod -a -G video $USER
# Then log out and back in
```

### No image data
```bash
# Test camera with OpenCV
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera opened:', cap.isOpened())"
```

### Low frame rate
- Reduce FPS: `--fps 15`
- Check CPU usage
- Ensure good lighting (cameras reduce FPS in low light)

## Integration with ROS Adapter

The camera publisher publishes to `/camera/image_raw`, which is the default topic the ROS adapter subscribes to.

Once this is running along with:
1. All pipeline services (Perception, Task, Action, Pipeline)
2. ROS adapter (`ros_node.py`)

The adapter will start publishing commands to the arm controllers.
