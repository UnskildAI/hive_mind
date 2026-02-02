# SoArm 100 Integration Guide - Testing Scripted Controller

This guide provides step-by-step instructions for testing the scripted controller on your SoArm 100 robot using the ROS adapter.

## Architecture Overview

```
SoArm 100 Robot (ROS2)
    ↓ (publishes)
/joint_states, /camera/image_raw
    ↓ (subscribes)
ROS Adapter Node
    ↓ (HTTP POST)
Pipeline Service (localhost:8000)
    ↓ (calls)
Perception → Task → Action (Scripted Mode)
    ↓ (returns)
ActionChunk (6 joints)
    ↓ (splits)
ROS Adapter
    ↙                    ↘
/so_100_arm_controller   /so_100_arm_gripper_controller

```

## Prerequisites

1. **ROS2 (Jazzy) installed and sourced**
2. **SoArm 100 ROS2 driver/package installed**
3. **All services running** (Perception, Task, Action, Pipeline)
4. **Camera connected** (USB or ROS camera node)

## Step 1: Configure Action Service for Scripted Mode

Edit `/home/mecha/hive_mind/services/action/config.yaml`:

```yaml
mode: scripted  # Enable scripted mode

scripted:
  trajectory_type: joint_oscillation  # Start with gentle oscillation
  frequency: 0.3  # Lower frequency for safety (0.3 Hz)
  amplitude: 0.05  # Small amplitude for initial test (0.05 rad ≈ 3°)
  joint_limits: [-3.14, 3.14]  # Adjust based on SoArm 100 limits
  velocity_limit: 0.5  # Conservative velocity limit
```

**Safety Note:** Start with small amplitude and low frequency. Gradually increase after verifying safe operation.

## Step 2: Start All Services

Open 4 terminal windows:

### Terminal 1: Perception Service
```bash
cd /home/mecha/hive_mind
source .venv/bin/activate
uvicorn services.perception.app:app --host 0.0.0.0 --port 8001
```

### Terminal 2: Task Service
```bash
cd /home/mecha/hive_mind
source .venv/bin/activate
uvicorn services.task.app:app --host 0.0.0.0 --port 8003
```

### Terminal 3: Action Service (Scripted Mode)
```bash
cd /home/mecha/hive_mind
source .venv/bin/activate
uvicorn services.action.app:app --host 0.0.0.0 --port 8002
```

Verify you see: `ScriptedController initialized.`

### Terminal 4: Pipeline Orchestrator
```bash
cd /home/mecha/hive_mind
source .venv/bin/activate
uvicorn services.pipeline.app:app --host 0.0.0.0 --port 8000
```

## Step 3: Verify Services

Test the pipeline endpoint:
```bash
curl http://localhost:8000/health
```

Should return: `{"status": "healthy"}`

## Step 4: Start SoArm 100 ROS2 Nodes

### Option A: If you have a SoArm 100 ROS2 package

```bash
# Source ROS2
source /opt/ros/jazzy/setup.bash

# Launch SoArm 100 driver (adjust package name as needed)
ros2 launch soarm_100 robot.launch.py

# Or start individual nodes
ros2 run soarm_100 joint_state_publisher
ros2 run soarm_100 command_subscriber
```

### Option B: Manual topic setup for testing

If you don't have a full driver, you can publish dummy data:

```bash
# Publish dummy joint states (for testing adapter)
ros2 topic pub /joint_state sensor_msgs/msg/JointState \
  "{header: {frame_id: 'base_link'}, \
    name: ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'gripper'], \
    position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
    velocity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}" \
  --rate 20
```

## Step 5: Start Camera Node

```bash
# If using USB camera
ros2 run usb_cam usb_cam_node_exe

# Or if using ROS camera package
ros2 run image_tools cam2image
```

Verify camera topic:
```bash
ros2 topic echo /camera/image_raw --no-arr
```

## Step 6: Start ROS Adapter

```bash
# Terminal 5
cd /home/mecha/hive_mind/hardware/ros_adapter
source /opt/ros/jazzy/setup.bash
export PYTHONPATH=/home/mecha/hive_mind:$PYTHONPATH

python3 ros_node.py
```

**Expected behavior:**
- Adapter subscribes to `/joint_state` and `/camera/image_raw`
- Sends HTTP requests to pipeline every 50ms (20 Hz)
- Publishes commands to `/joint_commands`

## Step 7: Monitor the System

### Check ROS topics:
```bash
# List active topics
ros2 topic list

# Monitor arm commands
ros2 topic echo /so_100_arm_controller/commands

# Monitor gripper commands
ros2 topic echo /so_100_arm_gripper_controller/commands
```

### Check Action service logs:
You should see periodic logs every 100 calls:
```
Scripted Controller Stats:
  Loop frequency: 20.00 Hz
  Avg command magnitude: 0.350
  Safety clamps (last 100): 0
```

## Step 8: Safety Monitoring

**IMPORTANT:** Keep your hand near the emergency stop!

Monitor for:
- ✅ Smooth, predictable motion
- ✅ No sudden jerks or jumps
- ✅ Commands within expected range
- ❌ Erratic behavior → STOP immediately
- ❌ High clamp counts → Reduce amplitude

## Step 9: Adjust Parameters

Once stable, you can experiment:

1. **Increase amplitude** (gradually):
   ```yaml
   amplitude: 0.1  # ~6 degrees
   ```

2. **Try different trajectories**:
   ```yaml
   trajectory_type: gripper_toggle  # Test gripper
   ```

3. **Increase frequency**:
   ```yaml
   frequency: 0.5  # Faster oscillation
   ```

## Troubleshooting

### No commands published
- Check pipeline is running: `curl http://localhost:8000/health`
- Verify ROS adapter sees data: Check logs for "Waiting for data..."
- Check topic names match your robot

### Robot not moving
- Verify SoArm 100 is subscribed to `/joint_commands`
- Check command format matches your robot's expectation
- May need to remap topic or adjust message type

### High latency warnings
- Pipeline timeout > 100ms
- Check service performance
- Reduce control loop frequency if needed

### Safety clamps triggered
- Commands exceeding joint limits
- Reduce amplitude or adjust `joint_limits` in config

## SoArm 100 Specific Notes

The SoArm 100 has:
- **5 DOF arm** (base, shoulder, elbow, wrist pitch, wrist roll)
- **1 DOF gripper**
- **Total: 6 DOF**
- **Joint Names**: `Shoulder_Rotation`, `Shoulder_Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Gripper` (Order matters for model!)
- **Communication**: ROS topics
- **Control modes**: Position control (default for scripted mode)

Configuration has been updated to handle the alphabetical sorting from `/joint_states` and reorder them correctly for the model.

You may need to:
1. Adjust topic names if your SoArm driver uses different conventions
2. Verify `action_dim: 6` in `services/action/config.yaml`

## Next Steps

Once scripted mode works reliably:
1. Switch to ML mode: `mode: learned` in config
2. Train Action Expert with real robot data
3. Test learned policies with same infrastructure

---

**Safety First:** Always start with conservative parameters and gradually increase complexity. Keep emergency stop accessible at all times.
