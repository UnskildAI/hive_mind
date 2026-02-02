# Quick Start: SoArm 100 + Scripted Controller

## Pre-flight Checklist

- [ ] ROS2 Jazzy sourced: `source /opt/ros/jazzy/setup.bash`
- [ ] SoArm 100 connected and powered
- [ ] Camera connected
- [ ] Emergency stop accessible

## 1. Configure (Safety First!)

Edit `services/action/config.yaml`:
```yaml
mode: scripted
scripted:
  trajectory_type: joint_oscillation
  frequency: 0.3        # Low for safety
  amplitude: 0.05       # Small movements (~3°)
```

## 2. Start Services (4 terminals)

```bash
# T1: Perception
.venv/bin/python -m uvicorn services.perception.app:app --port 8001

# T2: Task  
.venv/bin/python -m uvicorn services.task.app:app --port 8003

# T3: Action (verify "ScriptedController initialized")
.venv/bin/python -m uvicorn services.action.app:app --port 8002

# T4: Pipeline
.venv/bin/python -m uvicorn services.pipeline.app:app --port 8000
```

## 3. Start ROS Nodes

```bash
# T5: SoArm 100 driver (adjust command for your setup)
ros2 launch soarm_100 robot.launch.py

# T6: Camera
ros2 run usb_cam usb_cam_node_exe
```

## 4. Start ROS Adapter

```bash
# T7
cd hardware/ros_adapter
source /opt/ros/jazzy/setup.bash
export PYTHONPATH=/home/mecha/hive_mind:$PYTHONPATH
python3 ros_node.py
```

## 5. Monitor

```bash
# Watch commands
ros2 topic echo /joint_commands

# Check frequency
ros2 topic hz /joint_commands  # Should be ~20 Hz
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No commands | Check `curl http://localhost:8000/health` |
| Robot not moving | Verify topic: `ros2 topic list \| grep joint` |
| High latency | Check Action service logs |
| Erratic motion | **STOP!** Reduce amplitude/frequency |

## Safety

- Start with **amplitude: 0.05** (3°)
- Keep hand near **emergency stop**
- Monitor for smooth motion
- Gradually increase parameters

See [SOARM_100_INTEGRATION.md](file:///home/mecha/hive_mind/hardware/ros_adapter/SOARM_100_INTEGRATION.md) for detailed guide.
