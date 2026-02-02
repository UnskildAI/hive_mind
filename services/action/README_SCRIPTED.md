# Scripted Controller Mode - Usage Guide

## Overview

The Action service now supports a **scripted controller mode** for validation without ML inference. This enables testing of timing, safety, networking, and executor behavior with predictable motion.

## Configuration

Edit `services/action/config.yaml`:

```yaml
mode: scripted  # Switch between "scripted" and "learned"

scripted:
  trajectory_type: joint_oscillation  # Options: "joint_oscillation", "square", "gripper_toggle"
  frequency: 0.5  # Hz for oscillation
  amplitude: 0.1  # radians for joint motion
  joint_limits: [-3.14, 3.14]  # Safety limits
  velocity_limit: 1.0  # rad/s
```

## Trajectory Types

### 1. Joint Oscillation
Sinusoidal motion on all joints with phase offsets for visual variety.
- Oscillates around current robot position
- Configurable frequency and amplitude
- Safe for continuous operation

### 2. Square Trajectory
Square path in joint space (first two joints).
- 4-second cycle time
- Moves through four corners
- Good for testing discrete motion

### 3. Gripper Toggle
Periodic gripper open/close while maintaining joint positions.
- 2-second toggle period
- Tests gripper control
- Minimal joint motion

## Safety Features

- **Joint Limits**: All actions clamped to configured limits
- **Logging**: Every 100 calls logs:
  - Loop frequency (Hz)
  - Average command magnitude
  - Safety clamp events

## Running the Service

```bash
# Start Action service in scripted mode
cd /home/mecha/hive_mind
uvicorn services.action.app:app --host 0.0.0.0 --port 8002
```

## Testing Standalone

```bash
# Run test script
python3 services/action/test_scripted.py
```

## Integration with Pipeline

The scripted controller accepts the same inputs as the ML model:
- `TaskLatent` (ignored in scripted mode)
- `PerceptionState` (ignored in scripted mode)
- `RobotState` (used for current position)

Returns valid `ActionChunk` with:
- Horizon: 10 steps
- Control mode: position
- Actions: List of joint positions

## Switching Back to ML Mode

Change `config.yaml`:
```yaml
mode: learned
```

Restart the service.
