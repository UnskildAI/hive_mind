#!/usr/bin/env python3
"""Test script for scripted controller."""

import sys
sys.path.append("/home/mecha/hive_mind")

from services.action.policy import ScriptedController
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent
from common.schemas.robot_state import RobotState
import yaml
import time

# Load config
with open("/home/mecha/hive_mind/services/action/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize controller
controller = ScriptedController(config)

# Create dummy inputs
perception = PerceptionState(
    schema_version="1.0",
    scene_tokens=[[0.0] * 64],
    camera_pose=[[0.0] * 6],
    timestamp=time.time()
)

task = TaskLatent(
    schema_version="1.0",
    goal_embedding=[0.0] * 256,
    task_id="test_scripted",
    constraints={},
    subtask_id="0",
    confidence=1.0
)

robot = RobotState(
    joint_position=[0.0] * 7,
    joint_velocities=[0.0] * 7,
    gripper_state=0.0
)

print("Testing scripted controller...")
print(f"Trajectory type: {controller.trajectory_type}")
print(f"Frequency: {controller.frequency} Hz")
print(f"Amplitude: {controller.amplitude} rad")
print()

# Test multiple calls to see logging
for i in range(150):
    action_chunk = controller.act(task, perception, robot)
    
    if i == 0:
        print(f"First action chunk:")
        print(f"  Horizon: {action_chunk.horizon}")
        print(f"  Control mode: {action_chunk.control_mode}")
        print(f"  First action: {action_chunk.actions[0]}")
        print()
    
    # Simulate 20Hz control loop
    time.sleep(0.05)

print("\nTest complete!")
