import pytest
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent
from common.schemas.action import ActionChunk
from common.schemas.robot_state import RobotState
import time

def test_perception_schema():
    data = {
        "schema_version": "1.0.0",
        "scene_tokens": [[0.1, 0.2], [0.3, 0.4]],
        "camera_pose": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "timestamp": time.time()
    }
    p = PerceptionState(**data)
    assert p.schema_version == "1.0.0"
    assert len(p.scene_tokens) == 2

def test_task_schema():
    data = {
        "schema_version": "1.0.0",
        "goal_embedding": [0.1] * 256,
        "constraints": {"speed": "fast"},
        "subtask_id": "test_id",
        "confidence": 0.95
    }
    t = TaskLatent(**data)
    assert t.subtask_id == "test_id"

def test_action_schema():
    data = {
        "schema_version": "1.0.0",
        "actions": [[0.0]*7]*10,
        "horizon": 10,
        "control_mode": "position"
    }
    a = ActionChunk(**data)
    assert len(a.actions) == 10

def test_robot_state_schema():
    data = {
        "joint_position": [0.0]*7,
        "joint_velocities": [0.0]*7,
        "gripper_state": 1.0
    }
    r = RobotState(**data)
    assert r.gripper_state == 1.0

if __name__ == "__main__":
    test_perception_schema()
    test_task_schema()
    test_action_schema()
    test_robot_state_schema()
    print("All schema tests passed!")
