import numpy as np
from cv_bridge import CvBridge
from common.schemas.perception import PerceptionState
from common.schemas.robot_state import RobotState
from common.schemas.action import ActionChunk
import time
from typing import List
from std_msgs.msg import Float64MultiArray

bridge = CvBridge()

def image_to_perception(img_msg) -> PerceptionState:
    cv_img = bridge.imgmsg_to_cv2(img_msg, "rgb8")

    # Minimal example - real system should:
    # - resize
    # - normalize
    # - maybe compress

    # Dummy token extraction (calculating mean color as a placeholder feature)
    tokens = cv_img.mean(axis=(0,1)).tolist()

    return PerceptionState(
        schema_version="1.0",
        scene_tokens=[tokens],
        # Fixed: Schema uses camera_pose, not spatial_tokens
        camera_pose=[[0.0] * 6], # Placeholder identity pose
        timestamp=time.time()
    )

def jointstate_to_robotstate(joint_msg) -> RobotState:
    # Handle missing velocity
    velocities = list(joint_msg.velocity) if joint_msg.velocity else [0.0] * len(joint_msg.position)

    return RobotState(
        joint_position=list(joint_msg.position),
        joint_velocities=velocities,
        gripper_state=0.0 # Placeholder, assuming separate handling or integrated if in joint_msg
    )

def actionchunk_to_jointcmd(action_json: dict) -> Float64MultiArray:
    # Validate with schema first
    action_chunk = ActionChunk(**action_json)
    
    # Extract first action from the chunk (model might predict horizon > 1)
    if not action_chunk.actions:
        return Float64MultiArray() # Return empty msg if no actions

    action = action_chunk.actions[0]

    msg = Float64MultiArray()
    msg.data = action

    return msg