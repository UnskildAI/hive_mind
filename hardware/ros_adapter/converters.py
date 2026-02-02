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

def jointstate_to_robotstate(joint_msg, target_names: List[str] = None) -> RobotState:
    """
    Convert ROS JointState to RobotState.
    If target_names is provided, reorders the joint state to match that order.
    Handles NaN values by replacing them with 0.0.
    """
    import math
    
    def sanitize_value(val):
        """Replace NaN/Inf with 0.0"""
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    
    # Create a map of name -> index
    current_map = {name: i for i, name in enumerate(joint_msg.name)}
    
    if target_names:
        # Reorder based on target_names
        ordered_pos = []
        ordered_vel = []
        
        input_vel = list(joint_msg.velocity) if joint_msg.velocity else [0.0] * len(joint_msg.position)
        
        for name in target_names:
            if name in current_map:
                idx = current_map[name]
                ordered_pos.append(sanitize_value(joint_msg.position[idx]))
                ordered_vel.append(sanitize_value(input_vel[idx]))
            else:
                # Handle missing joint (e.g. if config has more joints than robot state?)
                # For safety, append 0.0 but this is a critical mismatch
                ordered_pos.append(0.0)
                ordered_vel.append(0.0)
                # Ideally log a warning here if we had a logger reference
                
        return RobotState(
            joint_position=ordered_pos,
            joint_velocities=ordered_vel,
            gripper_state=0.0 # This field might be deprecated if gripper is just a joint
        )
    else:
        # Fallback to raw order
        velocities = list(joint_msg.velocity) if joint_msg.velocity else [0.0] * len(joint_msg.position)
        
        # Sanitize all values
        clean_positions = [sanitize_value(p) for p in joint_msg.position]
        clean_velocities = [sanitize_value(v) for v in velocities]

        return RobotState(
            joint_position=clean_positions,
            joint_velocities=clean_velocities,
            gripper_state=0.0
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