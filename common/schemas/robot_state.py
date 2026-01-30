from pydantic import BaseModel
from typing import List

class RobotState(BaseModel):
    joint_position: List[float]
    joint_velocities: List[float]
    gripper_state: float