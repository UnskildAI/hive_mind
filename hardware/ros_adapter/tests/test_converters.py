import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Inject mocks
from tests.ros_mocks import install_ros_mocks
install_ros_mocks()

from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from converters import image_to_perception, jointstate_to_robotstate, actionchunk_to_jointcmd
from std_msgs.msg import Float64MultiArray
from common.schemas.perception import PerceptionState
from common.schemas.robot_state import RobotState

def test_image_to_perception():
    # create dummy ros image
    bridge = CvBridge()
    cv_image = np.zeros((100, 100, 3), dtype=np.uint8)
    ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="rgb8")
    
    perception = image_to_perception(ros_image)
    
    assert isinstance(perception, PerceptionState)
    assert len(perception.scene_tokens) == 1
    assert len(perception.camera_pose) == 1
    assert len(perception.camera_pose[0]) == 6
    assert isinstance(perception.timestamp, float)

def test_jointstate_to_robotstate_full():
    msg = JointState()
    msg.position = [0.1, 0.2, 0.3]
    msg.velocity = [0.01, 0.02, 0.03]
    msg.name = ["j1", "j2", "j3"]
    
    robot_state = jointstate_to_robotstate(msg)
    
    assert isinstance(robot_state, RobotState)
    assert robot_state.joint_position == [0.1, 0.2, 0.3]
    assert robot_state.joint_velocities == [0.01, 0.02, 0.03]

def test_jointstate_to_robotstate_missing_velocity():
    msg = JointState()
    msg.position = [0.5, 0.6]
    msg.velocity = [] # Simulating missing velocity
    
    robot_state = jointstate_to_robotstate(msg)
    
    assert isinstance(robot_state, RobotState)
    assert robot_state.joint_position == [0.5, 0.6]
    # Should automatically fill with zeros
    assert robot_state.joint_velocities == [0.0, 0.0]

def test_actionchunk_to_jointcmd():
    action_data = {
        "schema_version": "1.0",
        "actions": [[0.1, 0.2, 0.3]],
        "horizon": 1,
        "control_mode": "position"
    }
    
    cmd = actionchunk_to_jointcmd(action_data)
    
    assert isinstance(cmd, Float64MultiArray)
    assert list(cmd.data) == [0.1, 0.2, 0.3]

def test_actionchunk_to_jointcmd_empty():
    action_data = {
        "schema_version": "1.0",
        "actions": [],
        "horizon": 1,
        "control_mode": "position"
    }
    
    cmd = actionchunk_to_jointcmd(action_data)
    
    assert isinstance(cmd, Float64MultiArray)
    assert len(cmd.data) == 0
