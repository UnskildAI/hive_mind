import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Inject mocks
from tests.ros_mocks import install_ros_mocks
install_ros_mocks()

import pytest
import rclpy
from rclpy.node import Node
from unittest.mock import MagicMock, patch
from ros_node import RobotAdapter, PIPELINE_URL
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import numpy as np

# Mocking the requests.post to avoid hitting actual server during unit/integration test if desired
# But for "Integration" we might want to hit the actual mock server.
# Let's support both or default to hitting the mock server (which we can spawn).

@pytest.fixture
def mock_pipeline_server():
    # In a real heavy integration test, we might spawn the uvicorn process here.
    # For now, we assume the user/script runs the mock pipeline or we mock the requests library.
    # Given the instructions, let's mock the requests library to be self-contained and fast,
    # but the plan mentioned "Mock ROS + Pipeline", which could mean real integration.
    # Let's mock requests for reliability in this script, as spawning generic processes is flaky in tests.
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "schema_version": "1.0",
            "actions": [[0.5] * 6], # 6 DOF for SoArm 100
            "horizon": 1,
            "control_mode": "position"
        }
        mock_post.return_value = mock_response
        yield mock_post

def test_adapter_node_lifecycle(mock_pipeline_server):
    rclpy.init()
    try:
        node = RobotAdapter()
        
        # Verify subscriptions
        topic_names = [s.topic_name for s in node.subscriptions]
        assert '/camera/image_raw' in topic_names
        assert '/joint_states' in topic_names
        
        # Verify publishers
        pub_topics = [p.topic_name for p in node.publishers]
        assert '/so_100_arm_controller/commands' in pub_topics
        assert '/so_100_arm_gripper_controller/commands' in pub_topics
        
        node.destroy_node()
    finally:
        rclpy.shutdown()

def test_adapter_control_loop(mock_pipeline_server):
    rclpy.init()
    try:
        node = RobotAdapter()
        
        # Inject dummy data
        bridge = CvBridge()
        cv_img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_msg = bridge.cv2_to_imgmsg(cv_img, encoding="rgb8")
        
        # Create out-of-order joint state to test sorting
        joint_msg = JointState()
        # Alphabetical order from user echo: Elbow, Gripper, Shoulder_Pitch, Shoulder_Rotation, Wrist_Pitch, Wrist_Roll
        joint_msg.name = ['Elbow', 'Gripper', 'Shoulder_Pitch', 'Shoulder_Rotation', 'Wrist_Pitch', 'Wrist_Roll']
        joint_msg.position = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] 
        # Config expectation: Shoulder_Rotation, Shoulder_Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Gripper
        # Expected values passed to robot state: [0.4, 0.3, 0.1, 0.5, 0.6, 0.2]
        
        joint_msg.velocity = [0.0] * 6
        
        # Manually trigger callbacks
        node.image_callback(img_msg)
        node.joint_callback(joint_msg)
        
        # Mock the publishers in the map
        for pub in node.publishers_map.values():
            pub.publish = MagicMock()
        
        # Trigger control loop
        node.control_loop()
        
        # Verify pipeline was called
        mock_pipeline_server.assert_called_once()
        
        # Verify command was published to both controllers
        # Action is [0.5]*6. 
        # Arm (0-5) should get 0.5*5
        # Gripper (5-6) should get 0.5*1
        
        arm_pub = node.publishers_map['arm']
        arm_pub.publish.assert_called_once()
        arm_msg = arm_pub.publish.call_args[0][0]
        assert len(arm_msg.data) == 5
        assert list(arm_msg.data) == [0.5] * 5
        
        gripper_pub = node.publishers_map['gripper']
        gripper_pub.publish.assert_called_once()
        gripper_msg = gripper_pub.publish.call_args[0][0]
        assert len(gripper_msg.data) == 1
        assert list(gripper_msg.data) == [0.5]
        
        node.destroy_node()
    finally:
        rclpy.shutdown()
