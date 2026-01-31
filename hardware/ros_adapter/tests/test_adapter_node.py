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
            "actions": [[0.5] * 7],
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
        assert any(s.topic_name == '/camera/image_raw' for s in node.subscriptions)
        assert any(s.topic_name == '/joint_state' for s in node.subscriptions)
        
        # Verify publishers
        assert any(p.topic_name == '/joint_commands' for p in node.publishers)
        
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
        
        joint_msg = JointState()
        joint_msg.position = [0.0] * 7
        joint_msg.velocity = [0.0] * 7
        
        # Manually trigger callbacks
        node.image_callback(img_msg)
        node.joint_callback(joint_msg)
        
        # Create a mock publisher to check if command was published
        # Since we can't easily snoop on the real publisher without another node,
        # let's mock the publisher's publish method.
        node.cmd_pub.publish = MagicMock()
        
        # Trigger control loop
        node.control_loop()
        
        # Verify pipeline was called
        mock_pipeline_server.assert_called_once()
        assert mock_pipeline_server.call_args[0][0] == PIPELINE_URL
        
        # Verify command was published
        node.cmd_pub.publish.assert_called_once()
        published_msg = node.cmd_pub.publish.call_args[0][0]
        assert isinstance(published_msg, Float64MultiArray)
        assert list(published_msg.data) == [0.5] * 7
        
        node.destroy_node()
    finally:
        rclpy.shutdown()
