import sys
from unittest.mock import MagicMock

# Helper to register mocks in sys.modules
def install_ros_mocks():
    if "rclpy" in sys.modules:
        return
    
    # Mock specific message classes we use
    class MockImage:
        def __init__(self):
            self.header = MagicMock()
            self.height = 0
            self.width = 0
            self.encoding = "rgb8"
            self.is_bigendian = 0
            self.step = 0
            self.data = []

    class MockJointState:
        def __init__(self, position=None, velocity=None, name=None):
            self.header = MagicMock()
            self.name = name or []
            self.position = position or []
            self.velocity = velocity or []
            self.effort = []

    class MockFloat64MultiArray:
        def __init__(self):
            self.data = []

    # Mock Modules
    mock_sensor_msgs = MagicMock()
    mock_sensor_msgs.msg.Image = MockImage
    mock_sensor_msgs.msg.JointState = MockJointState
    
    mock_std_msgs = MagicMock()
    mock_std_msgs.msg.Float64MultiArray = MockFloat64MultiArray
    
    # Mock Node class
    class MockNode:
        def __init__(self, node_name):
            self.node_name = node_name
            self.publishers = []
            self.subscriptions = []
            self.timers = []
        
        def create_subscription(self, msg_type, topic, callback, qos_profile):
            sub = MagicMock()
            sub.topic_name = topic
            self.subscriptions.append(sub)
            return sub
            
        def create_publisher(self, msg_type, topic, qos_profile):
            pub = MagicMock()
            pub.topic_name = topic
            self.publishers.append(pub)
            return pub
            
        def create_timer(self, period, callback):
            timer = MagicMock()
            self.timers.append(timer)
            return timer
            
        def get_logger(self):
            return MagicMock()
            
        def destroy_node(self):
            pass

    mock_rclpy = MagicMock()
    mock_rclpy.node.Node = MockNode
    
    # Mock CvBridge
    class MockCvBridge:
        def imgmsg_to_cv2(self, msg, encoding="passive"):
            import numpy as np
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        def cv2_to_imgmsg(self, cv_img, encoding="passive"):
            return MockImage()
            
    mock_cv_bridge = MagicMock()
    mock_cv_bridge.CvBridge = MockCvBridge

    # Inject into sys.modules
    sys.modules["rclpy"] = mock_rclpy
    sys.modules["rclpy.node"] = mock_rclpy.node
    sys.modules["sensor_msgs"] = mock_sensor_msgs
    sys.modules["sensor_msgs.msg"] = mock_sensor_msgs.msg
    sys.modules["std_msgs"] = mock_std_msgs
    sys.modules["std_msgs.msg"] = mock_std_msgs.msg
    sys.modules["cv_bridge"] = mock_cv_bridge
