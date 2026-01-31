import argparse
import sys
import os
import time

# Adjust path to find local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import rclpy
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
except ImportError:
    print("ROS2 libraries not found. Ensure you have sourced ROS2 setup.bash.")
    sys.exit(1)

from ros_node import RobotAdapter
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

class OfflineAdapterTest:
    def __init__(self, bag_path):
        self.bag_path = bag_path
        self.adapter = RobotAdapter()
        
        # Mock publisher to capture output
        self.published_commands = []
        self.adapter.cmd_pub.publish = self.mock_publish

    def mock_publish(self, msg):
        self.published_commands.append(msg)
        # print(f"Published command: {msg.data}")

    def run(self):
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {t.name: t.type for t in topic_types}

        print(f"Reading from {self.bag_path}...")
        
        count = 0
        
        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            if topic == '/camera/image_raw':
                self.adapter.image_callback(msg)
            elif topic == '/joint_state':
                self.adapter.joint_callback(msg)
            
            # Trigger control loop manually after updates
            # In real execution, this runs on a timer.
            # Here we can trigger it every N messages or on specific conditions.
            # For deterministic replay, let's trigger it when we have both?
            # Or just trigger it every step and let the adapter handle 'None' checks.
            self.adapter.control_loop()
            
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} messages...")

        print(f"Finished. Total published commands: {len(self.published_commands)}")
        
        # Validation checks
        if len(self.published_commands) == 0:
            print("FAIL: No commands published during playback.")
            sys.exit(1)
        else:
            print("PASS: Commands generated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run offline rosbag test for adapter")
    parser.add_argument("bag", help="Path to rosbag directory/file")
    args = parser.parse_args()

    rclpy.init()
    tester = OfflineAdapterTest(args.bag)
    tester.run()
    rclpy.shutdown()
