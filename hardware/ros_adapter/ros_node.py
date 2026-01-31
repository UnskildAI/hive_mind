#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import requests
import time

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

from converters import (
    image_to_perception,
    jointstate_to_robotstate,
    actionchunk_to_jointcmd
)

PIPELINE_URL = "http://localhost:8000/infer"

class RobotAdapter(Node):

    def __init__(self):
        super().__init__("robot_adapter")

        self.latest_image = None
        self.latest_joint = None

        self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)

        self.create_subscription(JointState, "/joint_state", self.joint_callback, 10)

        self.cmd_pub = self.create_publisher(Float64MultiArray, "/joint_commands", 10)

        self.timer = self.create_timer(
            0.05, #20Hz Loop
            self.control_loop
        )

    def image_callback(self, msg):
        self.latest_image = msg
    
    def joint_callback(self, msg):
        self.latest_joint = msg
    
    def control_loop(self):
        # Strict checking for both sensors
        if self.latest_image is None or self.latest_joint is None:
            # self.get_logger().info("Waiting for data...", throttle_duration_sec=2.0)
            return
        
        try:
            perception = image_to_perception(self.latest_image)
            robot_state = jointstate_to_robotstate(self.latest_joint)

            payload = {
                "perception": perception.model_dump(),
                "robot": robot_state.model_dump()
            }

            # Using a short timeout to prevent blocking control loop
            r = requests.post(PIPELINE_URL, json=payload, timeout=0.1)
            r.raise_for_status()
            action = r.json()
            
            cmd = actionchunk_to_jointcmd(action)
            self.cmd_pub.publish(cmd)

        except requests.exceptions.Timeout:
            self.get_logger().warn("Pipeline timeout (latency > 100ms)", throttle_duration_sec=1.0)
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Pipeline communication error: {e}", throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f"Unexpected error in control_loop: {e}", throttle_duration_sec=1.0)

def main():
    rclpy.init()
    node = RobotAdapter()
    rclpy.spin(node)

if __name__ == "__main__":
    main()