#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import requests
import time

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, Bool

from converters import (
    image_to_perception,
    jointstate_to_robotstate,
    actionchunk_to_jointcmd
)
from safety_validator import SafetyValidator

PIPELINE_URL = "http://localhost:8000/run_step"

class RobotAdapter(Node):

    def __init__(self):
        super().__init__("robot_adapter")
        
        # Load configuration
        import yaml
        import os
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.latest_image = None
        self.latest_joint = None
        
        topics = self.config.get("topics", {})
        self.create_subscription(
            Image, 
            topics.get("camera", "/camera/image_raw"), 
            self.image_callback, 
            10
        )
        self.create_subscription(
            JointState, 
            topics.get("joint_state", "/joint_states"), 
            self.joint_callback, 
            10
        )
        
        # Emergency stop subscriber
        safety_config = self.config.get("safety", {})
        estop_topic = safety_config.get("emergency_stop_topic", "/emergency_stop")
        self.create_subscription(
            Bool,
            estop_topic,
            self.emergency_stop_callback,
            10
        )
        
        # Initialize safety validator
        self.safety_validator = SafetyValidator(self.config)

        # Load controller config
        self.controllers_config = self.config.get("controllers", {})
        self.publishers_map = {}
        self.joint_indices = {}
        
        # Create a publisher for each controller
        # And map joint names to their global index for splitting commands
        global_joint_idx = 0
        all_joint_names = []
        
        for ctrl_name, ctrl_cfg in self.controllers_config.items():
            topic = ctrl_cfg["topic"]
            self.publishers_map[ctrl_name] = self.create_publisher(Float64MultiArray, topic, 10)
            
            # Map valid indices for this controller
            joints = ctrl_cfg["joints"]
            start_idx = global_joint_idx
            end_idx = global_joint_idx + len(joints)
            
            self.joint_indices[ctrl_name] = list(range(start_idx, end_idx))
            
            all_joint_names.extend(joints)
            global_joint_idx += len(joints)
            
            self.get_logger().info(f"Initialized controller '{ctrl_name}' on topic '{topic}' with joints {joints}")

        self.all_joint_names = all_joint_names
        self.get_logger().info(f"Total expected joints: {len(self.all_joint_names)}")

        self.timer = self.create_timer(
            self.config.get("control", {}).get("frequency_hz", 20) ** -1, # default 50ms
            self.control_loop
        )

    def image_callback(self, msg):
        self.latest_image = msg
    
    def joint_callback(self, msg):
        self.latest_joint = msg
    
    def emergency_stop_callback(self, msg):
        """Handle emergency stop messages"""
        self.safety_validator.set_emergency_stop(msg.data)
    
    def control_loop(self):
        # Strict checking for both sensors
        if self.latest_image is None or self.latest_joint is None:
            # self.get_logger().info("Waiting for data...", throttle_duration_sec=2.0)
            return
        
        try:
            # Convert image to base64
            import base64
            from cv_bridge import CvBridge
            bridge = CvBridge()
            cv_img = bridge.imgmsg_to_cv2(self.latest_image, "rgb8")
            _, buffer = __import__('cv2').imencode('.jpg', cv_img)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Pass target names to ensure correct sorting/filtering
            robot_state = jointstate_to_robotstate(self.latest_joint, target_names=self.all_joint_names)

            # Pipeline expects: image_base64, camera_pose, instruction, robot_state
            payload = {
                "image_base64": image_base64,
                "camera_pose": [[0.0] * 6],  # Placeholder camera pose
                "instruction": "move to target",  # Placeholder instruction
                "robot_state": robot_state.model_dump()
            }

            # Using timeout from config to prevent blocking control loop
            url = self.config.get("pipeline", {}).get("url", PIPELINE_URL)
            timeout_ms = self.config.get("pipeline", {}).get("timeout_ms", 100)
            r = requests.post(url, json=payload, timeout=timeout_ms / 1000.0)
            r.raise_for_status()
            action_chunk = r.json()
            
            # Extract the raw action list (assuming horizon 1 for execution or just taking first step)
            # The ActionChunk schema has 'actions': List[List[float]]
            actions = action_chunk.get("actions", [])
            if not actions:
                return
                
            current_action = actions[0] # [First Step]
            
            # SAFETY VALIDATION
            is_safe, safe_action, reason = self.safety_validator.validate_command(current_action)
            if not is_safe:
                self.get_logger().error(f"SAFETY VIOLATION: {reason} - Command blocked")
                return
            
            if reason != "safe":
                self.get_logger().warning(f"Safety correction applied: {reason}")
                current_action = safe_action
            
            # Split and Publish
            for ctrl_name, pub in self.publishers_map.items():
                indices = self.joint_indices[ctrl_name]
                
                # Safety check
                if max(indices) >= len(current_action):
                    self.get_logger().error(f"Action dimension ({len(current_action)}) smaller than required index ({max(indices)})")
                    continue

                cmd_msg = Float64MultiArray()
                cmd_msg.data = [current_action[i] for i in indices]
                pub.publish(cmd_msg)

        except requests.exceptions.Timeout:
            timeout_ms = self.config.get("pipeline", {}).get("timeout_ms", 100)
            self.get_logger().warn(f"Pipeline timeout (latency > {timeout_ms}ms)", throttle_duration_sec=1.0)
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