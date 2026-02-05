#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import requests
import time
import numpy as np

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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
        self.step_count = 0
        self.last_published_action = None
        self.smoothing_alpha = self.config.get("control", {}).get("smoothing_alpha", 0.3)
        
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
            self.publishers_map[ctrl_name] = self.create_publisher(JointTrajectory, topic, 10)
            
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
            if self.latest_image is None:
                self.get_logger().warn("Waiting for image data...", throttle_duration_sec=5.0)
            if self.latest_joint is None:
                self.get_logger().warn("Waiting for joint data...", throttle_duration_sec=5.0)
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
            
            current_qpos = robot_state.joint_position
            # self.get_logger().info(f"Current QPos: {current_qpos}")

            # Pipeline expects: image_base64, camera_pose, instruction, robot_state
            payload = {
                "image_base64": image_base64,
                "camera_pose": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.0, 1.0]],
                "instruction": None,  # Let server use the instruction set via /set_instruction
                "robot_state": robot_state.model_dump()
            }

            # Using timeout from config to prevent blocking control loop
            url = self.config.get("pipeline", {}).get("url", PIPELINE_URL)
            timeout_ms = self.config.get("pipeline", {}).get("timeout_ms", 1000)
            
            start_time = time.time()
            r = requests.post(url, json=payload, timeout=timeout_ms / 1000.0)
            latency_ms = (time.time() - start_time) * 1000
            
            r.raise_for_status()
            action_chunk = r.json()
            
            # Extract the raw action list
            actions = action_chunk.get("actions", [])
            if not actions:
                self.get_logger().warn("Received empty actions from pipeline")
                return
                
            current_action = actions[0] # [First Step]
            
            # MOTION SMOOTHING (EMA Filter)
            if self.last_published_action is not None:
                # alpha * new + (1-alpha) * old
                smoothed_action = []
                for i in range(len(current_action)):
                    val = self.smoothing_alpha * current_action[i] + (1 - self.smoothing_alpha) * self.last_published_action[i]
                    smoothed_action.append(val)
                current_action = smoothed_action
            
            self.last_published_action = current_action
            # self.get_logger().info(f"Model action (first step): {current_action}")
            
            # SAFETY VALIDATION
            is_safe, safe_action, reason = self.safety_validator.validate_command(current_action)
            
            if not is_safe:
                self.get_logger().error(f"CRITICAL SAFETY BLOCK: {reason}")
                return
            
            if reason != "safe":
                # self.get_logger().warning(f"Safety adjustment: {reason}")
                current_action = safe_action
            
            # Split and Publish
            published_any = False
            for ctrl_name, pub in self.publishers_map.items():
                indices = self.joint_indices[ctrl_name]
                
                # Safety check
                if max(indices) >= len(current_action):
                    self.get_logger().error(f"Action dim mismatch for {ctrl_name}: {len(current_action)} < {max(indices)}")
                    continue

                # Create JointTrajectory message
                cmd_msg = JointTrajectory()
                cmd_msg.joint_names = [self.all_joint_names[i] for i in indices]
                
                # Point 1: START (Exactly where the robot is RIGHT NOW)
                # This ensures the trajectory is continuous from the actual hardware state
                start_point = JointTrajectoryPoint()
                start_point.positions = [float(current_qpos[i]) for i in indices]
                start_point.time_from_start.nanosec = 0
                
                # Point 2: GOAL (Where we want it to be in one loop period)
                goal_point = JointTrajectoryPoint()
                goal_point.positions = [float(current_action[i]) for i in indices]
                
                # Match interpolation time to the loop period exactly
                # For 10Hz (100ms), we set 100ms.
                # If we want it even smoother, we can use 150ms to slightly overlap.
                period_ms = int((1.0 / self.config.get("control", {}).get("frequency_hz", 10)) * 1000)
                goal_point.time_from_start.nanosec = period_ms * 1000000 
                
                cmd_msg.points = [start_point, goal_point]
                
                # Check for "Near Zero" movement
                diff = np.abs(np.array(goal_point.positions) - np.array(start_point.positions))
                if np.mean(diff) < 0.001:
                    pass
                
                pub.publish(cmd_msg)
                published_any = True
                
            if published_any:
                # SIDE-BY-SIDE TELEMETRY
                # Compare current_qpos with current_action
                deltas = []
                comparison_str = []
                for i, name in enumerate(self.all_joint_names):
                    curr = current_qpos[i]
                    target = current_action[i]
                    delta = target - curr
                    deltas.append(abs(delta))
                    comparison_str.append(f"{name}: {curr:.3f}->{target:.3f} (Δ={delta:.3f})")
                
                max_delta = max(deltas) if deltas else 0.0
                
                # Log a summary every second
                self.get_logger().info(f"Pipeline Step: Latency={latency_ms:.1f}ms, Max Δ={max_delta:.4f}")
                
                # Log detailed comparison every 5 seconds or if movement is "meaningful"
                if max_delta > 0.05: # > ~3 degrees
                    self.get_logger().info("MEANINGFUL MOVEMENT DETECTED:")
                    for line in comparison_str:
                        self.get_logger().info(f"  {line}")
                elif self.step_count % 100 == 0: # Periodic check for static/near-static
                    self.get_logger().info("STATIONARY/MICRO-MOVEMENT:")
                    for line in comparison_str:
                        self.get_logger().info(f"  {line}")
            
            self.step_count += 1

        except requests.exceptions.Timeout:
            self.get_logger().warn("Pipeline timeout", throttle_duration_sec=2.0)
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Pipeline connection error: {e}", throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f"Loop error: {e}", throttle_duration_sec=1.0)
            import traceback
            self.get_logger().error(traceback.format_exc(), throttle_duration_sec=5.0)

def main():
    rclpy.init()
    node = RobotAdapter()
    rclpy.spin(node)

if __name__ == "__main__":
    main()