#!/usr/bin/env python3
"""
Simple camera publisher for ROS2.
Reads from /dev/video0 and publishes to /camera/image_raw
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys


class CameraPublisher(Node):
    def __init__(self, device_id=0, fps=30):
        super().__init__('camera_publisher')
        
        # Create publisher
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Open camera with V4L2 backend specifically for Linux
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera device {device_id} with V4L2")
            # Try default if V4L2 fail
            self.cap = cv2.VideoCapture(device_id)
            if not self.cap.isOpened():
                sys.exit(1)
        
        # Set MJPG format which is often more compatible for USB bandwidth
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Get actual camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.get_logger().info(f"Camera opened: {width}x{height} @ {actual_fps} FPS")
        self.get_logger().info(f"Publishing to /camera/image_raw")
        
        # Create timer for publishing
        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.frame_count = 0

    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warn("Failed to read frame from camera", throttle_duration_sec=1.0)
            return
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to ROS Image message
        try:
            msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding="rgb8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_frame"
            
            self.publisher.publish(msg)
            
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                self.get_logger().info(f"Published {self.frame_count} frames")
                
        except Exception as e:
            self.get_logger().error(f"Error converting/publishing frame: {e}")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # Parse command line arguments for device ID
    import argparse
    parser = argparse.ArgumentParser(description='Camera Publisher Node')
    parser.add_argument('--device', type=int, default=0, 
                        help='Camera device ID (default: 0 for /dev/video0)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Publishing frame rate (default: 30)')
    
    # Parse only known args to avoid conflicts with ROS args
    parsed_args, _ = parser.parse_known_args()
    
    node = CameraPublisher(device_id=parsed_args.device, fps=parsed_args.fps)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
