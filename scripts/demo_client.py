import requests
import base64
import json
import numpy as np
import time
from PIL import Image
import io

# Configuration
PIPELINE_URL = "http://localhost:8000"

def generate_dummy_image():
    # Create a random RGB image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def main():
    print(f"Connecting to Pipeline at {PIPELINE_URL}...")
    
    image_b64 = generate_dummy_image()
    camera_pose = [[1.0, 0.0, 0.0, 0.0] for _ in range(1)] # Dummy pose
    instruction = "Pick up the blue cube"
    robot_state = {
        "joint_position": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "joint_velocities": [0.0] * 7,
        "gripper_state": 0.0
    }
    
    payload = {
        "image_base64": image_b64,
        "camera_pose": camera_pose,
        "instruction": instruction,
        "robot_state": robot_state
    }
    
    try:
        start_time = time.time()
        print("Sending request...")
        response = requests.post(f"{PIPELINE_URL}/run_step", json=payload)
        response.raise_for_status()
        
        result = response.json()
        duration = time.time() - start_time
        
        print(f"Success! Request took {duration:.3f}s")
        print("Received ActionChunk:")
        print(json.dumps(result, indent=2))
        
        # Verify basic structure
        assert "actions" in result
        assert len(result["actions"]) > 0
        print("Verification Successful.")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Pipeline service. Is it running?")
        print("Try running: docker-compose up --build")
    except Exception as e:
        print(f"Error: {e}")
        if 'response' in locals():
            print(f"Response content: {response.content}")

if __name__ == "__main__":
    main()
