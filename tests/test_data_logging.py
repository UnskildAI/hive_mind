import requests
import base64
import time
import os
import shutil
from pathlib import Path

def test_data_logging():
    url = "http://localhost:8000/run_step"
    log_dir = Path("/home/mecha/hive_mind/data/logs")
    
    # Clean up old logs for clean test
    # if log_dir.exists():
    #    shutil.rmtree(log_dir)
    
    # Mock data
    image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    payload = {
        "image_base64": image_base64,
        "camera_pose": [[0.0] * 6],
        "instruction": "move to target",
        "robot_state": {
            "joint_position": [0.0] * 6,
            "joint_velocities": [0.0] * 6,
            "gripper_state": 0.0
        }
    }
    
    print("Sending 5 steps to trigger logging...")
    for i in range(5):
        r = requests.post(url, json=payload)
        r.raise_for_status()
        time.sleep(0.1)
    
    print("Checking for log directory...")
    time.sleep(2) # Give worker thread time to write
    
    if not log_dir.exists():
        print("FAIL: Log directory not created")
        return

    chunks = list(log_dir.glob("chunk_*"))
    if not chunks:
        print("FAIL: No chunks found")
        return
    
    print(f"SUCCESS: Found {len(chunks)} log chunks.")
    
    for chunk in chunks:
        steps_file = chunk / "steps.jsonl"
        images_dir = chunk / "images"
        
        if steps_file.exists():
            with open(steps_file, "r") as f:
                lines = f.readlines()
                print(f"Chunk {chunk.name}: {len(lines)} steps logged.")
        
        if images_dir.exists():
            images = list(images_dir.glob("*.jpg"))
            print(f"Chunk {chunk.name}: {len(images)} images saved.")

if __name__ == "__main__":
    test_data_logging()
