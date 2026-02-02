import requests
import time
import numpy as np

def test_vlm_stability():
    url = "http://localhost:8003/process"
    perception = {
        "schema_version": "1.0.0",
        "scene_tokens": [[0.5] * 64] * 16,
        "camera_pose": [[0.0] * 6],
        "timestamp": time.time()
    }
    
    payload = {
        "perception": perception,
        "instruction": "pick up the red cube"
    }

    print("--- Phase 1: Frequency Check ---")
    start = time.time()
    for i in range(10):
        r = requests.post(url, json=payload)
        r.raise_for_status()
    end = time.time()
    
    # Since we are sending very fast, it should hit cache for almost all.
    # The actual processing happens only once or twice depending on timing.
    # The total time should be very low if cache works.
    print(f"Time for 10 requests: {end - start:.4f}s (Cache should be fast)")

    print("\n--- Phase 2: Subtask Stability Check ---")
    subtask_ids = []
    for i in range(5):
        r = requests.post(url, json=payload)
        data = r.json()
        subtask_ids.append(data["subtask_id"])
        time.sleep(0.5) # Still within 1.0s TTL
    
    print(f"Subtask IDs (Steady State): {subtask_ids}")
    if len(set(subtask_ids)) == 1:
        print("SUCCESS: Subtask ID is stable.")
    else:
        print("FAIL: Subtask ID flickered.")

    print("\n--- Phase 3: Human Override Check ---")
    payload2 = {
        "perception": perception,
        "instruction": "STOP"
    }
    r1 = requests.post(url, json=payload)
    id1 = r1.json()["subtask_id"]
    
    r2 = requests.post(url, json=payload2) # Immediate instruction change
    id2 = r2.json()["subtask_id"]
    
    print(f"ID 1: {id1}")
    print(f"ID 2: {id2}")
    if id1 != id2:
        print("SUCCESS: Instruction change triggered override.")
    else:
        print("FAIL: Instruction change ignored (stuck in cache).")

if __name__ == "__main__":
    test_vlm_stability()
