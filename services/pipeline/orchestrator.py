import requests
import time
import json
import logging
import base64
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent
from common.schemas.action import ActionChunk
from common.schemas.robot_state import RobotState

class PipelineOrchestrator:
    def __init__(self, perception_url, task_url, action_url):
        self.perception_url = perception_url
        self.task_url = task_url
        self.action_url = action_url
        
        self.logger = logging.getLogger("orchestrator")
        
        # State
        self.last_task_latent = None
        self.last_task_time = 0
        self.task_ttl = 0.5 # 2 Hz
        self.current_instruction = None
        
        # Replay logging
        self.replay_log = []

    def log_step(self, perception, task, action):
        entry = {
            "timestamp": time.time(),
            "perception": perception.dict(),
            "task": task.dict(),
            "action": action.dict()
        }
        self.replay_log.append(entry)
        # In real system, write to disk async

    def run_step(self, image_base64: str, camera_pose: list, instruction: str, robot_state: dict) -> ActionChunk:
        # 1. Perception
        try:
            p_payload = {"image_base64": image_base64, "camera_pose": camera_pose}
            resp = requests.post(f"{self.perception_url}/perceive", json=p_payload, timeout=0.5)
            resp.raise_for_status()
            perception_data = resp.json()
            perception = PerceptionState(**perception_data)
        except Exception as e:
            self.logger.error(f"Perception failed: {e}")
            raise e

        # 2. Task
        # Check if we need to call task
        now = time.time()
        should_call_task = (
            self.last_task_latent is None or 
            instruction != self.current_instruction or 
            (now - self.last_task_time) > self.task_ttl
        )
        
        task = self.last_task_latent
        
        if should_call_task:
            try:
                t_payload = {"perception": perception.dict(), "instruction": instruction}
                resp = requests.post(f"{self.task_url}/process", json=t_payload, timeout=1.0) # Longer timeout for task
                if resp.status_code == 200:
                    task_data = resp.json()
                    task = TaskLatent(**task_data)
                    self.last_task_latent = task
                    self.last_task_time = now
                    self.current_instruction = instruction
                else:
                    self.logger.warning(f"Task service returned {resp.status_code}, using old latent")
            except Exception as e:
                self.logger.warning(f"Task call failed: {e}, using old latent")
                # Graceful degradation: use last known latent if available
                if self.last_task_latent is None:
                    raise RuntimeError("Task service unavailable and no cached latent")
        
        # 3. Action
        try:
            # Construct RobotState from dict
            r_state = RobotState(**robot_state)
            
            a_payload = {
                "task": task.dict(), 
                "perception": perception.dict(), 
                "robot": r_state.dict()
            }
            resp = requests.post(f"{self.action_url}/act", json=a_payload, timeout=0.1) # Fast timeout
            resp.raise_for_status()
            action_data = resp.json()
            action = ActionChunk(**action_data)
        except Exception as e:
             self.logger.error(f"Action failed: {e}")
             raise e
             
        # Log
        self.log_step(perception, task, action)
        
        return action