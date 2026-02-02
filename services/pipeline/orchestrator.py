import requests
import time
import json
import logging
import base64
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent
from common.schemas.action import ActionChunk
from common.schemas.robot_state import RobotState
from services.pipeline.data_logger import RobotDataLogger
import os
import yaml

class PipelineOrchestrator:
    def __init__(self, perception_url, task_url, action_url):
        self.perception_url = perception_url
        self.task_url = task_url
        self.action_url = action_url
        
        self.logger = logging.getLogger("orchestrator")
        
        # Load config
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        # Services Config
        self.data_logger = RobotDataLogger(self.config)
        
        # State
        self.last_task_latent = None
        self.last_task_time = 0
        self.task_ttl = self.config.get("orchestrator", {}).get("task_ttl", 0.5) 
        self.current_instruction = None

    def log_step(self, perception, task, robot, action, image_base64):
        # Delegate to specialized logger
        self.data_logger.log_step(perception, task, robot, action, image_base64)

    def run_step(self, image_base64: str, camera_pose: list, instruction: str, robot_state: dict) -> ActionChunk:
        # 1. Perception
        try:
            p_payload = {"image_base64": image_base64, "camera_pose": camera_pose}
            resp = requests.post(f"{self.perception_url}/perceive", json=p_payload, timeout=1.0) # Increased to 1s
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
                t_payload = {"perception": perception.model_dump(), "instruction": instruction}
                resp = requests.post(f"{self.task_url}/process", json=t_payload, timeout=2.0) # Task can be heavy
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
        if task is None:
            self.logger.error("Cannot proceed to action: Task latent is None")
            raise RuntimeError("Task latent missing. Check Task service logs.")

        try:
            # Construct RobotState from dict
            r_state = RobotState(**robot_state)
            
            a_payload = {
                "task": task.model_dump(), 
                "perception": perception.model_dump(), 
                "robot": r_state.model_dump()
            }
            # Increased timeout to 1.0s to handle heavy load
            resp = requests.post(f"{self.action_url}/act", json=a_payload, timeout=1.0) 
            resp.raise_for_status()
            action_data = resp.json()
            action = ActionChunk(**action_data)
        except Exception as e:
            self.logger.error(f"Action failed: {e}")
            raise e
             
        # 4. Log Step (Include robot state and image)
        self.log_step(perception, task, r_state, action, image_base64)
        
        return action