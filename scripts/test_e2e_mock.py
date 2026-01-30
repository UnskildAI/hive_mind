import torch
from services.perception.model import PerceptionModel
from services.task.model import TaskModel
from services.action.policy import ActionExpert
from common.schemas.robot_state import RobotState
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("e2e_dry_run")

def run():
    logger.info("Initializing models...")
    perception_model = PerceptionModel()
    task_model = TaskModel()
    action_expert = ActionExpert()
    
    logger.info("Models initialized.")
   
    # Fake inputs
    logger.info("Generating fake inputs...")
    # Image: [1, 3, 224, 224]
    image_tensor = torch.randn(1, 3, 224, 224)
    camera_pose = [[0.0]*12]
    instruction = "pick up the red cube"
    robot_state = RobotState(
        joint_position=[0.0]*7, 
        joint_velocities=[0.0]*7, 
        gripper_state=0.0
    )
    
    # 1. Perception
    logger.info("Running Perception...")
    perception_state = perception_model.perceive(image_tensor, camera_pose)
    logger.info(f"Perception output tokens: {len(perception_state.scene_tokens)}")
    
    # 2. Task
    logger.info("Running Task...")
    task_latent = task_model.infer(perception_state, instruction)
    logger.info(f"Task latent subtask_id: {task_latent.subtask_id}")
    
    # 3. Action
    logger.info("Running Action...")
    action_chunk = action_expert.act(task_latent, perception_state, robot_state)
    logger.info(f"Action chunk returned actions: {len(action_chunk.actions)}")
    
    assert len(action_chunk.actions) == action_expert.horizon
    logger.info("E2E Dry Run PASSED.")

if __name__ == "__main__":
    run()
