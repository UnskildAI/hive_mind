"""
Diffusion Policy from LeRobot.

Diffusion Policy uses denoising diffusion for smooth action generation.
Excellent for complex, multimodal action distributions.

Reference: https://github.com/huggingface/lerobot
"""

import torch
import numpy as np
from typing import Dict, Any
import logging

from .base import ActionExpertBase
from common.schemas.task import TaskLatent
from common.schemas.perception import PerceptionState
from common.schemas.robot_state import RobotState
from common.schemas.action import ActionChunk

logger = logging.getLogger(__name__)


class DiffusionActionExpert(ActionExpertBase):
    """
    Diffusion Policy from LeRobot.
    
    Features:
    - DDPM/DDIM denoising for action generation
    - Smooth, natural trajectories
    - Handles multimodal action distributions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Diffusion policy."""
        super().__init__(config)
        
        self.checkpoint_config = config.get("checkpoint", {})
        self.device = config.get("device", "cuda:1")
        self.precision = config.get("precision", "fp16")
        
        self.policy = None
        self.is_loaded = False
        
        logger.info("Diffusion Action Expert initialized")
    
    def load_model(self) -> None:
        """Load Diffusion policy from HuggingFace Hub or local checkpoint."""
        if self.is_loaded:
            return
        
        logger.info("Loading Diffusion policy...")
        
        source = self.checkpoint_config.get("source", "huggingface")
        
        if source == "huggingface":
            repo_id = self.checkpoint_config.get("repo_id", "lerobot/diffusion_pusht")
            checkpoint_path = repo_id
        elif source == "local":
            checkpoint_path = self.checkpoint_config.get("local_path")
        else:
            raise ValueError(f"Unknown checkpoint source: {source}")
        
        try:
            from lerobot.common.policies.diffusion import DiffusionPolicy
            
            self.policy = DiffusionPolicy.from_pretrained(
                checkpoint_path,
                device=self.device,
            )
            
            self.policy.eval()
            self.is_loaded = True
            
            logger.info("Diffusion policy loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load Diffusion policy: {e}")
            raise
    
    def act(
        self,
        task: TaskLatent,
        perception: PerceptionState,
        robot: RobotState
    ) -> ActionChunk:
        """Generate action chunk using Diffusion policy."""
        if not self.is_loaded:
            self.load_model()
        
        # Prepare observation
        observation = self._prepare_observation(task, perception, robot)
        
        # Run diffusion sampling
        with torch.no_grad():
            action_pred = self.policy.select_action(observation)
        
        # Convert to numpy
        if isinstance(action_pred, torch.Tensor):
            actions_np = action_pred.cpu().numpy()
        else:
            actions_np = np.array(action_pred)
        
        # Ensure correct shape
        if actions_np.ndim == 1:
            actions_np = np.tile(actions_np, (self.horizon, 1))
        
        actions_list = actions_np[:self.horizon].tolist()
        
        return ActionChunk(
            schema_version="1.0.0",
            actions=actions_list,
            horizon=self.horizon,
            control_mode=self.control_mode
        )
    
    def _prepare_observation(self, task, perception, robot) -> Dict[str, torch.Tensor]:
        """Prepare observation for Diffusion policy."""
        qpos = torch.tensor(robot.joint_position, dtype=torch.float32).unsqueeze(0).to(self.device)
        task_emb = torch.tensor(task.goal_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return {
            "qpos": qpos,
            "task_embedding": task_emb,
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Diffusion policy metadata."""
        return {
            "model_name": "Diffusion Policy",
            "provider": "diffusion",
            "repo_id": self.checkpoint_config.get("repo_id", "lerobot/diffusion_pusht"),
            "model_size_gb": 1.5,
            "precision": self.precision,
            "device": self.device,
            "loaded": self.is_loaded,
            "horizon": self.horizon,
        }
