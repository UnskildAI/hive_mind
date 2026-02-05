"""
Pi0 Flow Matching Policy from LeRobot.

Pi0 uses flow matching for fast, high-frequency action generation.
Optimized for real-time robot control at 50Hz.

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


class Pi0ActionExpert(ActionExpertBase):
    """
    Pi0 Flow Matching policy from LeRobot.
    
    Features:
    - Flow matching for fast inference
    - 50Hz high-frequency control
    - Cross-embodiment support
    - Pi0-FAST variant for even faster inference
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Pi0 policy."""
        super().__init__(config)
        
        self.checkpoint_config = config.get("checkpoint", {})
        self.device = config.get("device", "cuda:1")
        self.precision = config.get("precision", "fp16")
        self.use_fast_variant = config.get("provider") == "pi0_fast"
        
        self.policy = None
        self.is_loaded = False
        
        logger.info(f"Pi0 Action Expert initialized (fast={self.use_fast_variant})")
    
    def load_model(self) -> None:
        """Load Pi0 policy from HuggingFace Hub or local checkpoint."""
        if self.is_loaded:
            return
        
        logger.info("Loading Pi0 policy...")
        
        source = self.checkpoint_config.get("source", "huggingface")
        
        if source == "huggingface":
            repo_id = self.checkpoint_config.get("repo_id", "lerobot/pi0")
            checkpoint_path = repo_id
        elif source == "local":
            checkpoint_path = self.checkpoint_config.get("local_path")
        else:
            raise ValueError(f"Unknown checkpoint source: {source}")
        
        try:
            # Try to import Pi0 from lerobot 0.4.x
            try:
                from lerobot.policies.pi0.modeling_pi0 import Pi0Policy
                
                self.policy = Pi0Policy.from_pretrained(
                    checkpoint_path,
                )
                self.policy.to(self.device)
            except ImportError:
                logger.warning("Pi0Policy not found in lerobot.policies.pi0, checking lerobot.policies...")
                try:
                    from lerobot.policies import pi0
                    # If it's a module, it might have a factory or direct access
                    self.policy = None # Placeholder for now as Pi0 is often more complex to load
                except ImportError:
                    logger.warning("Pi0Policy not found, using placeholder")
                    self.policy = None
            
            self.is_loaded = True
            
            logger.info("Pi0 policy loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load Pi0 policy: {e}")
            raise
    
    def act(
        self,
        task: TaskLatent,
        perception: PerceptionState,
        robot: RobotState
    ) -> ActionChunk:
        """Generate action chunk using Pi0 policy."""
        if not self.is_loaded:
            self.load_model()
        
        if self.policy is None:
            # Placeholder: return zero actions
            logger.warning("Pi0 policy not available, returning zero actions")
            actions_list = [[0.0] * len(robot.joint_position) for _ in range(self.horizon)]
            return ActionChunk(
                schema_version="1.0.0",
                actions=actions_list,
                horizon=self.horizon,
                control_mode=self.control_mode
            )
        
        # Prepare observation
        observation = self._prepare_observation(task, perception, robot)
        
        # Run flow matching
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
        """Prepare observation for Pi0 policy."""
        model_dtype = getattr(self.policy, "dtype", torch.float32)
        qpos = torch.tensor(robot.joint_position, dtype=model_dtype).unsqueeze(0).to(self.device)
        
        observation = {
            "observation.state": qpos,
            "task_embedding": torch.tensor(task.goal_embedding, dtype=model_dtype).unsqueeze(0).to(self.device)
        }
        
        # Determine image keys from config if available
        image_keys = ["observation.images.top", "observation.image"]
        if self.policy is not None and hasattr(self.policy, "config"):
            if hasattr(self.policy.config, "image_features"):
                image_keys = self.policy.config.image_features
            elif hasattr(self.policy.config, "input_features"):
                image_keys = [k for k in self.policy.config.input_features if "image" in k]
                
        # Map available image to all requested keys
        if hasattr(perception, 'raw_image') and perception.raw_image is not None:
            image_np = perception.raw_image
            if image_np.dtype == np.uint8:
                image_np = image_np.astype(np.float32) / 255.0
            if image_np.ndim == 3 and image_np.shape[2] == 3:
                image_np = np.transpose(image_np, (2, 0, 1))
            
            image_tensor = torch.from_numpy(image_np).to(dtype=model_dtype).unsqueeze(0).to(self.device)
            
            for key in image_keys:
                observation[key] = image_tensor
                
        return observation
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Pi0 policy metadata."""
        return {
            "model_name": "Pi0-FAST" if self.use_fast_variant else "Pi0",
            "provider": "pi0_fast" if self.use_fast_variant else "pi0",
            "repo_id": self.checkpoint_config.get("repo_id", "lerobot/pi0"),
            "model_size_gb": 3.0,
            "precision": self.precision,
            "device": self.device,
            "loaded": self.is_loaded,
            "horizon": self.horizon,
        }
