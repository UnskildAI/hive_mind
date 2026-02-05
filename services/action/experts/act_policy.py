"""
ACT (Action Chunking Transformer) Policy from LeRobot.

ACT is a DETR-based transformer policy that predicts action chunks.
Excellent for fine-grained manipulation tasks with temporal dependencies.

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
from common.checkpoint_manager import get_checkpoint_manager
from common.utils.gpu_utils import get_gpu_manager

logger = logging.getLogger(__name__)


class ACTActionExpert(ActionExpertBase):
    """
    Action Chunking Transformer (ACT) policy from LeRobot.
    
    Features:
    - DETR-based encoder-decoder transformer
    - Predicts action chunks (horizon 50-100)
    - Handles temporal dependencies
    - Fast training with few demonstrations
    - Supports bimanual control
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ACT policy.
        
        Args:
            config: Configuration from master_config.yaml['action_expert']
        """
        super().__init__(config)
        
        self.checkpoint_config = config.get("checkpoint", {})
        self.device = config.get("device", "cuda:1")
        self.precision = config.get("precision", "fp16")
        self.temporal_ensemble = config.get("temporal_ensemble", False)
        
        self.checkpoint_manager = get_checkpoint_manager()
        self.gpu_manager = get_gpu_manager()
        
        self.policy = None
        self.is_loaded = False
        
        logger.info("ACT Action Expert initialized")
    
    def load_model(self) -> None:
        """Load ACT policy from HuggingFace Hub or local checkpoint."""
        if self.is_loaded:
            logger.info("ACT policy already loaded")
            return
        
        logger.info("Loading ACT policy...")
        
        # Get checkpoint path
        source = self.checkpoint_config.get("source", "huggingface")
        
        if source == "huggingface":
            repo_id = self.checkpoint_config.get("repo_id", "lerobot/act_aloha_sim_insertion_human")
            checkpoint_path = repo_id
        elif source == "local":
            checkpoint_path = self.checkpoint_config.get("local_path")
        else:
            raise ValueError(f"Unknown checkpoint source: {source}")
        
        try:
            # Import LeRobot ACT policy (0.4.x API)
            from lerobot.policies.act.modeling_act import ACTPolicy
            import torch
            
            logger.info(f"Loading ACT from: {checkpoint_path}")
            
            # Move to device and set precision
            torch_dtype = self.gpu_manager.get_torch_dtype(self.precision)
            
            # Load policy with explicit dtype (HF-style)
            self.policy = ACTPolicy.from_pretrained(
                checkpoint_path,
                torch_dtype=torch_dtype,
            )
            
            # Move to device
            self.policy.to(self.device)
            
            # Additional safety: explicitly force half if precision is fp16
            if self.precision == "fp16":
                self.policy.half()
            
            # Set to eval mode
            self.policy.eval()
            
            self.is_loaded = True
            
            logger.info(f"ACT policy loaded successfully in {self.precision}")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Internal Model DType: {next(self.policy.parameters()).dtype}")
        
        except ImportError as e:
            logger.error(f"LeRobot 0.4.3 import error: {e}")
            logger.warning("Attempting alternative import for ACTPolicy...")
            try:
                from lerobot.policies.act.modeling_act import ACTPolicy
                torch_dtype = self.gpu_manager.get_torch_dtype(self.precision)
                self.policy = ACTPolicy.from_pretrained(checkpoint_path, torch_dtype=torch_dtype)
                self.policy.to(self.device)
                self.policy.eval()
                self.is_loaded = True
            except Exception as e2:
                logger.error(f"Fallback import also failed: {e2}")
                self.policy = None
        except Exception as e:
            logger.error(f"Failed to load ACT policy: {e}")
            raise
    
    def act(
        self,
        task: TaskLatent,
        perception: PerceptionState,
        robot: RobotState
    ) -> ActionChunk:
        """
        Generate action chunk using ACT policy.
        
        Args:
            task: Task latent from VLM
            perception: Current perception state
            robot: Current robot state
        
        Returns:
            ActionChunk with predicted actions
        """
        if not self.is_loaded:
            self.load_model()
        
        # Prepare inputs for ACT
        # ACT expects: observation dict with keys like 'qpos', 'images', etc.
        observation = self._prepare_observation(task, perception, robot)
        
        # Check if policy is loaded
        if self.policy is None:
            logger.warning("ACT policy not loaded, returning zero actions")
            # Return zero actions as placeholder
            import numpy as np
            actions_np = np.zeros((self.horizon, len(robot.joint_position)))
        else:
            # Run ACT forward pass
            try:
                # Use autocast to handle any internal dtype mismatches (e.g. noise generation)
                device_type = "cuda" if "cuda" in str(self.device) else "cpu"
                model_dtype = next(self.policy.parameters()).dtype
                
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device_type, dtype=model_dtype):
                        # ACT returns action predictions
                        action_pred = self.policy.select_action(observation)
                
                # Convert to numpy
                if isinstance(action_pred, torch.Tensor):
                    actions_np = action_pred.cpu().numpy()
                else:
                    actions_np = np.array(action_pred)
            except Exception as e:
                logger.error(f"ACT model forward pass failed: {e}")
                raise
        
        # Ensure correct shape: [horizon, action_dim]
        if actions_np.ndim == 1:
            # Single action, expand to horizon
            actions_np = np.tile(actions_np, (self.horizon, 1))
        elif actions_np.shape[0] < self.horizon:
            # Pad to horizon
            pad_size = self.horizon - actions_np.shape[0]
            last_action = actions_np[-1:]
            padding = np.tile(last_action, (pad_size, 1))
            actions_np = np.vstack([actions_np, padding])
        elif actions_np.shape[0] > self.horizon:
            # Truncate to horizon
            actions_np = actions_np[:self.horizon]
        
        # Convert to list of lists
        actions_list = actions_np.tolist()
        
        return ActionChunk(
            schema_version="1.0.0",
            actions=actions_list,
            horizon=self.horizon,
            control_mode=self.control_mode
        )
    
    def _prepare_observation(
        self,
        task: TaskLatent,
        perception: PerceptionState,
        robot: RobotState
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare observation dict for ACT policy.
        
        Args:
            task: Task latent
            perception: Perception state
            robot: Robot state
        
        Returns:
            Observation dictionary
        """
        # Determine model precision
        try:
            model_dtype = next(self.policy.parameters()).dtype
            # Peek at a few parameters to be sure
            params = list(self.policy.parameters())
            logger.info(f"Model precision check: param[0]={params[0].dtype}, param[-1]={params[-1].dtype}")
        except (StopIteration, AttributeError):
            model_dtype = torch.float32
            
        logger.info(f"ACT model detected dtype: {model_dtype} on device: {self.device}")
            
        # Create standard tensors
        qpos = torch.tensor(robot.joint_position, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        observation = {}
        
        # Redundant keys for proprioception
        observation["observation.state"] = qpos
        observation["state"] = qpos
        observation["qpos"] = qpos
        
        # Add task embedding
        task_emb = torch.tensor(task.goal_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        observation["task_embedding"] = task_emb
        
        # Handle Images
        image_keys = []
        if self.policy is not None and hasattr(self.policy, "config"):
            if hasattr(self.policy.config, "image_features"):
                image_keys = self.policy.config.image_features
            elif hasattr(self.policy.config, "input_features"):
                image_keys = [k for k in self.policy.config.input_features if "image" in k]
        
        if not image_keys:
            image_keys = ["observation.images.top", "observation.image"]
            
        if hasattr(perception, 'raw_image') and perception.raw_image is not None:
            image_tensor = self._image_to_tensor(perception.raw_image).unsqueeze(0).to(self.device)
            # Ensure image is float32 before recursive cast
            image_tensor = image_tensor.float()
            for key in image_keys:
                observation[key] = image_tensor
        else:
            logger.warning("No image found in perception state for ACT observation")
            
        # CRITICAL FIX: Explicitly provide the latent sample to avoid internal Float32 defaults
        # The latent_dim is usually in self.policy.config.latent_dim
        latent_dim = getattr(self.policy.config, "latent_dim", 32)
        bs = 1
        observation["latent_sample"] = torch.zeros((bs, latent_dim), device=self.device, dtype=model_dtype)
        
        # FINAL STEP: Recursive cast EVERYTHING to model precision
        observation = self._to_device_and_dtype(observation, self.device, model_dtype)
        
        # Deep debug log dtypes
        for k, v in observation.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"Observation Tensor {k}: dtype={v.dtype}, device={v.device}")
        
        return observation
    
    def _image_to_tensor(self, image) -> torch.Tensor:
        """Convert image to tensor."""
        if isinstance(image, np.ndarray):
            # Normalize to [0, 1] and convert to CHW format
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            if image.ndim == 3 and image.shape[2] == 3:
                # HWC -> CHW
                image = np.transpose(image, (2, 0, 1))
            return torch.from_numpy(image).float()
        return image
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ACT policy metadata."""
        return {
            "model_name": "ACT (Action Chunking Transformer)",
            "provider": "act",
            "repo_id": self.checkpoint_config.get("repo_id", "lerobot/act_aloha_sim_insertion_human"),
            "model_size_gb": 2.0,  # Approximate
            "precision": self.precision,
            "device": self.device,
            "loaded": self.is_loaded,
            "horizon": self.horizon,
            "control_mode": self.control_mode,
        }
