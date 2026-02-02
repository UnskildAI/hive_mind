"""
Base class for Action Expert providers.

All Action Expert implementations must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from common.schemas.task import TaskLatent
from common.schemas.perception import PerceptionState
from common.schemas.robot_state import RobotState
from common.schemas.action import ActionChunk


class ActionExpertBase(ABC):
    """
    Abstract base class for Action Expert providers.
    
    All Action Expert implementations (ACT, Diffusion, Pi0, etc.) must
    inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Action Expert.
        
        Args:
            config: Configuration dictionary from master_config.yaml
        """
        self.config = config
        self.policy = None
        self.device = None
        self.precision = None
        self.horizon = config.get("horizon", 10)
        self.control_mode = config.get("control_mode", "position")
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the action policy model from checkpoint.
        
        This method should:
        1. Load policy from HuggingFace Hub or local path
        2. Set up device and precision
        3. Set policy to eval mode
        """
        pass
    
    @abstractmethod
    def act(
        self,
        task: TaskLatent,
        perception: PerceptionState,
        robot: RobotState
    ) -> ActionChunk:
        """
        Generate action chunk from task, perception, and robot state.
        
        Args:
            task: Task latent from VLM (goal embedding, subtask_id, etc.)
            perception: Current perception state (scene tokens, images, etc.)
            robot: Current robot state (joint positions, velocities, gripper)
        
        Returns:
            ActionChunk with actions, horizon, and control_mode
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary with model information:
            - model_name: str
            - model_size_gb: float
            - precision: str
            - device: str
            - loaded: bool
            - horizon: int
        """
        pass
    
    def unload_model(self) -> None:
        """
        Unload model from memory.
        
        Useful for switching between models or freeing GPU memory.
        """
        if self.policy is not None:
            del self.policy
            self.policy = None
        
        # Clear GPU cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
