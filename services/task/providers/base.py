"""
Base class for VLM (Vision-Language Model) providers.

All VLM providers must implement this interface for consistency.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent


class VLMProviderBase(ABC):
    """
    Abstract base class for Vision-Language Model providers.
    
    All VLM implementations (OpenVLA, PaliGemma, Gemini, etc.) must
    inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VLM provider.
        
        Args:
            config: Configuration dictionary from master_config.yaml
        """
        self.config = config
        self.model = None
        self.processor = None
        self.device = None
        self.precision = None
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the VLM model from checkpoint.
        
        This method should:
        1. Load model from HuggingFace Hub or local path
        2. Set up device mapping and precision
        3. Initialize processor/tokenizer
        4. Set model to eval mode
        """
        pass
    
    @abstractmethod
    def infer(self, perception: PerceptionState, instruction: str) -> TaskLatent:
        """
        Generate task latent from perception and instruction.
        
        Args:
            perception: Current perception state (scene tokens, images, etc.)
            instruction: Natural language instruction (e.g., "Pick up the red cube")
        
        Returns:
            TaskLatent with goal_embedding, subtask_id, and confidence
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
        """
        pass
    
    def preprocess_image(self, image: np.ndarray) -> Any:
        """
        Preprocess image for model input.
        
        Args:
            image: Raw image as numpy array (H, W, C)
        
        Returns:
            Preprocessed image tensor
        """
        # Default implementation - override if needed
        if self.processor is not None:
            return self.processor(images=image, return_tensors="pt")
        return image
    
    def preprocess_instruction(self, instruction: str) -> Any:
        """
        Preprocess instruction text for model input.
        
        Args:
            instruction: Raw instruction string
        
        Returns:
            Preprocessed instruction tokens
        """
        # Default implementation - override if needed
        if self.processor is not None:
            return self.processor(text=instruction, return_tensors="pt")
        return instruction
    
    def postprocess_output(self, model_output: Any) -> np.ndarray:
        """
        Postprocess model output to goal embedding.
        
        Args:
            model_output: Raw model output
        
        Returns:
            Goal embedding as numpy array
        """
        # Default implementation - override if needed
        if hasattr(model_output, 'last_hidden_state'):
            # Extract last hidden state and pool
            hidden = model_output.last_hidden_state
            # Mean pooling over sequence dimension
            embedding = hidden.mean(dim=1).squeeze().cpu().numpy()
            return embedding
        return model_output
    
    def unload_model(self) -> None:
        """
        Unload model from memory.
        
        Useful for switching between models or freeing GPU memory.
        """
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # Clear GPU cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
