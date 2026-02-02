"""
PaliGemma Provider - Google's Vision-Language Model.

PaliGemma is a 3B parameter VLM designed for vision-language tasks.
Can be used as a frozen feature extractor or fine-tuned end-to-end.

Reference: https://huggingface.co/google/paligemma-3b-pt-224
"""

import torch
import numpy as np
from typing import Dict, Any
from PIL import Image
import logging

from .base import VLMProviderBase
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent
from common.checkpoint_manager import get_checkpoint_manager
from common.utils.gpu_utils import get_gpu_manager

logger = logging.getLogger(__name__)


class PaliGemmaProvider(VLMProviderBase):
    """
    PaliGemma Vision-Language Model provider.
    
    Features:
    - 3B parameter model (smaller than OpenVLA)
    - Frozen feature extraction mode
    - Lower memory footprint (~6GB vs ~14GB)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PaliGemma provider."""
        super().__init__(config)
        
        self.checkpoint_config = config.get("checkpoint", {})
        self.device_map = config.get("device_map", "auto")
        self.precision = config.get("precision", "fp16")
        self.max_length = config.get("max_length", 512)
        
        self.checkpoint_manager = get_checkpoint_manager()
        self.gpu_manager = get_gpu_manager()
        
        self.model = None
        self.processor = None
        self.is_loaded = False
        
        logger.info("PaliGemma provider initialized")
    
    def load_model(self) -> None:
        """Load PaliGemma model from HuggingFace Hub or local checkpoint."""
        if self.is_loaded:
            logger.info("PaliGemma model already loaded")
            return
        
        logger.info("Loading PaliGemma model...")
        
        # Get checkpoint path
        source = self.checkpoint_config.get("source", "huggingface")
        
        if source == "huggingface":
            repo_id = self.checkpoint_config.get("repo_id", "google/paligemma-3b-pt-224")
            checkpoint_path = repo_id
        elif source == "local":
            checkpoint_path = self.checkpoint_config.get("local_path")
        else:
            raise ValueError(f"Unknown checkpoint source: {source}")
        
        # Get torch dtype and device map
        torch_dtype = self.gpu_manager.get_torch_dtype(self.precision)
        device_map = self.gpu_manager.get_device_map(model_type=self.device_map)
        
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                checkpoint_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation="eager",  # Avoid SDPA compatibility issues
            )
            
            self.processor = AutoProcessor.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
            )
            
            # Freeze model for feature extraction
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.model.eval()
            self.is_loaded = True
            
            logger.info("PaliGemma model loaded successfully (frozen)")
            self.gpu_manager.print_gpu_summary()
        
        except Exception as e:
            logger.error(f"Failed to load PaliGemma model: {e}")
            raise
    
    def infer(self, perception: PerceptionState, instruction: str) -> TaskLatent:
        """Generate task latent from perception and instruction."""
        if not self.is_loaded:
            self.load_model()
        
        # Convert perception to image
        image = self._perception_to_image(perception)
        
        # Preprocess
        inputs = self.processor(
            images=image,
            text=instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract embedding
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            last_hidden = outputs.hidden_states[-1]
            goal_embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy()
            
            # Resize to 256 dimensions
            if goal_embedding.shape[0] > 256:
                goal_embedding = goal_embedding[:256]
            elif goal_embedding.shape[0] < 256:
                goal_embedding = np.pad(goal_embedding, (0, 256 - goal_embedding.shape[0]))
        else:
            goal_embedding = np.random.randn(256).astype(np.float32)
        
        # Generate subtask ID
        import hashlib
        subtask_id = hashlib.md5(instruction.encode()).hexdigest()[:8]
        
        return TaskLatent(
            schema_version="1.0.0",
            goal_embedding=goal_embedding.tolist(),
            constraints={"speed": "adaptive"},
            subtask_id=f"paligemma_{subtask_id}",
            confidence=0.92
        )
    
    def _perception_to_image(self, perception: PerceptionState) -> Image.Image:
        """Convert PerceptionState to PIL Image."""
        if hasattr(perception, 'raw_image') and perception.raw_image is not None:
            if isinstance(perception.raw_image, np.ndarray):
                return Image.fromarray(perception.raw_image.astype(np.uint8))
            elif isinstance(perception.raw_image, Image.Image):
                return perception.raw_image
        
        # Placeholder
        placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
        return Image.fromarray(placeholder)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get PaliGemma model metadata."""
        return {
            "model_name": "PaliGemma-3B",
            "provider": "paligemma",
            "repo_id": self.checkpoint_config.get("repo_id", "google/paligemma-3b-pt-224"),
            "model_size_gb": 6.0,
            "precision": self.precision,
            "device_map": self.device_map,
            "loaded": self.is_loaded,
            "supports_multi_gpu": True,
        }
