"""
OpenVLA Provider - Vision-Language-Action Model from OpenVLA.

OpenVLA is a 7B parameter VLA model optimized for robot manipulation.
Supports OFT (Optimized Fine-Tuning) variant for 25-50x faster inference.

Reference: https://github.com/openvla/openvla
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
import logging

from .base import VLMProviderBase
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent
from common.checkpoint_manager import get_checkpoint_manager
from common.utils.gpu_utils import get_gpu_manager

logger = logging.getLogger(__name__)


class OpenVLAProvider(VLMProviderBase):
    """
    OpenVLA Vision-Language-Action model provider.
    
    Features:
    - 7B parameter model with vision-language understanding
    - Optimized for robot manipulation tasks
    - Supports OFT variant for fast inference
    - Multi-GPU support via device_map
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenVLA provider.
        
        Args:
            config: Configuration from master_config.yaml['vlm']
        """
        super().__init__(config)
        
        self.checkpoint_config = config.get("checkpoint", {})
        self.device_map = config.get("device_map", "auto")
        self.precision = config.get("precision", "fp16")
        self.max_length = config.get("max_length", 512)
        
        self.checkpoint_manager = get_checkpoint_manager()
        self.gpu_manager = get_gpu_manager()
        
        # Model will be loaded lazily
        self.model = None
        self.processor = None
        self.is_loaded = False
        
        logger.info("OpenVLA provider initialized")
    
    def load_model(self) -> None:
        """Load OpenVLA model from HuggingFace Hub or local checkpoint."""
        if self.is_loaded:
            logger.info("OpenVLA model already loaded")
            return
        
        logger.info("Loading OpenVLA model...")
        
        # Get checkpoint path
        source = self.checkpoint_config.get("source", "huggingface")
        
        if source == "huggingface":
            repo_id = self.checkpoint_config.get("repo_id", "openvla/openvla-7b")
            revision = self.checkpoint_config.get("revision", "main")
            
            logger.info(f"Loading from HuggingFace Hub: {repo_id}")
            checkpoint_path = repo_id  # transformers can load directly from repo_id
        
        elif source == "local":
            checkpoint_path = self.checkpoint_config.get("local_path")
            if not checkpoint_path:
                raise ValueError("local_path must be specified when source='local'")
            logger.info(f"Loading from local path: {checkpoint_path}")
        
        else:
            raise ValueError(f"Unknown checkpoint source: {source}")
        
        # Get torch dtype
        torch_dtype = self.gpu_manager.get_torch_dtype(self.precision)
        
        # Get device map
        device_map = self.gpu_manager.get_device_map(
            model_type=self.device_map,
            max_memory_per_gpu=self.config.get("max_memory_per_gpu")
        )
        
        try:
            # Import transformers models
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            # Load model with device mapping and precision
            logger.info(f"Loading model with device_map={device_map}, dtype={torch_dtype}")
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                checkpoint_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,  # OpenVLA may require custom code
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
            )
            
            # Set to eval mode
            self.model.eval()
            
            self.is_loaded = True
            
            logger.info("OpenVLA model loaded successfully")
            
            # Print GPU summary
            self.gpu_manager.print_gpu_summary()
        
        except Exception as e:
            logger.error(f"Failed to load OpenVLA model: {e}")
            raise
    
    def infer(self, perception: PerceptionState, instruction: str) -> TaskLatent:
        """
        Generate task latent from perception and instruction.
        
        Args:
            perception: Perception state with scene_tokens or raw images
            instruction: Natural language instruction
        
        Returns:
            TaskLatent with goal_embedding, subtask_id, and confidence
        """
        if not self.is_loaded:
            self.load_model()
        
        # Convert perception to image
        # If perception has raw image, use it; otherwise reconstruct from tokens
        image = self._perception_to_image(perception)
        
        # Preprocess inputs
        inputs = self.processor(
            images=image,
            text=instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Move inputs to device (first GPU if multi-GPU)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract goal embedding from hidden states
        # Use last hidden state and pool over sequence dimension
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # Get last layer hidden states
            last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
            
            # Mean pooling over sequence dimension
            goal_embedding = last_hidden.mean(dim=1).squeeze()  # [hidden_dim]
            
            # Convert to numpy and resize to expected dimension (256)
            goal_embedding_np = goal_embedding.cpu().numpy()
            
            # If embedding is larger than 256, use linear projection or truncate
            if goal_embedding_np.shape[0] > 256:
                # Simple truncation (in production, use learned projection)
                goal_embedding_np = goal_embedding_np[:256]
            elif goal_embedding_np.shape[0] < 256:
                # Pad with zeros
                goal_embedding_np = np.pad(
                    goal_embedding_np,
                    (0, 256 - goal_embedding_np.shape[0]),
                    mode='constant'
                )
        
        else:
            # Fallback: use random embedding (shouldn't happen with proper model)
            logger.warning("No hidden states found, using random embedding")
            goal_embedding_np = np.random.randn(256).astype(np.float32)
        
        # Generate subtask ID (hash of instruction for now)
        import hashlib
        subtask_id = hashlib.md5(instruction.encode()).hexdigest()[:8]
        
        # Create TaskLatent
        task_latent = TaskLatent(
            schema_version="1.0.0",
            goal_embedding=goal_embedding_np.tolist(),
            constraints={"speed": "adaptive"},
            subtask_id=f"openvla_{subtask_id}",
            confidence=0.95  # High confidence for pretrained model
        )
        
        return task_latent
    
    def _perception_to_image(self, perception: PerceptionState) -> Image.Image:
        """
        Convert PerceptionState to PIL Image.
        
        Args:
            perception: Perception state
        
        Returns:
            PIL Image
        """
        # Check if perception has raw image data
        if hasattr(perception, 'raw_image') and perception.raw_image is not None:
            # Assume raw_image is numpy array (H, W, C)
            if isinstance(perception.raw_image, np.ndarray):
                return Image.fromarray(perception.raw_image.astype(np.uint8))
            elif isinstance(perception.raw_image, Image.Image):
                return perception.raw_image
        
        # If no raw image, create placeholder (in production, reconstruct from tokens)
        logger.warning("No raw image in perception, using placeholder")
        # Create a simple placeholder image (224x224 RGB)
        placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
        return Image.fromarray(placeholder)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenVLA model metadata."""
        return {
            "model_name": "OpenVLA-7B",
            "provider": "openvla",
            "repo_id": self.checkpoint_config.get("repo_id", "openvla/openvla-7b"),
            "model_size_gb": 14.0,  # Approximate size in FP16
            "precision": self.precision,
            "device_map": self.device_map,
            "loaded": self.is_loaded,
            "supports_multi_gpu": True,
        }
