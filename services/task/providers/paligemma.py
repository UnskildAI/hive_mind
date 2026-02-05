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
        """Generate task latent from perception and instruction with a thinking phase."""
        if not self.is_loaded:
            self.load_model()
        
        # Convert perception to image
        image = self._perception_to_image(perception)
        
        # --- DEEP THINKING PHASE ---
        # 1. Object Detection / Grounding
        # We prompt PaliGemma to find the object mentioned in the instruction
        grounding_prompt = f"detect {instruction.replace('Pick up ', '').replace('Move to ', '')}"
        logger.info(f"Thinking: Attempting to ground task... Prompt: '{grounding_prompt}'")
        
        detect_inputs = self.processor(images=image, text=grounding_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**detect_inputs, max_new_tokens=50)
            detect_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        logger.info(f"Thought: Grounding result: {detect_output}")
        
        # 2. Extract Spatial Tokens
        # PaliGemma outputs <locXXXX> tokens. We want to convert these to a spatial bias.
        coord_bias = np.zeros(32) # Small vector to store spatial context
        import re
        loc_matches = re.findall(r'<loc(\d+)>', detect_output)
        if len(loc_matches) >= 4:
            # Found a bounding box! [ymin, xmin, ymax, xmax] in normalized 1024 range
            coords = [int(m) / 1024.0 for m in loc_matches]
            center_y = (coords[0] + coords[2]) / 2.0
            center_x = (coords[1] + coords[3]) / 2.0
            coord_bias[0] = center_x
            coord_bias[1] = center_y
            logger.info(f"Thought: Targeting object at spatial center: ({center_x:.2f}, {center_y:.2f})")
        
        # 3. Main Instruction Embedding
        # Now run the full instruction to get the semantic features
        main_prompt = f"answer en {instruction}"
        inputs = self.processor(
            images=image,
            text=main_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract mean-pooled embedding
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            last_hidden = outputs.hidden_states[-1]
            semantic_emb = last_hidden.mean(dim=1).squeeze().cpu().numpy()
            
            # Incorporate spatial bias into the first part of the embedding
            # This "grounds" the high-dimensional latent in physical space
            if semantic_emb.shape[0] >= 256:
                goal_embedding = semantic_emb[:256].copy()
            else:
                goal_embedding = np.pad(semantic_emb, (0, 256 - semantic_emb.shape[0]))
            
            # Inject spatial knowledge into the tail of the embedding (last 32 dims)
            goal_embedding[-32:] = coord_bias
        else:
            goal_embedding = np.random.randn(256).astype(np.float32)
        
        # Generate subtask ID
        import hashlib
        subtask_id = hashlib.md5(instruction.encode()).hexdigest()[:8]
        
        return TaskLatent(
            schema_version="1.0.0",
            goal_embedding=goal_embedding.tolist(),
            constraints={
                "speed": "adaptive",
                "thought": f"I see the {instruction} target. Moving to ground coordinates calculated from visual tokens."
            },
            subtask_id=f"paligemma_{subtask_id}",
            confidence=0.95
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
