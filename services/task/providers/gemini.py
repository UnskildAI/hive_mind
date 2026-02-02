"""
Gemini Provider - Google's Gemini 2.0 Flash API.

Uses Google's Gemini API for vision-language understanding.
Requires internet connection and API key.

Reference: https://ai.google.dev/gemini-api
"""

import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
import logging
import os

from .base import VLMProviderBase
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent

logger = logging.getLogger(__name__)


class GeminiProvider(VLMProviderBase):
    """
    Google Gemini 2.0 Flash API provider.
    
    Features:
    - Cloud-based VLM (no local GPU required)
    - State-of-the-art vision-language understanding
    - Requires API key and internet connection
    
    Note: Higher latency (~200-500ms) compared to local models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Gemini provider."""
        super().__init__(config)
        
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set 'api_key' in config or "
                "GOOGLE_API_KEY environment variable."
            )
        
        self.model_name = config.get("model_name", "gemini-2.0-flash-exp")
        self.temperature = config.get("temperature", 0.7)
        
        self.client = None
        self.is_loaded = False
        
        logger.info(f"Gemini provider initialized (model: {self.model_name})")
    
    def load_model(self) -> None:
        """Initialize Gemini API client."""
        if self.is_loaded:
            return
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            
            self.is_loaded = True
            logger.info("Gemini API client initialized")
        
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def infer(self, perception: PerceptionState, instruction: str) -> TaskLatent:
        """
        Generate task latent using Gemini API.
        
        Args:
            perception: Perception state with image
            instruction: Natural language instruction
        
        Returns:
            TaskLatent
        """
        if not self.is_loaded:
            self.load_model()
        
        # Convert perception to image
        image = self._perception_to_image(perception)
        
        # Create prompt for structured output
        prompt = f"""
You are a robot vision-language model. Given an image and instruction, 
generate a semantic goal embedding for the robot task.

Instruction: {instruction}

Respond with a JSON object containing:
- task_description: Brief description of the task
- key_objects: List of relevant objects in the scene
- goal_state: Desired end state
- confidence: Confidence score (0-1)
"""
        
        try:
            # Generate response
            response = self.client.generate_content(
                [prompt, image],
                generation_config={
                    "temperature": self.temperature,
                }
            )
            
            # Parse response (simplified - in production, use structured output)
            response_text = response.text
            
            # Generate deterministic embedding from response
            # (In production, use Gemini's embedding API)
            import hashlib
            embedding_hash = hashlib.sha256(response_text.encode()).digest()
            goal_embedding = np.frombuffer(embedding_hash, dtype=np.uint8)[:256]
            goal_embedding = (goal_embedding.astype(np.float32) / 255.0) * 2 - 1  # Normalize to [-1, 1]
            
            # Generate subtask ID
            subtask_id = hashlib.md5(instruction.encode()).hexdigest()[:8]
            
            return TaskLatent(
                schema_version="1.0.0",
                goal_embedding=goal_embedding.tolist(),
                constraints={"speed": "adaptive", "api_response": response_text[:100]},
                subtask_id=f"gemini_{subtask_id}",
                confidence=0.90
            )
        
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            # Fallback to random embedding
            return TaskLatent(
                schema_version="1.0.0",
                goal_embedding=np.random.randn(256).tolist(),
                constraints={"error": str(e)},
                subtask_id="gemini_error",
                confidence=0.0
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
        """Get Gemini model metadata."""
        return {
            "model_name": self.model_name,
            "provider": "gemini",
            "model_size_gb": 0.0,  # Cloud-based
            "precision": "cloud",
            "device": "cloud",
            "loaded": self.is_loaded,
            "supports_multi_gpu": False,
        }
