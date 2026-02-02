"""
GR00T Provider - NVIDIA Isaac GR00T N (Placeholder).

Placeholder for future NVIDIA Isaac GR00T N integration.
Will be implemented when GR00T is available in LeRobot.

Reference: https://developer.nvidia.com/isaac/groot
"""

import numpy as np
from typing import Dict, Any
import logging

from .base import VLMProviderBase
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent

logger = logging.getLogger(__name__)


class GR00TProvider(VLMProviderBase):
    """
    NVIDIA Isaac GR00T N provider (placeholder).
    
    TODO: Implement when GR00T is available in LeRobot/HuggingFace.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize GR00T provider."""
        super().__init__(config)
        logger.warning("GR00T provider is a placeholder - not yet implemented")
    
    def load_model(self) -> None:
        """Load GR00T model (not implemented)."""
        raise NotImplementedError(
            "GR00T provider is not yet implemented. "
            "Use 'openvla', 'paligemma', or 'gemini' instead."
        )
    
    def infer(self, perception: PerceptionState, instruction: str) -> TaskLatent:
        """Generate task latent (not implemented)."""
        raise NotImplementedError("GR00T provider is not yet implemented")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model info."""
        return {
            "model_name": "GR00T (Not Implemented)",
            "provider": "gr00t",
            "loaded": False,
        }
