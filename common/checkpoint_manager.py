"""
Checkpoint Manager for HuggingFace Model Loading and Caching.

Handles:
- Downloading models from HuggingFace Hub with automatic caching
- Loading local checkpoints from custom paths
- Model metadata and registry
- GPU memory management and device mapping
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a model checkpoint."""
    repo_id: str
    model_type: str  # "vlm", "action_expert"
    size_gb: float
    precision: str  # "fp32", "fp16", "bf16", "int8"
    device: str
    supports_multi_gpu: bool = True


class CheckpointManager:
    """
    Centralized manager for downloading and loading model checkpoints.
    
    Features:
    - HuggingFace Hub integration with caching
    - Local checkpoint support
    - Multi-GPU device mapping
    - Mixed precision loading
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            cache_dir: Custom cache directory. Defaults to ~/.cache/huggingface/
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/")
        self.loaded_models: Dict[str, Any] = {}
        
        logger.info(f"CheckpointManager initialized with cache_dir: {self.cache_dir}")
    
    def download_from_hub(
        self,
        repo_id: str,
        revision: str = "main",
        allow_patterns: Optional[list] = None,
        ignore_patterns: Optional[list] = None,
    ) -> Path:
        """
        Download model from HuggingFace Hub.
        
        Args:
            repo_id: HuggingFace repo ID (e.g., "openvla/openvla-7b")
            revision: Git revision (branch, tag, or commit hash)
            allow_patterns: File patterns to download (e.g., ["*.safetensors"])
            ignore_patterns: File patterns to ignore
        
        Returns:
            Path to downloaded checkpoint directory
        """
        logger.info(f"Downloading {repo_id} from HuggingFace Hub...")
        
        try:
            checkpoint_path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=self.cache_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            
            logger.info(f"Downloaded to: {checkpoint_path}")
            return Path(checkpoint_path)
        
        except Exception as e:
            logger.error(f"Failed to download {repo_id}: {e}")
            raise
    
    def load_model(
        self,
        checkpoint_path: Union[str, Path],
        model_class: Any,
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> Any:
        """
        Load model from checkpoint with device mapping and precision.
        
        Args:
            checkpoint_path: Path to checkpoint (local or HF repo ID)
            model_class: Model class to instantiate (e.g., AutoModel)
            device_map: Device mapping strategy ("auto", "balanced", or dict)
            torch_dtype: Precision (torch.float16, torch.bfloat16, torch.float32)
            **kwargs: Additional arguments for model loading
        
        Returns:
            Loaded model instance
        """
        checkpoint_str = str(checkpoint_path)
        
        # Check if already loaded
        if checkpoint_str in self.loaded_models:
            logger.info(f"Model {checkpoint_str} already loaded, returning cached instance")
            return self.loaded_models[checkpoint_str]
        
        logger.info(f"Loading model from {checkpoint_str}")
        logger.info(f"  Device map: {device_map}")
        logger.info(f"  Precision: {torch_dtype}")
        
        try:
            model = model_class.from_pretrained(
                checkpoint_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                **kwargs
            )
            
            # Cache loaded model
            self.loaded_models[checkpoint_str] = model
            
            logger.info(f"Model loaded successfully")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load model from {checkpoint_str}: {e}")
            raise
    
    def load_processor(
        self,
        checkpoint_path: Union[str, Path],
        **kwargs
    ) -> Any:
        """
        Load processor/tokenizer from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (local or HF repo ID)
            **kwargs: Additional arguments for processor loading
        
        Returns:
            Loaded processor instance
        """
        logger.info(f"Loading processor from {checkpoint_path}")
        
        try:
            # Try AutoProcessor first (for VLMs)
            try:
                processor = AutoProcessor.from_pretrained(checkpoint_path, **kwargs)
                logger.info("Loaded AutoProcessor")
                return processor
            except:
                # Fall back to AutoTokenizer (for text-only models)
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, **kwargs)
                logger.info("Loaded AutoTokenizer")
                return tokenizer
        
        except Exception as e:
            logger.error(f"Failed to load processor from {checkpoint_path}: {e}")
            raise
    
    def get_model_info(self, repo_id: str) -> ModelMetadata:
        """
        Get metadata for a model.
        
        Args:
            repo_id: HuggingFace repo ID
        
        Returns:
            ModelMetadata instance
        """
        # This is a simplified version - in production, you'd query HF API
        # or maintain a registry file
        
        # Example metadata (hardcoded for common models)
        metadata_registry = {
            "openvla/openvla-7b": ModelMetadata(
                repo_id="openvla/openvla-7b",
                model_type="vlm",
                size_gb=14.0,
                precision="fp16",
                device="cuda",
                supports_multi_gpu=True
            ),
            "google/paligemma-3b-pt-224": ModelMetadata(
                repo_id="google/paligemma-3b-pt-224",
                model_type="vlm",
                size_gb=6.0,
                precision="fp16",
                device="cuda",
                supports_multi_gpu=True
            ),
        }
        
        return metadata_registry.get(repo_id, ModelMetadata(
            repo_id=repo_id,
            model_type="unknown",
            size_gb=0.0,
            precision="fp32",
            device="cuda"
        ))
    
    def clear_cache(self, model_id: Optional[str] = None):
        """
        Clear loaded model cache.
        
        Args:
            model_id: Specific model to clear, or None to clear all
        """
        if model_id:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                logger.info(f"Cleared cache for {model_id}")
        else:
            self.loaded_models.clear()
            logger.info("Cleared all model cache")
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()


# Global singleton instance
_checkpoint_manager = None


def get_checkpoint_manager(cache_dir: Optional[str] = None) -> CheckpointManager:
    """Get or create global CheckpointManager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager(cache_dir)
    return _checkpoint_manager
