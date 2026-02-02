"""
GPU Utilities for Multi-GPU Management and Optimization.

Handles:
- Device mapping for multi-GPU setups
- Mixed precision configuration
- Memory profiling and monitoring
- Batch size optimization
"""

import torch
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    device_id: int
    name: str
    total_memory_gb: float
    allocated_memory_gb: float
    free_memory_gb: float
    utilization_percent: float


class GPUManager:
    """
    Manager for multi-GPU setups and optimization.
    
    Features:
    - Automatic device mapping
    - Memory monitoring
    - Mixed precision support
    """
    
    def __init__(self):
        """Initialize GPU manager."""
        self.num_gpus = torch.cuda.device_count()
        self.available_gpus = list(range(self.num_gpus))
        
        logger.info(f"GPUManager initialized with {self.num_gpus} GPUs")
        
        if self.num_gpus > 0:
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    
    def get_device_map(
        self,
        model_type: str = "auto",
        num_gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[str] = None
    ) -> Union[str, Dict[str, int]]:
        """
        Get device mapping for model loading.
        
        Args:
            model_type: Type of mapping ("auto", "balanced", "sequential")
            num_gpus: Number of GPUs to use (None = all available)
            max_memory_per_gpu: Max memory per GPU (e.g., "22GB")
        
        Returns:
            Device map (str or dict)
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            return "cpu"
        
        if model_type == "auto":
            # Let accelerate handle device mapping automatically
            if max_memory_per_gpu:
                return {i: max_memory_per_gpu for i in range(num_gpus or self.num_gpus)}
            return "auto"
        
        elif model_type == "balanced":
            # Distribute layers evenly across GPUs
            return "balanced"
        
        elif model_type == "sequential":
            # Place model on first GPU only
            return {"": 0}
        
        else:
            raise ValueError(f"Unknown device map type: {model_type}")
    
    def get_torch_dtype(self, precision: str) -> torch.dtype:
        """
        Get torch dtype from precision string.
        
        Args:
            precision: Precision type ("fp32", "fp16", "bf16", "int8")
        
        Returns:
            torch.dtype
        """
        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        
        if precision not in dtype_map:
            logger.warning(f"Unknown precision '{precision}', defaulting to fp32")
            return torch.float32
        
        dtype = dtype_map[precision]
        
        # Check if bfloat16 is supported
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            logger.warning("BF16 not supported on this GPU, falling back to FP16")
            return torch.float16
        
        return dtype
    
    def get_gpu_info(self, device_id: int = 0) -> GPUInfo:
        """
        Get information about a specific GPU.
        
        Args:
            device_id: GPU device ID
        
        Returns:
            GPUInfo instance
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        if device_id >= self.num_gpus:
            raise ValueError(f"Invalid device_id {device_id}, only {self.num_gpus} GPUs available")
        
        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory / 1e9  # GB
        
        # Get current memory usage
        allocated = torch.cuda.memory_allocated(device_id) / 1e9  # GB
        free = total_memory - allocated
        
        # Utilization (simplified - would need nvidia-ml-py for accurate utilization)
        utilization = (allocated / total_memory) * 100 if total_memory > 0 else 0.0
        
        return GPUInfo(
            device_id=device_id,
            name=props.name,
            total_memory_gb=total_memory,
            allocated_memory_gb=allocated,
            free_memory_gb=free,
            utilization_percent=utilization
        )
    
    def get_all_gpu_info(self) -> List[GPUInfo]:
        """Get information about all GPUs."""
        return [self.get_gpu_info(i) for i in range(self.num_gpus)]
    
    def print_gpu_summary(self):
        """Print summary of all GPUs."""
        if not torch.cuda.is_available():
            logger.info("CUDA not available")
            return
        
        logger.info("=" * 80)
        logger.info("GPU Summary")
        logger.info("=" * 80)
        
        for info in self.get_all_gpu_info():
            logger.info(f"GPU {info.device_id}: {info.name}")
            logger.info(f"  Total Memory: {info.total_memory_gb:.2f} GB")
            logger.info(f"  Allocated:    {info.allocated_memory_gb:.2f} GB")
            logger.info(f"  Free:         {info.free_memory_gb:.2f} GB")
            logger.info(f"  Utilization:  {info.utilization_percent:.1f}%")
        
        logger.info("=" * 80)
    
    def optimize_for_inference(self):
        """Apply optimizations for inference."""
        if torch.cuda.is_available():
            # Enable TF32 for faster matmul on Ampere GPUs (A5000)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN autotuner for optimal convolution algorithms
            torch.backends.cudnn.benchmark = True
            
            logger.info("Applied inference optimizations:")
            logger.info("  - TF32 enabled for matmul")
            logger.info("  - cuDNN autotuner enabled")
    
    def clear_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")


# Global singleton instance
_gpu_manager = None


def get_gpu_manager() -> GPUManager:
    """Get or create global GPUManager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager
