#!/usr/bin/env python3
"""
Test VLA Inference Pipeline

Tests VLM and Action Expert inference with mock inputs.
Usage:
    python scripts/test_vla_inference.py --vlm openvla --action act
    python scripts/test_vla_inference.py --vlm paligemma --action diffusion
"""

import sys
import os
import argparse
import logging
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.task.model import TaskModelFactory
from services.action.policy import ActionExpertFactory
from common.schemas.perception import PerceptionState
from common.schemas.robot_state import RobotState
from common.utils.gpu_utils import get_gpu_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_perception() -> PerceptionState:
    """Create mock perception state with dummy image."""
    # Create dummy RGB image (224x224)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Create dummy scene tokens (16 tokens x 64 dim)
    scene_tokens = np.random.randn(16, 64).tolist()
    
    # Create dummy camera pose (4x4 transformation matrix)
    camera_pose = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0]
    ]
    
    return PerceptionState(
        schema_version="1.0.0",
        scene_tokens=scene_tokens,
        camera_pose=camera_pose,
        raw_image=dummy_image,
        timestamp=0.0
    )


def create_mock_robot_state() -> RobotState:
    """Create mock robot state."""
    return RobotState(
        joint_position=[0.0, -0.5, 0.5, 0.0, 0.0, 0.0],  # 6 DOF
        joint_velocities=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        gripper_state=0.5
    )


def test_vlm_inference(vlm_provider: str):
    """Test VLM inference."""
    logger.info("=" * 80)
    logger.info(f"Testing VLM Provider: {vlm_provider}")
    logger.info("=" * 80)
    
    # Update config to use specified provider
    import yaml
    config_path = "configs/master_config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config["vlm"]["provider"] = vlm_provider
    
    # Write temporary config
    temp_config_path = "/tmp/test_master_config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)
    
    # Create VLM
    try:
        vlm = TaskModelFactory.create(temp_config_path)
        
        # Get model info
        if hasattr(vlm, 'get_model_info'):
            info = vlm.get_model_info()
            logger.info(f"Model Info: {info}")
        
        # Create mock inputs
        perception = create_mock_perception()
        instruction = "Pick up the red cube and place it in the blue box"
        
        # Run inference
        logger.info(f"Running inference with instruction: '{instruction}'")
        
        import time
        start_time = time.time()
        
        task_latent = vlm.infer(perception, instruction)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Print results
        logger.info(f"✓ Inference successful!")
        logger.info(f"  Latency: {latency_ms:.2f} ms")
        logger.info(f"  Goal embedding shape: {len(task_latent.goal_embedding)}")
        logger.info(f"  Subtask ID: {task_latent.subtask_id}")
        logger.info(f"  Confidence: {task_latent.confidence}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ VLM inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_expert_inference(action_provider: str):
    """Test Action Expert inference."""
    logger.info("=" * 80)
    logger.info(f"Testing Action Expert: {action_provider}")
    logger.info("=" * 80)
    
    # Update config
    import yaml
    config_path = "configs/master_config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config["action_expert"]["provider"] = action_provider
    
    # Write temporary config
    temp_config_path = "/tmp/test_master_config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)
    
    # Create Action Expert
    try:
        from services.action.policy import ActionExpertFactory
        
        action_expert = ActionExpertFactory.create(temp_config_path)
        
        # Get model info
        if hasattr(action_expert, 'get_model_info'):
            info = action_expert.get_model_info()
            logger.info(f"Model Info: {info}")
        
        # Create mock inputs
        from common.schemas.task import TaskLatent
        
        task_latent = TaskLatent(
            schema_version="1.0.0",
            goal_embedding=np.random.randn(256).tolist(),
            constraints={},
            subtask_id="test_subtask",
            confidence=0.95
        )
        
        perception = create_mock_perception()
        robot_state = create_mock_robot_state()
        
        # Run inference
        logger.info("Running action inference...")
        
        import time
        start_time = time.time()
        
        action_chunk = action_expert.act(task_latent, perception, robot_state)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Print results
        logger.info(f"✓ Inference successful!")
        logger.info(f"  Latency: {latency_ms:.2f} ms")
        logger.info(f"  Horizon: {action_chunk.horizon}")
        logger.info(f"  Action shape: {len(action_chunk.actions)} x {len(action_chunk.actions[0])}")
        logger.info(f"  Control mode: {action_chunk.control_mode}")
        logger.info(f"  First action: {action_chunk.actions[0]}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Action Expert inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test VLA Inference Pipeline")
    parser.add_argument("--vlm", type=str, default=None, 
                        help="VLM provider to test (openvla, paligemma, gemini)")
    parser.add_argument("--action", type=str, default=None,
                        help="Action expert to test (act, diffusion, pi0)")
    parser.add_argument("--gpu-info", action="store_true",
                        help="Print GPU information")
    
    args = parser.parse_args()
    
    # Print GPU info
    if args.gpu_info or (args.vlm or args.action):
        gpu_manager = get_gpu_manager()
        gpu_manager.print_gpu_summary()
        gpu_manager.optimize_for_inference()
    
    success = True
    
    # Test VLM
    if args.vlm:
        success = test_vlm_inference(args.vlm) and success
    
    # Test Action Expert
    if args.action:
        success = test_action_expert_inference(args.action) and success
    
    if not args.vlm and not args.action:
        logger.info("No tests specified. Use --vlm or --action to run tests.")
        logger.info("Example: python scripts/test_vla_inference.py --vlm openvla --action act")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
