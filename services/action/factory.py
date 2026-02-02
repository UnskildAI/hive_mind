"""
Action Expert Factory for creating action policies from configuration.
"""

import yaml
import os
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ActionExpertFactory:
    """
    Factory for creating Action Expert instances from configuration.
    
    Supports:
    - ACT (Action Chunking Transformer)
    - Diffusion Policy
    - Pi0 Flow Matching
    - Legacy policies (Scripted, Frozen, MLP)
    """
    
    @staticmethod
    def create(config_path: str = "configs/master_config.yaml") -> Any:
        """
        Create Action Expert from configuration.
        
        Args:
            config_path: Path to master config (default: configs/master_config.yaml)
        
        Returns:
            ActionExpertBase instance (or legacy policy)
        """
        # Load master config
        if not os.path.exists(config_path):
            # Try legacy config path
            legacy_path = "services/action/config.yaml"
            if os.path.exists(legacy_path):
                logger.warning(f"Using legacy config {legacy_path}, consider migrating to master_config.yaml")
                config_path = legacy_path
            else:
                raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Get Action Expert provider from config
        # Priority: Env Var > Config File > Default
        if "action_expert" in config:
            # New master config format
            action_config = config["action_expert"]
            provider = os.getenv("ACTION_EXPERT_PROVIDER", action_config.get("provider", "scripted"))
        else:
            # Legacy config format
            provider = os.getenv("ACTION_MODE", config.get("mode", "scripted"))
        
        logger.info(f"Loading Action Expert provider: {provider}")
        
        # Create provider instance
        if provider == "act":
            from services.action.experts.act_policy import ACTActionExpert
            return ACTActionExpert(config.get("action_expert", {}))
        
        elif provider == "diffusion":
            from services.action.experts.diffusion_policy import DiffusionActionExpert
            return DiffusionActionExpert(config.get("action_expert", {}))
        
        elif provider == "pi0" or provider == "pi0_fast":
            from services.action.experts.pi0_policy import Pi0ActionExpert
            return Pi0ActionExpert(config.get("action_expert", {}))
        
        # Legacy providers (backward compatibility)
        elif provider == "scripted":
            from services.action.policy import ScriptedController
            return ScriptedController(config)
        
        elif provider == "frozen_policy":
            from services.action.policy import FrozenPolicy
            return FrozenPolicy(config)
        
        elif provider == "learned" or provider == "mlp_chunk":
            from services.action.policy import ActionExpert
            return ActionExpert(config_path)
        
        else:
            logger.warning(f"Unknown Action Expert provider '{provider}', defaulting to scripted")
            from services.action.policy import ScriptedController
            return ScriptedController(config)
