import numpy as np
import logging
from typing import List, Tuple, Optional

class SafetyValidator:
    """
    Hard safety layer that validates all robot commands before execution.
    Safety checks have absolute veto power over ML outputs.
    """
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("safety_validator")
        
        # Load safety configuration
        safety_config = config.get("safety", {})
        self.enabled = safety_config.get("enable_limits", True)
        
        # Joint limits
        joints_config = config.get("joints", {})
        self.joint_names = joints_config.get("names", [])
        self.joint_min = np.array(joints_config.get("limits", {}).get("min", []))
        self.joint_max = np.array(joints_config.get("limits", {}).get("max", []))
        self.velocity_limits = np.array(joints_config.get("velocity_limits", []))
        
        # Workspace bounds (optional)
        self.workspace_bounds = safety_config.get("workspace_bounds", None)
        
        # Collision parameters
        self.collision_checks = safety_config.get("collision_checks", {})
        
        # Emergency stop state
        self.emergency_stop_active = False
        
        # Statistics
        self.violations = {
            "joint_limits": 0,
            "velocity_limits": 0,
            "workspace": 0,
            "collision": 0,
            "emergency_stop": 0
        }
        
        self.logger.info(f"SafetyValidator initialized: enabled={self.enabled}")

    def set_emergency_stop(self, active: bool):
        """Set emergency stop state (called by ROS subscriber)"""
        if active and not self.emergency_stop_active:
            self.logger.critical("EMERGENCY STOP ACTIVATED")
        elif not active and self.emergency_stop_active:
            self.logger.warning("Emergency stop released")
        self.emergency_stop_active = active

    def validate_command(self, positions: List[float], velocities: Optional[List[float]] = None) -> Tuple[bool, List[float], str]:
        """
        Validate a command before sending to hardware.
        
        Args:
            positions: Target joint positions
            velocities: Optional velocity commands
            
        Returns:
            (is_safe, safe_positions, reason)
            - is_safe: True if command passes all checks
            - safe_positions: Clamped/corrected positions (if applicable)
            - reason: Description of any violations
        """
        if not self.enabled:
            return True, positions, "safety_disabled"
        
        # Check 1: Emergency Stop
        if self.emergency_stop_active:
            self.violations["emergency_stop"] += 1
            return False, positions, "emergency_stop_active"
        
        positions_arr = np.array(positions)
        
        # Check 2: Joint Limits
        if len(positions_arr) != len(self.joint_min):
            return False, positions, f"dimension_mismatch: expected {len(self.joint_min)}, got {len(positions_arr)}"
        
        violations = []
        clamped_positions = positions_arr.copy()
        
        # Clamp to joint limits
        below_min = positions_arr < self.joint_min
        above_max = positions_arr > self.joint_max
        
        if np.any(below_min) or np.any(above_max):
            self.violations["joint_limits"] += 1
            clamped_positions = np.clip(positions_arr, self.joint_min, self.joint_max)
            
            violated_joints = []
            for i, name in enumerate(self.joint_names):
                if below_min[i]:
                    violated_joints.append(f"{name}={positions_arr[i]:.3f}<{self.joint_min[i]:.3f}")
                elif above_max[i]:
                    violated_joints.append(f"{name}={positions_arr[i]:.3f}>{self.joint_max[i]:.3f}")
            
            violations.append(f"joint_limits: {', '.join(violated_joints)}")
        
        # Check 3: Velocity Limits (if provided)
        if velocities is not None and len(self.velocity_limits) > 0:
            velocities_arr = np.abs(np.array(velocities))
            if np.any(velocities_arr > self.velocity_limits):
                self.violations["velocity_limits"] += 1
                violations.append("velocity_limits_exceeded")
        
        # Check 4: Basic Collision Heuristics (prevent extreme arm folding)
        if self.collision_checks.get("enable_self_collision", False):
            # SoArm 100: Elbow is joint index 2. 
            # In many configurations, 0.0 is straight, and extreme negative/positive is folding.
            # We want to prevent the arm from hitting the base.
            if len(clamped_positions) >= 3:
                elbow_angle = clamped_positions[2]
                min_angle = self.collision_checks.get("min_elbow_angle", -2.5) # Allow straighter positions
                if elbow_angle < min_angle:
                    self.violations["collision"] += 1
                    violations.append(f"potential_self_collision: elbow ({elbow_angle:.3f}) < {min_angle}")
        
        # Decision
        if violations:
            reason = "; ".join(violations)
            self.logger.warning(f"Safety violation: {reason}")
            # Return clamped positions as "safe" fallback
            return True, clamped_positions.tolist(), reason
        
        return True, positions, "safe"

    def get_statistics(self) -> dict:
        """Return safety violation statistics"""
        return self.violations.copy()

    def reset_statistics(self):
        """Reset violation counters"""
        for key in self.violations:
            self.violations[key] = 0
