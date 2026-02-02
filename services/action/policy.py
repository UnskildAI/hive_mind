import torch
import torch.nn as nn
from typing import List
import yaml
import time
import math
import logging
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent
from common.schemas.robot_state import RobotState
from common.schemas.action import ActionChunk

# Import Action Expert Factory for easy access
from services.action.factory import ActionExpertFactory

logger = logging.getLogger("action_service")

class ScriptedController:
    """Scripted controller for validation without ML inference."""
    
    def __init__(self, config: dict):
        self.config = config
        self.scripted_config = config.get("scripted", {})
        self.policy_config = config.get("policy", {})
        
        self.trajectory_type = self.scripted_config.get("trajectory_type", "joint_oscillation")
        self.frequency = self.scripted_config.get("frequency", 0.5)
        self.amplitude = self.scripted_config.get("amplitude", 0.1)
        self.joint_limits = self.scripted_config.get("joint_limits", [-3.14, 3.14])
        self.velocity_limit = self.scripted_config.get("velocity_limit", 1.0)
        
        self.action_dim = self.policy_config.get("action_dim", 7)
        self.horizon = self.policy_config.get("horizon", 10)
        self.control_mode = self.policy_config.get("control_mode", "position")
        
        self.start_time = time.time()
        self.call_count = 0
        self.clamp_count = 0
        self.last_log_time = time.time()
        
        logger.info(f"ScriptedController initialized: {self.trajectory_type}")
    
    def _clamp_action(self, action: List[float]) -> List[float]:
        """Clamp actions to joint limits."""
        clamped = []
        for i, val in enumerate(action):
            if val < self.joint_limits[0] or val > self.joint_limits[1]:
                self.clamp_count += 1
                val = max(self.joint_limits[0], min(self.joint_limits[1], val))
            clamped.append(val)
        return clamped
    
    def _generate_joint_oscillation(self, robot: RobotState) -> List[List[float]]:
        """Generate sinusoidal joint oscillation."""
        t = time.time() - self.start_time
        actions = []
        
        for step in range(self.horizon):
            # Future time for this step (assuming 20Hz control)
            future_t = t + step * 0.05
            action = []
            
            for joint_idx in range(self.action_dim):
                # Phase offset per joint for visual variety
                phase = joint_idx * (2 * math.pi / self.action_dim)
                # Sinusoidal motion around current position
                base_pos = robot.joint_position[joint_idx] if joint_idx < len(robot.joint_position) else 0.0
                offset = self.amplitude * math.sin(2 * math.pi * self.frequency * future_t + phase)
                action.append(base_pos + offset)
            
            actions.append(self._clamp_action(action))
        
        return actions
    
    def _generate_gripper_toggle(self, robot: RobotState) -> List[List[float]]:
        """Generate periodic gripper open/close."""
        t = time.time() - self.start_time
        actions = []
        
        # Toggle every 2 seconds
        gripper_state = 1.0 if (int(t / 2.0) % 2 == 0) else 0.0
        
        for step in range(self.horizon):
            action = list(robot.joint_position[:self.action_dim-1]) if len(robot.joint_position) >= self.action_dim-1 else [0.0] * (self.action_dim - 1)
            action.append(gripper_state)
            actions.append(self._clamp_action(action))
        
        return actions
    
    def _generate_square_trajectory(self, robot: RobotState) -> List[List[float]]:
        """Generate square trajectory in joint space."""
        t = time.time() - self.start_time
        actions = []
        
        # Simple square in first two joints
        cycle_time = 4.0  # 4 seconds per cycle
        phase = (t % cycle_time) / cycle_time
        
        for step in range(self.horizon):
            future_phase = ((t + step * 0.05) % cycle_time) / cycle_time
            action = list(robot.joint_position) if len(robot.joint_position) == self.action_dim else [0.0] * self.action_dim
            
            # Square corners at 0, 0.25, 0.5, 0.75
            if future_phase < 0.25:
                action[0] = self.amplitude
                action[1] = self.amplitude
            elif future_phase < 0.5:
                action[0] = -self.amplitude
                action[1] = self.amplitude
            elif future_phase < 0.75:
                action[0] = -self.amplitude
                action[1] = -self.amplitude
            else:
                action[0] = self.amplitude
                action[1] = -self.amplitude
            
            actions.append(self._clamp_action(action))
        
        return actions
    
    def act(self, task: TaskLatent, perception: PerceptionState, robot: RobotState) -> ActionChunk:
        """Generate scripted action chunk."""
        self.call_count += 1
        
        # Generate trajectory based on type
        if self.trajectory_type == "joint_oscillation":
            actions = self._generate_joint_oscillation(robot)
        elif self.trajectory_type == "gripper_toggle":
            actions = self._generate_gripper_toggle(robot)
        elif self.trajectory_type == "square":
            actions = self._generate_square_trajectory(robot)
        else:
            raise ValueError(f"Unknown trajectory type: {self.trajectory_type}")
        
        # Logging every 100 calls
        if self.call_count % 100 == 0:
            elapsed = time.time() - self.last_log_time
            frequency = 100.0 / elapsed if elapsed > 0 else 0.0
            
            # Calculate command magnitude
            magnitudes = [sum(abs(a) for a in action) for action in actions]
            avg_magnitude = sum(magnitudes) / len(magnitudes)
            
            logger.info(f"Scripted Controller Stats:")
            logger.info(f"  Loop frequency: {frequency:.2f} Hz")
            logger.info(f"  Avg command magnitude: {avg_magnitude:.3f}")
            logger.info(f"  Safety clamps (last 100): {self.clamp_count}")
            
            self.last_log_time = time.time()
            self.clamp_count = 0
        
        return ActionChunk(
            schema_version="1.0.0",
            actions=actions,
            horizon=self.horizon,
            control_mode=self.control_mode
        )


class TimingStats:
    """Helper for tracking latency and jitter."""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.latencies = []
        self.last_time = None
        self.period_errors = []

    def update(self, start_time, end_time):
        latency = (end_time - start_time) * 1000.0  # ms
        self.latencies.append(latency)
        
        current_time = end_time
        if self.last_time is not None:
            # Jitter calculation: variation in loop period
            actual_period = (current_time - self.last_time) * 1000.0
            self.period_errors.append(actual_period)
        
        self.last_time = current_time
        
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)
        if len(self.period_errors) > self.window_size:
            self.period_errors.pop(0)

    def get_stats(self):
        if not self.latencies:
            return 0.0, 0.0, 0.0
        
        avg_latency = sum(self.latencies) / len(self.latencies)
        max_latency = max(self.latencies)
        
        # Jitter as std dev of periods
        jitter = 0.0
        if len(self.period_errors) > 1:
            avg_period = sum(self.period_errors) / len(self.period_errors)
            variance = sum((p - avg_period) ** 2 for p in self.period_errors) / len(self.period_errors)
            jitter = math.sqrt(variance)
            
        return avg_latency, max_latency, jitter

class FrozenPolicy:
    """Deterministic policy moving towards a fixed target."""
    
    def __init__(self, config: dict):
        self.config = config
        self.frozen_config = config.get("frozen_policy", {})
        self.policy_config = config.get("policy", {})
        
        self.target_pose = self.frozen_config.get("target_pose", [0.0] * 6)
        self.smoothing = self.frozen_config.get("smoothing", 0.05)
        
        self.action_dim = self.policy_config.get("action_dim", 6)
        self.horizon = self.policy_config.get("horizon", 10)
        self.control_mode = self.policy_config.get("control_mode", "position")
        
        self.stats = TimingStats()
        self.call_count = 0
        
        logger.info(f"FrozenPolicy initialized: target={self.target_pose}, smoothing={self.smoothing}")

    def act(self, task: TaskLatent, perception: PerceptionState, robot: RobotState) -> ActionChunk:
        start_time = time.perf_counter()
        self.call_count += 1
        
        current_pos = list(robot.joint_position)
        # Ensure current_pos matches action_dim
        if len(current_pos) < self.action_dim:
            current_pos.extend([0.0] * (self.action_dim - len(current_pos)))
        elif len(current_pos) > self.action_dim:
            current_pos = current_pos[:self.action_dim]
            
        actions = []
        temp_pos = current_pos
        
        for step in range(self.horizon):
            step_action = []
            for i in range(self.action_dim):
                target = self.target_pose[i] if i < len(self.target_pose) else temp_pos[i]
                # Simple exponential smoothing towards target
                new_val = temp_pos[i] + self.smoothing * (target - temp_pos[i])
                step_action.append(new_val)
            
            actions.append(step_action)
            temp_pos = step_action # Predictive horizon

        end_time = time.perf_counter()
        self.stats.update(start_time, end_time)
        
        if self.call_count % 100 == 0:
            avg_lat, max_lat, jitter = self.stats.get_stats()
            logger.info(f"FrozenPolicy Stats [last 100]:")
            logger.info(f"  Avg Latency: {avg_lat:.3f} ms")
            logger.info(f"  Max Latency: {max_lat:.3f} ms")
            logger.info(f"  Timing Jitter: {jitter:.3f} ms")
            
        return ActionChunk(
            schema_version="1.0.0",
            actions=actions,
            horizon=self.horizon,
            control_mode=self.control_mode
        )

class MLPPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, horizon):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * horizon)
        )
        self.action_dim = action_dim
        self.horizon = horizon

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.net(x)
        return out.view(batch_size, self.horizon, self.action_dim)

class ActionExpert:
    def __init__(self, config_path="services/action/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config.get("device", "cpu")
        policy_config = self.config["policy"]
        
        # Dimensions need to match other models
        # For simplicity, we'll calculate or hardcode based on known schema sizes
        # Perception: 16 tokens * 64 dim = 1024 flat? Or we use pool?
        # Task: 256
        # Robot: 7 (pos) + 7 (vel) + 1 (gripper) = 15
        
        self.perception_flat_dim = 1024
        self.task_dim = 256
        self.robot_dim = 15
        self.input_dim = self.perception_flat_dim + self.task_dim + self.robot_dim
        
        self.action_dim = policy_config["action_dim"]
        self.horizon = policy_config["horizon"]
        self.control_mode = policy_config["control_mode"]
        
        if policy_config["type"] == "mlp_chunk":
            self.model = MLPPolicy(self.input_dim, self.action_dim, self.horizon)
        else:
            raise ValueError("Unknown policy type")
            
        self.model.to(self.device)
        self.model.eval()

    def act(self, task: TaskLatent, perception: PerceptionState, robot: RobotState) -> ActionChunk:
        with torch.no_grad():
            # Prepare Inputs
            
            # 1. Perception
            # Assume [N, Dv] -> Flatten. In real system might use attention.
            p_tensor = torch.tensor(perception.scene_tokens).flatten() # [N*Dv]
            # Pad/Crop to expected size
            if p_tensor.shape[0] != self.perception_flat_dim:
                if p_tensor.shape[0] < self.perception_flat_dim:
                     p_tensor = torch.nn.functional.pad(p_tensor, (0, self.perception_flat_dim - p_tensor.shape[0]))
                else:
                     p_tensor = p_tensor[:self.perception_flat_dim]
            
            # 2. Task
            t_tensor = torch.tensor(task.goal_embedding)
            
            # 3. Robot
            r_tensor = torch.cat([
                torch.tensor(robot.joint_position),
                torch.tensor(robot.joint_velocities),
                torch.tensor([robot.gripper_state])
            ])
            
            # Concat
            x = torch.cat([p_tensor, t_tensor, r_tensor]).unsqueeze(0).to(self.device) # [1, InputDim]
            
            # Inference
            actions_tensor = self.model(x) # [1, H, A]
            actions = actions_tensor[0].cpu().tolist()
            
            return ActionChunk(
                schema_version="1.0.0",
                actions=actions,
                horizon=self.horizon,
                control_mode=self.control_mode
            )

class ACTPolicy:
    """
    Action Chunking Transformer (ACT) implementation.
    Handles temporal ensembles and multi-step action predictions.
    """
    def __init__(self, config: dict):
        self.config = config
        self.policy_config = config.get("policy", {})
        self.horizon = self.policy_config.get("horizon", 50) # ACT usually uses larger horizons
        self.action_dim = self.policy_config.get("action_dim", 7)
        
        # Load your ACT model weights here:
        # self.model = DETR(...)
        # self.model.load_state_dict(torch.load(path))
        
    def act(self, task: TaskLatent, perception: PerceptionState, robot: RobotState) -> ActionChunk:
        # 1. Prepare joint inputs and VLM embeddings
        # 2. Run Transformer encoder-decoder
        # 3. Handle chunking/temporal ensemble if needed
        
        # Mocking 50 steps of output
        actions = [[0.0] * self.action_dim for _ in range(self.horizon)]
        
        return ActionChunk(
            schema_version="1.0.0",
            actions=actions,
            horizon=self.horizon,
            control_mode="position"
        )
