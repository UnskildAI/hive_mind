import torch
import torch.nn as nn
from typing import List
import yaml
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent
from common.schemas.robot_state import RobotState
from common.schemas.action import ActionChunk

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
