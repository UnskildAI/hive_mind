import torch
import torch.nn as nn
from typing import List
import yaml
import numpy as np
import hashlib
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent

class SimpleMultimodalMLP(nn.Module):
    def __init__(self, perception_dim, text_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(perception_dim + text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh() # normalize to [-1, 1] roughly
        )

    def forward(self, perception_flat, text_emb):
        x = torch.cat([perception_flat, text_emb], dim=-1)
        return self.net(x)

class TaskModel:
    def __init__(self, config_path="services/task/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config.get("device", "cpu")
        model_config = self.config["model"]
        
        # Mock text embedding dim
        self.text_dim = 128
        self.perception_dim = model_config["perception_dim"]
        self.output_dim = model_config["output_dim"]
        self.smoothing_alpha = model_config["smoothing_alpha"]
        
        self.model = SimpleMultimodalMLP(
            self.perception_dim, 
            self.text_dim, 
            self.output_dim
        )
        self.model.to(self.device)
        self.model.eval()
        
        # State for temporal smoothing
        self.prev_embedding = None
        self.prev_subtask = "init"

    def _embed_text(self, text: str) -> torch.Tensor:
        # deterministic mock embedding
        h = hashlib.md5(text.encode()).digest()
        # repeat to fill size 128 (16 * 8)
        vec = np.frombuffer(h * 8, dtype=np.uint8)[:self.text_dim]
        vec = vec.astype(np.float32) / 255.0
        return torch.tensor(vec).to(self.device).unsqueeze(0) # [1, Dt]

    def infer(self, perception: PerceptionState, instruction: str) -> TaskLatent:
        with torch.no_grad():
            # Flatten perception scene tokens
            # perception.scene_tokens is List[List[float]] -> [N, Dv]
            scene_tokens = torch.tensor(perception.scene_tokens).to(self.device)
            perception_flat = scene_tokens.view(1, -1) # [1, N*Dv]
            
            if perception_flat.shape[1] != self.perception_dim:
                # Handle dimension mismatch if perception changes (mock fix)
                # In real system, we'd resize or project
                current_dim = perception_flat.shape[1]
                if current_dim < self.perception_dim:
                     perception_flat = torch.nn.functional.pad(perception_flat, (0, self.perception_dim - current_dim))
                else:
                     perception_flat = perception_flat[:, :self.perception_dim]

            text_emb = self._embed_text(instruction)
            
            goal_embedding = self.model(perception_flat, text_emb) # [1, OutputDim]
            goal_np = goal_embedding[0].cpu().numpy()

            # Temporal smoothing
            if self.prev_embedding is not None:
                goal_np = self.smoothing_alpha * self.prev_embedding + (1 - self.smoothing_alpha) * goal_np
            
            self.prev_embedding = goal_np
            
            # Simple subtask generation (hash of instruction)
            subtask_id = hashlib.md5(instruction.encode()).hexdigest()[:8]

            return TaskLatent(
                schema_version="1.0.0",
                goal_embedding=goal_np.tolist(),
                constraints={"speed": "slow"}, # Mock constraint
                subtask_id=subtask_id,
                confidence=0.9
            )