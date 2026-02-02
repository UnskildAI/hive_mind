import torch
import torch.nn as nn
from typing import List
import yaml
import numpy as np
import hashlib
import time
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

class TaskModelBase:
    def infer(self, perception: PerceptionState, instruction: str) -> TaskLatent:
        raise NotImplementedError

class SimpleMLPTaskModel(TaskModelBase):
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
        self.change_threshold = model_config.get("subtask_change_threshold", 0.1)
        
        self.model = SimpleMultimodalMLP(
            self.perception_dim, 
            self.text_dim, 
            self.output_dim
        )
        self.model.to(self.device)
        self.model.eval()
        
        # State for temporal smoothing
        self.prev_embedding = None
        self.current_subtask_id = "initial_subtask"
        self.prev_instruction = ""

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
            unfiltered_goal = goal_embedding[0].cpu().numpy()

            # Temporal smoothing (Aggressive)
            if self.prev_embedding is not None and instruction == self.prev_instruction:
                goal_np = self.smoothing_alpha * self.prev_embedding + (1 - self.smoothing_alpha) * unfiltered_goal
            else:
                goal_np = unfiltered_goal
            
            # Subtask Stability Logic
            # Only change subtask_id if:
            # 1. Instruction changes (Major switch)
            # 2. Embedding changes significantly (Substep switch)
            if instruction != self.prev_instruction:
                self._subtask_counter = getattr(self, "_subtask_counter", 0) + 1
                self.current_subtask_id = hashlib.md5(f"{instruction}_{self._subtask_counter}".encode()).hexdigest()[:8]
            elif self.prev_embedding is not None:
                delta = np.linalg.norm(goal_np - self.prev_embedding)
                if delta > self.change_threshold:
                    self._subtask_counter = getattr(self, "_subtask_counter", 0) + 1
                    self.current_subtask_id = hashlib.md5(f"{instruction}_{self._subtask_counter}".encode()).hexdigest()[:8]

            self.prev_embedding = goal_np
            self.prev_instruction = instruction

            return TaskLatent(
                schema_version="1.0.0",
                goal_embedding=goal_np.tolist(),
                constraints={"speed": "stable"}, 
                subtask_id=self.current_subtask_id,
                confidence=0.95
            )

class MockTaskModel(TaskModelBase):
    """Simple mock VLM that returns zero embeddings and fixed subtask."""
    def infer(self, perception: PerceptionState, instruction: str) -> TaskLatent:
        return TaskLatent(
            schema_version="1.0.0",
            goal_embedding=[0.0] * 256,
            constraints={"speed": "slow"},
            subtask_id="mock_subtask",
            confidence=0.5
        )

class PaliGemmaTaskModel(TaskModelBase):
    """
    Skeleton for PaliGemma-based VLM.
    Users can initialize their HF/Local PaliGemma model here.
    """
    def __init__(self, config_path: str):
        # In a real integration, you'd load the model here:
        # self.model = AutoModelForVision2Seq.from_pretrained(...)
        pass

    def infer(self, perception: PerceptionState, instruction: str) -> TaskLatent:
        # 1. Convert perception tokens back to image or use directly
        # 2. Tokenize instruction
        # 3. Generate embedding or action tokens
        return TaskLatent(
            schema_version="1.0.0",
            goal_embedding=[0.0] * 256, # Placeholder
            constraints={"speed": "adaptive"},
            subtask_id="paligemma_output",
            confidence=0.98
        )

class TaskModelFactory:
    @staticmethod
    def create(config_path="configs/master_config.yaml") -> TaskModelBase:
        """
        Create Task Model (VLM) from configuration.
        
        Args:
            config_path: Path to master config (default: configs/master_config.yaml)
        
        Returns:
            TaskModelBase instance (VLM provider or legacy model)
        """
        import yaml
        import os
        
        # Load master config
        if not os.path.exists(config_path):
            # Try legacy config path
            legacy_path = "services/task/config.yaml"
            if os.path.exists(legacy_path):
                print(f"WARNING: Using legacy config {legacy_path}, consider migrating to master_config.yaml")
                config_path = legacy_path
            else:
                raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Get VLM provider from config
        # Priority: Env Var > Config File > Default
        if "vlm" in config:
            # New master config format
            vlm_config = config["vlm"]
            provider = os.getenv("VLM_PROVIDER", vlm_config.get("provider", "simple_mlp"))
        else:
            # Legacy config format
            provider = os.getenv("TASK_MODEL", config.get("model", {}).get("type", "simple_mlp"))
        
        print(f"Loading VLM provider: {provider}")
        
        # Create provider instance
        if provider == "openvla":
            from services.task.providers.openvla import OpenVLAProvider
            return OpenVLAProvider(config.get("vlm", {}))
        
        elif provider == "paligemma":
            from services.task.providers.paligemma import PaliGemmaProvider
            return PaliGemmaProvider(config.get("vlm", {}))
        
        elif provider == "gemini":
            from services.task.providers.gemini import GeminiProvider
            return GeminiProvider(config.get("vlm", {}))
        
        elif provider == "gr00t":
            from services.task.providers.gr00t import GR00TProvider
            return GR00TProvider(config.get("vlm", {}))
        
        # Legacy providers (backward compatibility)
        elif provider == "simple_mlp":
            return SimpleMLPTaskModel(config_path)
        
        elif provider == "mock":
            return MockTaskModel()
        
        else:
            print(f"WARNING: Unknown VLM provider '{provider}', defaulting to simple_mlp")
            return SimpleMLPTaskModel(config_path)