import torch
import torch.nn as nn
import time
from typing import List
from common.schemas.perception import PerceptionState
import yaml

class SimpleCNN(nn.Module):
    def __init__(self, input_shape, output_dim, num_tokens):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Mock projection to fixed number of tokens
        self.fc = nn.Linear(64 * (input_shape[1]//4) * (input_shape[2]//4), output_dim * num_tokens)
        self.output_dim = output_dim
        self.num_tokens = num_tokens

    def forward(self, x):
        # x is [B, C, H, W]
        features = self.conv(x)
        features = self.fc(features)
        return features.view(-1, self.num_tokens, self.output_dim)

class PerceptionModel:
    def __init__(self, config_path="services/perception/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config.get("device", "cpu")
        model_config = self.config["model"]
        
        if model_config["type"] == "simple_cnn":
             self.model = SimpleCNN(
                 model_config["input_shape"], 
                 model_config["output_dim"],
                 model_config["num_tokens"]
             )
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
        
        self.model.to(self.device)
        self.model.eval()

    def perceive(self, image_tensor: torch.Tensor, camera_pose: List[List[float]]) -> PerceptionState:
        """
        Args:
            image_tensor: [B, C, H, W] float tensor
            camera_pose: [B, Ds] list of lists
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            scene_tokens = self.model(image_tensor)
            
            # For this strict task, we assume batch size 1 for the return object or handle list return
            # The schema expects List[List[float]] for scene_tokens, which implies flat list of vectors? 
            # Or [N, Dv]? Schema says [N, Dv]. 
            # Let's assume N is the number of tokens.
            
            # We will process the first item in batch for the single request
            tokens = scene_tokens[0].tolist() # [NumTokens, OutputDim]
            
        return PerceptionState(
            schema_version="1.0.0",
            scene_tokens=tokens,
            camera_pose=camera_pose, # Pass through
            timestamp=time.time()
        )
