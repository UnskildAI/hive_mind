from pydantic import BaseModel
from typing import List

class PerceptionState(BaseModel):
    schema_version: str
    scene_tokens: List[List[float]] #[N, Dv]
    camera_pose: List[List[float]] #[N, Ds]
    timestamp: float