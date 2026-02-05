from pydantic import BaseModel, ConfigDict
from typing import List, Any, Optional

class PerceptionState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    schema_version: str
    scene_tokens: List[List[float]] #[N, Dv]
    camera_pose: List[List[float]] #[N, Ds]
    raw_image: Optional[Any] = None # numpy array or PIL image
    timestamp: float