from pydantic import BaseModel
from typing import List

class ActionChunk(BaseModel):
    schema_version: str
    actions: List[List[float]]  #[T, A]
    horizon: int
    control_mode: str