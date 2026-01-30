from pydantic import BaseModel
from typing import List

class TaskLatent(BaseModel):
    schema_version: str
    goal_embedding: List[float] # [Dt]
    constraints: dict
    subtask_id: str
    confidence: float