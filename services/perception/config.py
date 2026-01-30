from pydantic import BaseModel
from common.config.base import ServiceConfig

class PerceptionModelConfig(BaseModel):
    type: str
    checkpoint: str
    image_size: int

class PerceptionConfig(ServiceConfig):
    model: PerceptionModelConfig