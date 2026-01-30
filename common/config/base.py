from pydantic import BaseModel

class ServiceConfig(BaseModel):
    service_name: str
    device: str
    log_level: str = "INFO"