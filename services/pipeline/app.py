from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from typing import List
from services.pipeline.orchestrator import PipelineOrchestrator
from common.schemas.action import ActionChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline_service")

app = FastAPI()

# Service URLs from env or default (localhost for docker compose)
PERCEPTION_URL = os.getenv("PERCEPTION_URL", "http://localhost:8001")
TASK_URL = os.getenv("TASK_URL", "http://localhost:8002")
ACTION_URL = os.getenv("ACTION_URL", "http://localhost:8003")

orchestrator = PipelineOrchestrator(PERCEPTION_URL, TASK_URL, ACTION_URL)

class RunStepInput(BaseModel):
    image_base64: str
    camera_pose: List[List[float]]
    instruction: str
    robot_state: dict # Matches RobotState fields

@app.post("/run_step", response_model=ActionChunk)
async def run_step(data: RunStepInput):
    try:
        action = orchestrator.run_step(
            data.image_base64, 
            data.camera_pose, 
            data.instruction, 
            data.robot_state
        )
        return action
    except Exception as e:
        logger.error(f"Pipeline step failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}
