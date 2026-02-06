from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import yaml
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent
from common.schemas.robot_state import RobotState
from common.schemas.action import ActionChunk
from services.action.policy import ActionExpert, ScriptedController, FrozenPolicy, ACTPolicy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("action_service")

app = FastAPI()

import os

# Initialize controller using factory
from services.action.factory import ActionExpertFactory
try:
    # Use master_config.yaml if it exists
    master_config_path = "configs/master_config.yaml"
    if os.path.exists(master_config_path):
        policy = ActionExpertFactory.create(master_config_path)
    else:
        # Fallback to local config.yaml
        policy = ActionExpertFactory.create("services/action/config.yaml")
    
    logger.info(f"Action Policy '{type(policy).__name__}' initialized.")
except Exception as e:
    logger.error(f"Failed to initialize controller: {e}")
    raise e

class ActionInput(BaseModel):
    task: TaskLatent
    perception: PerceptionState
    robot: RobotState

@app.post("/act", response_model=ActionChunk)
async def act(data: ActionInput):
    try:
        # High freq logic
        action_chunk = policy.act(data.task, data.perception, data.robot)
        return action_chunk
    except Exception as e:
        logger.error(f"Action inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}
