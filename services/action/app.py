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

# Load config
with open("services/action/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Priority: Env Var > Config File > Default
mode = os.getenv("ACTION_MODE", config.get("mode", "learned"))

# Initialize controller based on mode
try:
    if mode == "scripted":
        policy = ScriptedController(config)
        logger.info("ScriptedController initialized.")
    elif mode == "frozen_policy":
        policy = FrozenPolicy(config)
        logger.info("FrozenPolicy initialized.")
    elif mode == "act":
        policy = ACTPolicy(config)
        logger.info("ACTPolicy initialized.")
    else:
        policy = ActionExpert("services/action/config.yaml")
        logger.info("Action Expert (ML) initialized.")
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
