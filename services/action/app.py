from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent
from common.schemas.robot_state import RobotState
from common.schemas.action import ActionChunk
from services.action.policy import ActionExpert

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("action_service")

app = FastAPI()

# Initialize model
try:
    expert = ActionExpert()
    logger.info("Action Expert initialized.")
except Exception as e:
    logger.error(f"Failed to initialize action expert: {e}")
    raise e

class ActionInput(BaseModel):
    task: TaskLatent
    perception: PerceptionState
    robot: RobotState

@app.post("/act", response_model=ActionChunk)
async def act(data: ActionInput):
    try:
        # High freq logic
        action_chunk = expert.act(data.task, data.perception, data.robot)
        return action_chunk
    except Exception as e:
        logger.error(f"Action inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}
