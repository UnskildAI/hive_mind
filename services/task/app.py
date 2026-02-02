from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import time
import logging
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent
from services.task.model import TaskModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task_service")

app = FastAPI()

# Initialize model
try:
    model = TaskModelFactory.create()
    logger.info(f"Task model initialized: {type(model).__name__}")
except Exception as e:
    logger.error(f"Failed to initialize task model: {e}")
    raise e

class TaskInput(BaseModel):
    perception: PerceptionState
    instruction: str

# Cache state
last_execution_time = 0
last_result: TaskLatent = None
last_instruction = ""

TTL = model.config.get("caching", {}).get("ttl_seconds", 1.0)

@app.post("/process", response_model=TaskLatent)
async def process(data: TaskInput):
    global last_execution_time, last_result, last_instruction
    
    current_time = time.time()
    
    # Human Override: If instruction changed, bypass TTL
    instruction_changed = (data.instruction != last_instruction)
    
    # Strict 1Hz / TTL enforcement
    is_too_soon = (current_time - last_execution_time < TTL)
    
    if is_too_soon and not instruction_changed and (last_result is not None):
        return last_result

    try:
        result = model.infer(data.perception, data.instruction)
        
        # Update cache
        last_execution_time = current_time
        last_result = result
        last_instruction = data.instruction
        
        return result
    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}
