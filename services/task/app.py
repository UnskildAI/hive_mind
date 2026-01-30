from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import time
import logging
from common.schemas.perception import PerceptionState
from common.schemas.task import TaskLatent
from services.task.model import TaskModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task_service")

app = FastAPI()

# Initialize model
try:
    model = TaskModel()
    logger.info("Task model initialized.")
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

@app.post("/process", response_model=TaskLatent)
async def process(data: TaskInput):
    global last_execution_time, last_result, last_instruction
    
    current_time = time.time()
    
    # Simple caching/throttling strategy
    # If same instruction and request is too soon, return cached
    # However, if perception changed significantly, we might want to re-run.
    # For this strict assignment, we'll enforce the 2Hz roughly, 
    # but allow re-run if > 0.5s OR instruction changed.
    
    if (data.instruction == last_instruction) and (current_time - last_execution_time < 0.5) and (last_result is not None):
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
