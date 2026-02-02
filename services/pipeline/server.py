from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import time
import logging
import os
import uvicorn
import base64
import numpy as np
from PIL import Image
import io
from typing import List, Optional, Dict, Any

from services.task.model import TaskModelFactory
from services.action.factory import ActionExpertFactory
from common.schemas.perception import PerceptionState
from common.schemas.robot_state import RobotState
from common.schemas.action import ActionChunk
from common.schemas.task import TaskLatent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pipeline_server")

app = FastAPI(title="Production VLA Pipeline Server")

class PipelineState:
    def __init__(self):
        self.vlm = None
        self.action_expert = None
        self.current_instruction = "standby"
        self.last_task_latent = None
        self.last_task_time = 0
        self.task_vlm_freq = 1.0  # Hz - update VLM every 1 second
        
        # Performance metrics
        self.inference_count = 0
        self.avg_latency_ms = 0.0

state = PipelineState()

class InferenceRequest(BaseModel):
    image_base64: str
    camera_pose: List[List[float]]
    instruction: Optional[str] = None
    robot_state: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing VLA Pipeline models...")
    try:
        # Load models using factories
        # This will use configs/master_config.yaml by default
        state.vlm = TaskModelFactory.create()
        state.action_expert = ActionExpertFactory.create()
        
        # Warm up models
        logger.info("Warming up models...")
        # (Optional: run a dummy inference here if needed)
        
        logger.info("VLA Pipeline Server ready.")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        # We don't exit here to allow debugging via API if possible
        # but the server will fail on inference calls

@app.post("/set_instruction")
async def set_instruction(instruction: str):
    state.current_instruction = instruction
    logger.info(f"Instruction updated to: {instruction}")
    return {"status": "success", "instruction": instruction}

@app.post("/run_step", response_model=Dict[str, Any])
async def run_step(request: InferenceRequest):
    start_time = time.time()
    
    if state.vlm is None or state.action_expert is None:
        raise HTTPException(status_code=503, detail="Models not initialized")

    try:
        # 1. Decode Image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image.convert("RGB"))

        # 2. Update VLM (Task Latent)
        # We might not want to run the heavy VLM every single step (e.g. at 20Hz)
        # OpenVLA/PaliGemma are slow. ACT/Diffusion are fast.
        now = time.time()
        instruction = request.instruction or state.current_instruction
        
        # Frequency control for VLM
        if state.last_task_latent is None or (now - state.last_task_time) > (1.0 / state.task_vlm_freq):
            logger.debug(f"Running VLM for instruction: {instruction}")
            
            # Create PerceptionState for VLM
            # VLM usually expects image + instruction
            perception = PerceptionState(
                schema_version="1.0.0",
                scene_tokens=[], # Placeholder
                camera_pose=request.camera_pose,
                raw_image=image_np,
                timestamp=now
            )
            
            state.last_task_latent = state.vlm.infer(perception, instruction)
            state.last_task_time = now
            state.current_instruction = instruction

        # 3. Action Expert
        # Action Expert runs every step (e.g. 20Hz)
        robot_state = RobotState(**request.robot_state)
        
        # We need a perception state for the Action Expert too (it might use tokens/images)
        perception_for_action = PerceptionState(
            schema_version="1.0.0",
            scene_tokens=[], 
            camera_pose=request.camera_pose,
            raw_image=image_np,
            timestamp=now
        )
        
        action_chunk = state.action_expert.act(
            state.last_task_latent, 
            perception_for_action, 
            robot_state
        )

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Update metrics
        state.inference_count += 1
        state.avg_latency_ms = (state.avg_latency_ms * (state.inference_count - 1) + latency_ms) / state.inference_count

        if state.inference_count % 100 == 0:
            logger.info(f"Inference metrics: count={state.inference_count}, avg_latency={state.avg_latency_ms:.2f}ms")

        # FastAPI handles the conversion of ActionChunk to dict/json via response_model
        # But we return a dict to match the orchestrator expectation if needed
        return action_chunk.model_dump()

    except Exception as e:
        logger.error(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "vlm_loaded": state.vlm is not None,
        "action_expert_loaded": state.action_expert is not None,
        "instruction": state.current_instruction,
        "metrics": {
            "count": state.inference_count,
            "avg_latency_ms": state.avg_latency_ms
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
