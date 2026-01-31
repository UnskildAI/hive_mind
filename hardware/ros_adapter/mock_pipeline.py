from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import random
import uvicorn
from typing import List, Optional
from common.schemas.action import ActionChunk

app = FastAPI()

class PipelineState:
    latency_ms: float = 0.0
    failure_rate: float = 0.0
    force_timeout: bool = False

state = PipelineState()

class InferenceRequest(BaseModel):
    perception: dict
    robot: dict

@app.post("/infer", response_model=ActionChunk)
async def infer(request: InferenceRequest):
    # Simulate processing delay
    if state.latency_ms > 0:
        time.sleep(state.latency_ms / 1000.0)
    
    # Simulate timeout (sleep longer than client timeout)
    if state.force_timeout:
        time.sleep(2.0)

    # Simulate random failure
    if state.failure_rate > 0 and random.random() < state.failure_rate:
        raise HTTPException(status_code=500, detail="Simulated internal server error")

    # Generate dummy action
    # For testing, we just return a valid ActionChunk with simple values
    
    # Simple deterministic action: Move all joints to 0.1
    # Example action dim = 7 (standard arm + gripper)
    action_dim = 7
    action_data = [0.1] * action_dim

    return ActionChunk(
        schema_version="1.0",
        actions=[action_data],
        horizon=1,
        control_mode="position"
    )

@app.post("/config/latency/{ms}")
async def set_latency(ms: float):
    state.latency_ms = ms
    return {"latency_ms": state.latency_ms}

@app.post("/config/failure_rate/{rate}")
async def set_failure_rate(rate: float):
    state.failure_rate = float(rate)
    return {"failure_rate": state.failure_rate}

@app.post("/config/timeout/{enabled}")
async def set_timeout(enabled: bool):
    state.force_timeout = enabled
    return {"force_timeout": state.force_timeout}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
