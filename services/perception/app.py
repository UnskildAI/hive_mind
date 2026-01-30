from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64
import io
import torch
from PIL import Image
from torchvision import transforms
from services.perception.model import PerceptionModel
from common.schemas.perception import PerceptionState
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("perception_service")

app = FastAPI()

# Initialize model
try:
    model = PerceptionModel()
    logger.info("Perception model initialized.")
except Exception as e:
    logger.error(f"Failed to initialize perception model: {e}")
    raise e

class PerceptionInput(BaseModel):
    image_base64: str
    camera_pose: List[List[float]] # [1, Ds] or [N, Ds]

def decode_image(base64_string: str):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/perceive", response_model=PerceptionState)
async def perceive(data: PerceptionInput):
    try:
        # Decode image
        image = decode_image(data.image_base64)
        image_tensor = transform(image).unsqueeze(0) # [1, C, H, W]
        
        # Run inference
        perception_state = model.perceive(image_tensor, data.camera_pose)
        
        return perception_state
    except Exception as e:
        logger.error(f"Inference processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}
