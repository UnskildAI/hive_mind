import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
from io import BytesIO

device = "cpu"
model_id = "vikhyatk/moondream2"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.to(device)

print(f"Model type: {type(model)}")
print(f"Text model type: {type(model.text_model)}")
print(f"Has generate: {hasattr(model.text_model, 'generate')}")

# Basic functional test
url = "https://t3.ftcdn.net/jpg/05/65/52/64/360_F_565526485_9U4G08e8P2N8U9QW6X7X6I0zX6V1P4q6.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

print("Encoding image...")
enc_image = model.encode_image(image)
question = "Is there a robot in the image?"
print(f"Answering question: {question}")
answer = model.answer_question(enc_image, question, tokenizer)
print(f"Answer: {answer}")

assert "yes" in answer.lower() or "robot" in answer.lower()
print("Verification SUCCESSFUL!")
