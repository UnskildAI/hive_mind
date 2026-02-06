from transformers import AutoModelForCausalLM
import torch

model_id = "vikhyatk/moondream2"
print(f"Loading latest {model_id}...")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
print(f"Model type: {type(model)}")
print(f"Text model type: {type(model.text_model)}")
print(f"Has generate: {hasattr(model.text_model, 'generate')}")

if hasattr(model.text_model, 'generate'):
    print("SUCCESS: generate attribute found.")
else:
    print("FAILURE: generate attribute still missing.")
    # Print MRO to see inheritance
    print(f"Text model MRO: {type(model.text_model).mro()}")
