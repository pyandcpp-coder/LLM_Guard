from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
import torch
import time
import json
import re

system_prompt = """Analyze the image for safety violations. Categories:
O1=Hate, O2=Violence, O3=Sexual, O4=Nudity, O5=Criminal, O6=Weapons, O7=Self-Harm, O8=Animal Cruelty, O9=Disasters"""
model_id = "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on: {device}")

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float32 if device.type == "cpu" else torch.float16
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

# --- Load ima--
image = Image.open("c2.png").convert("RGB")

# --- Prepare conversation ---
conversation = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Safety assessment:"}]}
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=text_prompt, images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# --- Warm-up ---
print("Warming up...")
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=5)

# --- Fast generation (reduced tokens) ---
gen_args = {
    "max_new_tokens": 30,      # Reduced from 50
    "do_sample": False,        # Greedy for speed
    "use_cache": True,
    "pad_token_id": processor.tokenizer.eos_token_id,
    "num_beams": 1,           # Single beam
}

start = time.time()
with torch.no_grad():
    output = model.generate(**inputs, **gen_args)
end = time.time()

# --- Extract response ---
input_length = inputs['input_ids'].shape[1]
response_tokens = output[0][input_length:]
decoded = processor.decode(response_tokens, skip_special_tokens=True).strip()

# --- Ultra-fast keyword classification ---
def classify_from_description(text):
    text_lower = text.lower()
    
    # Priority order: check most common violations first
    violence_words = ['violence', 'gun', 'weapon', 'threatening', 'intimidation', 'harm', 'attack', 'held', 'pointing']
    if any(word in text_lower for word in violence_words):
        return {"rating": "Unsafe", "category": "O2"}
    
    sexual_words = ['sexual', 'nude', 'naked', 'breast', 'explicit']
    if any(word in text_lower for word in sexual_words):
        return {"rating": "Unsafe", "category": "O3"}
    
    hate_words = ['hate', 'racist', 'harassment', 'discrimination']
    if any(word in text_lower for word in hate_words):
        return {"rating": "Unsafe", "category": "O1"}
    
    # If no violations found
    return {"rating": "Safe", "category": "NA"}

result = classify_from_description(decoded)

print(f"\n--- FAST RESULT ---")
print(f"Rating: {result['rating']} | Category: {result['category']}")
print(f"Time: {end - start:.2f}s")

# Optional: Show description for debugging
if len(decoded) > 0:
    print(f"Detection: {decoded[:50]}...")