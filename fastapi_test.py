from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
import torch
import time
import json
import re

# --- Simple system prompt (let it describe, we'll extract) ---
system_prompt = """Analyze the image for safety violations. Categories:
O1=Hate, O2=Violence, O3=Sexual, O4=Nudity, O5=Criminal, O6=Weapons, O7=Self-Harm, O8=Animal Cruelty, O9=Disasters"""

# --- Load model ---
model_id = "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on: {device}")

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float32 if device.type == "cpu" else torch.float16
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

# --- Load image ---
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

# --- Fast generation ---
gen_args = {
    "max_new_tokens": 50,      # Enough for basic description
    "do_sample": False,        # Greedy for speed
    "use_cache": True,
    "pad_token_id": processor.tokenizer.eos_token_id,
}

start = time.time()
with torch.no_grad():
    output = model.generate(**inputs, **gen_args)
end = time.time()

# --- Extract response ---
input_length = inputs['input_ids'].shape[1]
response_tokens = output[0][input_length:]
decoded = processor.decode(response_tokens, skip_special_tokens=True).strip()

# --- Smart classification from description ---
def classify_from_description(text):
    text_lower = text.lower()
    
    # Violence indicators
    violence_keywords = ['violence', 'gun', 'weapon', 'threatening', 'intimidation', 'harm', 'attack', 'assault', 'fight', 'blood', 'injury', 'death']
    
    # Sexual content indicators  
    sexual_keywords = ['sexual', 'nude', 'naked', 'breast', 'genitalia', 'erotic', 'intimate', 'pornographic']
    
    # Hate indicators
    hate_keywords = ['hate', 'discrimination', 'racist', 'harassment', 'offensive', 'slur', 'prejudice']
    
    # Criminal indicators
    criminal_keywords = ['theft', 'robbery', 'kidnapping', 'drugs', 'illegal', 'crime', 'criminal']
    
    # Self-harm indicators
    selfharm_keywords = ['suicide', 'self-harm', 'cutting', 'overdose', 'hanging', 'jumping']
    
    # Animal cruelty indicators
    animal_keywords = ['animal cruelty', 'animal abuse', 'animal harm', 'torturing animals']
    
    # Disaster indicators
    disaster_keywords = ['disaster', 'emergency', 'casualties', 'victims', 'tragedy', 'accident']
    
    # Check for violations
    if any(keyword in text_lower for keyword in violence_keywords):
        return {"rating": "Unsafe", "category": "O2"}
    elif any(keyword in text_lower for keyword in sexual_keywords):
        return {"rating": "Unsafe", "category": "O3"}  
    elif any(keyword in text_lower for keyword in hate_keywords):
        return {"rating": "Unsafe", "category": "O1"}
    elif any(keyword in text_lower for keyword in criminal_keywords):
        return {"rating": "Unsafe", "category": "O5"}
    elif any(keyword in text_lower for keyword in selfharm_keywords):
        return {"rating": "Unsafe", "category": "O7"}
    elif any(keyword in text_lower for keyword in animal_keywords):
        return {"rating": "Unsafe", "category": "O8"}
    elif any(keyword in text_lower for keyword in disaster_keywords):
        return {"rating": "Unsafe", "category": "O9"}
    else:
        return {"rating": "Safe", "category": "NA"}

result = classify_from_description(decoded)

print(f"\n--- Classification Result ---")
print(f"🔍 Description: {decoded[:100]}...")
print(f"📋 Classification: {json.dumps(result)}")
print(f"⚡ Time: {end - start:.2f}s")

print(f"\n✅ Rating: {result['rating']}")
print(f"🏷️  Category: {result['category']}")

# --- Category meanings ---
categories = {
    "O1": "Hate, Humiliation, Harassment",
    "O2": "Violence, Harm, or Cruelty", 
    "O3": "Sexual Content",
    "O4": "Nudity Content",
    "O5": "Criminal Planning",
    "O6": "Weapons or Substance Abuse",
    "O7": "Self-Harm",
    "O8": "Animal Cruelty",
    "O9": "Disasters or Emergencies",
    "NA": "Safe - No violations"
}

print(f"📝 Meaning: {categories.get(result['category'], 'Unknown')}")