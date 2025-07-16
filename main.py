import os
import io
import cv2
import torch
import tempfile
import requests
from PIL import Image
from io import BytesIO
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    pipeline,
    AutoImageProcessor,
    AutoModelForImageClassification
)
import uvicorn


app = FastAPI(
    title="Unified Safety Classification API",
    description="NSFW + Harmful Content Classification using Transformers & LLMs",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORIES = {
    "O1: Hate, Humiliation, Harassment": "Hate, harassment, or humiliation",
    "O2: Violence, Harm, or Cruelty": "Violence, harm, or cruelty",
    "O3: Sexual Content": "Sexual content",
    "O4: Nudity Content": "Nudity",
    "O5: Criminal Planning": "Criminal planning",
    "O6: Weapons or Substance Abuse": "Weapons or substance abuse",
    "O7: Self-Harm": "Self-harm",
    "O8: Animal Cruelty": "Animal cruelty",
    "O9: Disasters or Emergencies": "Disasters or emergencies",
    "NA: None applying": "Safe content"
}

generation_args = {
    "max_new_tokens": 100,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_beams": 1,
    "use_cache": True,
}

llava_model = None
llava_processor = None
classifier = None
nsfw_processor = None
nsfw_model = None

class SafetyResponse(BaseModel):
    rating: str
    category: str
    explanation: str
    nsfw_prediction: str
    nsfw_confidence: float

class TextRequest(BaseModel):
    text: str

@app.on_event("startup")
def load_models():
    global llava_model, llava_processor, classifier, nsfw_model, nsfw_processor

    print("Loading LLavaGuard model...")
    llava_model_id = "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf"
    # llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    #     llava_model_id,
    #     torch_dtype=torch.float32
    # ).to(device)
    llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    llava_model_id,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)

    llava_processor = AutoProcessor.from_pretrained(llava_model_id)

    print("Loading BART classifier...")
    # classifier = pipeline(
    #     "zero-shot-classification",
    #     model="facebook/bart-large-mnli",
    #     device=-1
    # )
    classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if device.type == "cuda" else -1
    )
    print("Loading NSFW model...")
    # nsfw_processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
    # nsfw_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
    nsfw_processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
    nsfw_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection").to(device)

# def predict_nsfw(image: Image.Image):
#     inputs = nsfw_processor(images=image, return_tensors="pt")
#     outputs = nsfw_model(**inputs)
#     logits = outputs.logits
#     predicted_class_idx = logits.argmax(-1).item()
#     label = nsfw_model.config.id2label[predicted_class_idx]
#     confidence = logits.softmax(-1)[0][predicted_class_idx].item()
#     return label, confidence

def predict_nsfw(image: Image.Image):
    inputs = nsfw_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = nsfw_model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = nsfw_model.config.id2label[predicted_class_idx]
    confidence = logits.softmax(-1)[0][predicted_class_idx].item()
    return label, confidence


def get_llava_output(image, text=""):
    prompt = "Describe any potential harmful content in the image/text. Do not assume."
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt + "\n\n" + text}
        ]
    }]
    prompt_text = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = llava_processor(text=prompt_text, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = llava_model.generate(**inputs, **generation_args)
    return llava_processor.decode(output[0], skip_special_tokens=True).strip()

def classify_safety(explanation: str) -> Dict[str, Any]:
    result = classifier(explanation, list(CATEGORIES.values()), multi_label=True)
    best_score = 0
    best_category = "NA: None applying"
    for label, score in zip(result["labels"], result["scores"]):
        if label == "Safe content" and score > 0.6:
            return {"rating": "Safe", "category": "NA: None applying", "explanation": explanation}
        elif score > best_score and score > 0.4:
            best_score = score
            best_category = next((k for k, v in CATEGORIES.items() if v == label), "NA: None applying")
    return {
        "rating": "Unsafe" if best_category != "NA: None applying" else "Safe",
        "category": best_category,
        "explanation": explanation
    }

def classify_image_internal(image: Image.Image):
    explanation = get_llava_output(image)
    safety = classify_safety(explanation)
    nsfw_pred, nsfw_conf = predict_nsfw(image)
    safety.update({
        "nsfw_prediction": nsfw_pred,
        "nsfw_confidence": nsfw_conf
    })
    return safety

@app.get("/")
def root():
    return {"message": "Unified NSFW + Safety API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": all([llava_model, classifier, nsfw_model])}

@app.get("/categories")
def get_categories():
    return {"categories": CATEGORIES}

@app.post("/classify/image", response_model=SafetyResponse)
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image.")
    result = classify_image_internal(image)
    return SafetyResponse(**result)

@app.post("/classify/text", response_model=SafetyResponse)
async def classify_text(text_req: TextRequest):
    dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    explanation = get_llava_output(dummy_image, text_req.text)
    result = classify_safety(explanation)
    result.update({"nsfw_prediction": "sfw", "nsfw_confidence": 0.0})  
    return SafetyResponse(**result)

@app.post("/classify/video", response_model=SafetyResponse)
async def classify_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files are supported.")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        success, frame = cap.read()
        cap.release()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to read video frame.")
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = classify_image_internal(pil_image)
        os.unlink(tmp.name)
        return SafetyResponse(**result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8020, reload=True)
