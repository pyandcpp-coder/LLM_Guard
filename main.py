from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, pipeline
from PIL import Image
import torch
import cv2
import io
import os
import tempfile
import uvicorn
from typing import Dict, Any
import warnings



app = FastAPI(
    title="Safety Classification API",
    description="API for classifying images, text, and videos for safety content",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class SafetyResponse(BaseModel):
    rating: str
    category: str
    explanation: str

device = torch.device("cpu")

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
CATEGORY_LIST = list(CATEGORIES.keys())
CATEGORY_DESC = list(CATEGORIES.values())

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

def load_models():
    global llava_model, llava_processor, classifier
    
    print("Loading LLavaGuard model...")
    llava_model_id = "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf"
    llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        llava_model_id,
        torch_dtype=torch.float32
    ).to(device)
    llava_processor = AutoProcessor.from_pretrained(llava_model_id)
    
    print("Loading zero-shot classification model...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )
    
    print("Models loaded successfully!")

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

def classify_safety(explanation):
    result = classifier(explanation, CATEGORY_DESC, multi_label=True)
    
    scores = result["scores"]
    labels = result["labels"]

    desc_to_key = {v: k for k, v in CATEGORIES.items()}
    
    best_score = 0
    best_category = "NA: None applying"
    
    for label, score in zip(labels, scores):
        if label in desc_to_key:
            if label == "Safe content":

                if score > 0.6:
                    return {"rating": "Safe", "category": "NA: None applying", "explanation": explanation}
            else:

                if score > best_score and score > 0.4:
                    best_score = score
                    best_category = desc_to_key[label]
    

    if best_category != "NA: None applying" and best_score > 0.4:
        return {"rating": "Unsafe", "category": best_category, "explanation": explanation}
    else:
        return {"rating": "Safe", "category": "NA: None applying", "explanation": explanation}

def classify_image_internal(image):
    explanation = get_llava_output(image)
    return classify_safety(explanation)

def classify_text_internal(text):
    dummy = Image.new("RGB", (224, 224), color=(255, 255, 255))
    explanation = get_llava_output(dummy, text)
    return classify_safety(explanation)

def classify_video_internal(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if not success:
        return {"rating": "Error", "category": "Video Failed", "explanation": "Could not read frame"}
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        cv2.imwrite(temp_file.name, frame)
        cap.release()
        

        pil_image = Image.open(temp_file.name).convert("RGB")
        result = classify_image_internal(pil_image)
        
        os.unlink(temp_file.name)
        
        return result




@app.get("/")
async def root():
    return {"message": "Safety Classification API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": llava_model is not None and classifier is not None}

    
@app.post("/classify/image", response_model=SafetyResponse)
async def classify_image_endpoint(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        result = classify_image_internal(image)
        
        return SafetyResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/classify/video", response_model=SafetyResponse)
async def classify_video_endpoint(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file.flush()
            
            result = classify_video_internal(temp_file.name)
            
            os.unlink(temp_file.name)
            
            return SafetyResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/categories")
async def get_categories():
    return {"categories": CATEGORIES}

@app.on_event("startup")
async def startup_event():
    load_models()





if __name__ == "__main__":
    uvicorn.run(
        "main:app",  
        host="0.0.0.0",
        port=8002,
        reload=True
    )

