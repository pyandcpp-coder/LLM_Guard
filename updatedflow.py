# from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, pipeline
# from PIL import Image
# import torch
# import cv2

# # === Setup Device (GPU on Kaggle T4) ===
# device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# # === Load LLavaGuard Model ===
# llava_model_id = "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf"
# llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
#     llava_model_id,
#     torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
# ).to(device)
# llava_processor = AutoProcessor.from_pretrained(llava_model_id)

# # === Load Zero-Shot Classification Pipeline ===
# classifier = pipeline(
#     "zero-shot-classification",
#     model="facebook/bart-large-mnli",
#     device=0 if device.type == "cuda" else -1
# )

# # === Safety Categories ===
# CATEGORIES = {
#     "O1: Hate, Humiliation, Harassment": "Hate, harassment, or humiliation",
#     "O2: Violence, Harm, or Cruelty": "Violence, harm, or cruelty",
#     "O3: Sexual Content": "Sexual content",
#     "O4: Nudity Content": "Nudity",
#     "O5: Criminal Planning": "Criminal planning",
#     "O6: Weapons or Substance Abuse": "Weapons or substance abuse",
#     "O7: Self-Harm": "Self-harm",
#     "O8: Animal Cruelty": "Animal cruelty",
#     "O9: Disasters or Emergencies": "Disasters or emergencies",
#     "NA: None applying": "Safe content"
# }
# CATEGORY_LIST = list(CATEGORIES.keys())
# CATEGORY_DESC = list(CATEGORIES.values())

# # === Inference Settings ===
# generation_args = {
#     "max_new_tokens": 300,
#     "do_sample": True,
#     "temperature": 0.2,
#     "top_p": 0.95,
#     "top_k": 50,
#     "num_beams": 2,
#     "use_cache": True,
# }

# # === Get Explanation (LLava) ===
# def get_llava_output(image, text=""):
#     prompt = "Describe any potential harmful content in the image/text. Do not assume."
#     conversation = [{
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": prompt + "\n\n" + text}
#         ]
#     }]
#     prompt_text = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
#     inputs = llava_processor(text=prompt_text, images=image, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         output = llava_model.generate(**inputs, **generation_args)

#     return llava_processor.decode(output[0], skip_special_tokens=True).strip()

# # === Classify using zero-shot on LLava explanation ===
# def classify_safety(explanation):
#     result = classifier(explanation, CATEGORY_DESC, multi_label=True)
#     top_idx = result["scores"].index(max(result["scores"]))
#     top_category = CATEGORY_LIST[top_idx]
#     score = result["scores"][top_idx]

#     if top_category != "NA: None applying" and score > 0.4:
#         return {"rating": "Unsafe", "category": top_category}
#     else:
#         return {"rating": "Safe", "category": "NA: None applying"}

# # === For Images ===
# def classify_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     explanation = get_llava_output(image)
#     return classify_safety(explanation)

# # === For Text ===
# def classify_text(text):
#     dummy = Image.new("RGB", (224, 224), color=(255, 255, 255))
#     explanation = get_llava_output(dummy, text)
#     return classify_safety(explanation)

# # === For Videos ===
# def classify_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     success, frame = cap.read()
#     if not success:
#         print("❌ Could not read frame.")
#         return {"rating": "Error", "category": "Video Failed"}
#     cv2.imwrite("temp_frame.jpg", frame)
#     cap.release()
#     return classify_image("temp_frame.jpg")

# # === Test All ===
# if __name__ == "__main__":
#     print("\n=== IMAGE TEST ===")
#     print(classify_image("c2.png"))

#     print("\n=== TEXT TEST ===")
#     print(classify_text("I think people of a certain race should be banned from public places."))

#     print("\n=== VIDEO TEST ===")
#     # print(classify_video("/kaggle/input/input1/test_video.mp4"))


from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, pipeline
from PIL import Image
import torch

# === Setup Device (CPU) ===
device = torch.device("cpu")

# === Load LLavaGuard Model ===
llava_model_id = "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf"
llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    llava_model_id,
    torch_dtype=torch.float32  # Use float32 for CPU
).to(device)
llava_processor = AutoProcessor.from_pretrained(llava_model_id)

# === Load Zero-Shot Classification Pipeline ===
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1  # Force CPU usage
)

# === Safety Categories ===
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
    "max_new_tokens": 100,  # Reduce token generation
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_beams": 1,  # Reduce beams for faster inference
    "use_cache": True,
}
# === Get Explanation (LLava) ===
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

# === Classify using zero-shot on LLava explanation ===
def classify_safety(explanation):
    result = classifier(explanation, CATEGORY_DESC, multi_label=True)
    top_idx = result["scores"].index(max(result["scores"]))
    top_category = CATEGORY_LIST[top_idx]
    score = result["scores"][top_idx]

    if top_category != "NA: None applying" and score > 0.4:
        return {"rating": "Unsafe", "category": top_category}
    else:
        return {"rating": "Safe", "category": "NA: None applying"}

# === For Images ===
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    explanation = get_llava_output(image)
    return classify_safety(explanation)

# === For Text ===
def classify_text(text):
    dummy = Image.new("RGB", (224, 224), color=(255, 255, 255))
    explanation = get_llava_output(dummy, text)
    return classify_safety(explanation)

# === For Videos ===
def classify_video(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if not success:
        print("❌ Could not read frame.")
        return {"rating": "Error", "category": "Video Failed"}
    cv2.imwrite("temp_frame.jpg", frame)
    cap.release()
    return classify_image("temp_frame.jpg")

# === Test All ===
if __name__ == "__main__":
    print("\n=== IMAGE TEST ===")
    print(classify_image("test.png"))

    print("\n=== TEXT TEST ===")
    # print(classify_text("I think people of a certain race should be banned from public places."))

    print("\n=== VIDEO TEST ===")
    # print(classify_video("/kaggle/input/input1/test_video.mp4"))