import os
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

IMAGE_FOLDER = 'images'
INDEX_FILENAME = 'dino_index.index'
PATHS_FILENAME = 'dino_image_paths.npy'
SIMILARITY_THRESHOLD = 0.80

ONLINE_MODEL_ID = "facebook/dinov2-with-registers-base" 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "dinoV2b14reg")

DINO_DIM = 768 

def load_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"Loading DINOv2 from local path: {LOCAL_MODEL_PATH}")
        try:
            model = AutoModel.from_pretrained(LOCAL_MODEL_PATH)
            processor = AutoImageProcessor.from_pretrained(
                LOCAL_MODEL_PATH,
                use_fast=True
            )
            return model, processor
        except Exception as e:
            print(f"Failed to load local model: {e}. Fallback to online.")

    print(f"Loading DINOv2 from online: {ONLINE_MODEL_ID}")
    model = AutoModel.from_pretrained(ONLINE_MODEL_ID)
    processor = AutoImageProcessor.from_pretrained(
        ONLINE_MODEL_ID,
        use_fast=True
    )
    return model, processor

model, processor = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

dino_index = faiss.IndexFlatIP(DINO_DIM)

def get_dino_embedding(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        # FIX: Squish to 224x224 (No Center Crop) - matches Colab behavior
        inputs = processor(
            images=img,
            return_tensors="pt",
            do_center_crop=False,
            do_resize=True,
            size={"height": 224, "width": 224}
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :]

        emb = cls_embedding.detach().cpu().numpy().astype('float32')
        faiss.normalize_L2(emb)
        
        return emb

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def add_image_to_faiss(image_path):
    emb = get_dino_embedding(image_path)
    
    if emb is not None:
        dino_index.add(emb)
        image_paths.append(image_path)
        print(f"Added {image_path} to dino index.")
    
image_paths = []

# for root, dirs, files in os.walk(IMAGE_FOLDER):
#     for file in files:
#         if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 add_image_to_faiss(os.path.join(root, file))

# faiss.write_index(dino_index, "index" ,INDEX_FILENAME)
# np.save(PATHS_FILENAME, "index" , np.array(image_paths))