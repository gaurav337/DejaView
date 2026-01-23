import os
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Constants
IMAGE_FOLDER = 'images'
INDEX_FILENAME = 'clip_index.index'
PATHS_FILENAME = 'clip_image_paths.npy'
EMBEDDINGS_FILENAME = 'clip_embeddings.npy'
SIMILARITY_THRESHOLD = 0.74
ONLINE_MODEL_ID = "openai/clip-vit-base-patch32"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "clipViTb32")
CLIP_DIM = 512

def load_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"Loading CLIP from local path: {LOCAL_MODEL_PATH}")
        try:
            model = CLIPModel.from_pretrained(LOCAL_MODEL_PATH)
            processor = CLIPProcessor.from_pretrained(LOCAL_MODEL_PATH, use_fast=True)
            return model, processor
        except Exception as e:
            print(f"Failed to load local model: {e}. Fallback to online.")
    
    print(f"Loading CLIP from online: {ONLINE_MODEL_ID}")
    model = CLIPModel.from_pretrained(ONLINE_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(ONLINE_MODEL_ID,use_fast=True)
    return model, processor

model, processor = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()


clip_index = faiss.IndexFlatIP(CLIP_DIM)

image_paths = []


def get_clip_embedding(image_path):

    img = Image.open(image_path).convert("RGB")
    
    inputs = processor(
        images=img, 
        return_tensors="pt", 
        padding=True,
        do_center_crop=False,
        do_resize=True,
        size={"height": 224, "width": 224}
    )
    inputs = inputs.to(device)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    emb = image_features.detach().cpu().numpy().astype('float32')
    faiss.normalize_L2(emb)
    return emb


def add_image_to_faiss(image_path):
    
    emb = get_clip_embedding(image_path)
    if emb is not None:
        clip_index.add(emb)
        image_paths.append(image_path)
        print(f"Added {image_path} to clip index.")

