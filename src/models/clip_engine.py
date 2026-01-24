import os
import sys
import numpy as np
import torch
import src.utils.faiss_ops as fi
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src import config
from src.models.pooling import gem

def load_model():
    if config.CLIP_MODEL_PATH.exists():
        print(f"Loading CLIP from local path: {config.CLIP_MODEL_PATH}")
        try:
            model = CLIPModel.from_pretrained(config.CLIP_MODEL_PATH)
            processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_PATH, use_fast=True)
            return model, processor
        except Exception as e:
            print(f"Failed to load local model: {e}. Fallback to online.")
    
    print(f"Loading CLIP from online: {config.CLIP_ONLINE_ID}")
    model = CLIPModel.from_pretrained(config.CLIP_ONLINE_ID)
    processor = CLIPProcessor.from_pretrained(config.CLIP_ONLINE_ID, use_fast=True)
    return model, processor

model, processor = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

clip_index = fi.create_flat_ip_index(config.CLIP_DIM)
image_paths = []

def get_clip_embedding(image_path):
    try:
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
            vision_outputs = model.vision_model(**inputs)
        
        last_hidden_state = vision_outputs.last_hidden_state
        patch_tokens = last_hidden_state[:, 1:, :]
        
        pooled_embedding = gem(patch_tokens)
        
        emb = pooled_embedding.detach().cpu().numpy().astype('float32')
        fi.normalize_l2(emb)
        return emb
    except Exception as e:
        print(f"Error processing CLIP embedding for {image_path}: {e}")
        return None

def add_image_to_faiss(image_path):
    emb = get_clip_embedding(image_path)
    if emb is not None:
        clip_index.add(emb)
        image_paths.append(image_path)
        print(f"Added {image_path} to clip index.")
