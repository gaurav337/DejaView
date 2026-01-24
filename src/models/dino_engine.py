import os
import sys
import numpy as np
import torch
import src.utils.faiss_ops as fi
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src import config
from src.models.pooling import gem

def load_model():
    if config.DINO_MODEL_PATH.exists():
        print(f"Loading DINOv2 from local path: {config.DINO_MODEL_PATH}")
        try:
            model = AutoModel.from_pretrained(config.DINO_MODEL_PATH)
            processor = AutoImageProcessor.from_pretrained(
                config.DINO_MODEL_PATH,
                use_fast=True
            )
            return model, processor
        except Exception as e:
            print(f"Failed to load local model: {e}. Fallback to online.")

    print(f"Loading DINOv2 from online: {config.DINO_ONLINE_ID}")
    model = AutoModel.from_pretrained(config.DINO_ONLINE_ID)
    processor = AutoImageProcessor.from_pretrained(
        config.DINO_ONLINE_ID,
        use_fast=True
    )
    return model, processor

model, processor = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

dino_index = fi.create_flat_ip_index(config.DINO_DIM)
image_paths = []

def get_dino_embedding(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
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
        patch_tokens = last_hidden_states[:, 5:, :]
        
        pooled_embedding = gem(patch_tokens)
        
        emb = pooled_embedding.detach().cpu().numpy().astype('float32')
        fi.normalize_l2(emb)
        
        return emb

    except Exception as e:
        print(f"Error processing DINO embedding for {image_path}: {e}")
        return None

def add_image_to_faiss(image_path):
    emb = get_dino_embedding(image_path)
    if emb is not None:
        dino_index.add(emb)
        image_paths.append(image_path)
        print(f"Added {image_path} to dino index.")