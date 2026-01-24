import numpy as np
import torch
from PIL import Image
import os
import sys

sys.path.append(os.getcwd())

try:
    import dino_train as dt
    import clip_train as ct
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def verify_embeddings():
    print("Verifying Embeddings...")
    
    dummy_img_path = "dummy_verify.jpg"
    Image.new('RGB', (224, 224), color='blue').save(dummy_img_path)
    
    try:
        print("\nTesting DINO Embedding...")
        dino_emb = dt.get_dino_embedding(dummy_img_path)
        if dino_emb is not None:
            print(f"DINO Shape: {dino_emb.shape}")
            if dino_emb.shape == (768,):
                 print("SUCCESS: DINO shape is correct (768,)")
            else:
                 print(f"FAILURE: DINO shape mismatch. Expected (768,), got {dino_emb.shape}")
        else:
            print("FAILURE: DINO embedding returned None")

        print("\nTesting CLIP Embedding...")
        clip_emb = ct.get_clip_embedding(dummy_img_path)
        if clip_emb is not None:
            print(f"CLIP Shape: {clip_emb.shape}")
            if clip_emb.shape == (768,):
                 print("SUCCESS: CLIP shape is correct (768,)")
            else:
                 print(f"FAILURE: CLIP shape mismatch. Expected (768,), got {clip_emb.shape}")
        else:
            print("FAILURE: CLIP embedding returned None")
            
    finally:
        if os.path.exists(dummy_img_path):
            os.remove(dummy_img_path)

if __name__ == "__main__":
    verify_embeddings()
