import os
import sys
from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor, Dinov2Model, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src import config

def download_clip_model():
    model_id = config.CLIP_ONLINE_ID
    local_dir = config.CLIP_MODEL_PATH
    
    if local_dir.exists():
        print(f"Directory {local_dir} already exists. Checking contents...")
        if len(os.listdir(local_dir)) > 0:
             print("Model likely already downloaded. Skipping.")
             return

    print(f"Downloading {model_id} to {local_dir}...")
    
    try:
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
        
        model.save_pretrained(local_dir)
        processor.save_pretrained(local_dir)
        
        print("Download complete! Model saved to:", local_dir)
    except Exception as e:
        print(f"An error occurred during download: {e}")

def download_dino_model():
    model_id = config.DINO_ONLINE_ID
    local_dir = config.DINO_MODEL_PATH
    
    if local_dir.exists():
        print(f"Directory {local_dir} already exists. Checking contents...")
        if len(os.listdir(local_dir)) > 0:
             print("Model likely already downloaded. Skipping.")
             return

    print(f"Downloading {model_id} to {local_dir}...")
    
    try:
        model = AutoModel.from_pretrained(model_id)
        processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        
        model.save_pretrained(local_dir)
        processor.save_pretrained(local_dir)
        
        print("Download complete! Model saved to:", local_dir)
    except Exception as e:
        print(f"An error occurred during download: {e}")

if __name__ == "__main__":
    print("Enter 1 to download CLIP model, 2 to download DINO model, or 3 to download both:")
    try:
        x = int(input())
        if x == 1:
            download_clip_model()
        elif x == 2:
            download_dino_model()
        elif x == 3:
            download_clip_model()
            download_dino_model()
        else:
            print("Invalid input")
    except ValueError:
        print("Invalid input")
