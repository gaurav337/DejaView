from transformers import CLIPModel, CLIPProcessor
import os

def download_clip_model():
    model_id = "openai/clip-vit-base-patch32"
    local_dir = "./local_clip_model"
    
    if os.path.exists(local_dir):
        print(f"Directory {local_dir} already exists. checking contents...")
        if len(os.listdir(local_dir)) > 0:
             print("Model likely already downloaded. Skipping.")
             return

    print(f"Downloading {model_id} to {local_dir}...")
    
    try:
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id,use_fast=True)
        
        model.save_pretrained(local_dir)
        processor.save_pretrained(local_dir)
        
        print("Download complete! Model saved to:", local_dir)
    except Exception as e:
        print(f"An error occurred during download: {e}")

if __name__ == "__main__":
    download_clip_model()
