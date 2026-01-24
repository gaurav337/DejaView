import os
from transformers import AutoImageProcessor, AutoModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "dinoV2b14reg")
ONLINE_MODEL_ID = "facebook/dinov2-with-registers-base"

def download_and_save():
    print(f"Downloading {ONLINE_MODEL_ID}...")
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model = AutoModel.from_pretrained(ONLINE_MODEL_ID)
    processor = AutoImageProcessor.from_pretrained(ONLINE_MODEL_ID)
    
    model.save_pretrained(MODEL_DIR)
    processor.save_pretrained(MODEL_DIR)
    
    print("Download and save complete!")

if __name__ == "__main__":
    download_and_save()
