import os
from transformers import AutoImageProcessor, AutoModel

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "dinoV2b14reg")
ONLINE_MODEL_ID = "facebook/dinov2-with-registers-base"

def download_and_save():
    print(f"Downloading {ONLINE_MODEL_ID}...")
    
    # Create directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created directory: {MODEL_DIR}")
    
    # Download model and processor
    print("Downloading Model...")
    model = AutoModel.from_pretrained(ONLINE_MODEL_ID)
    
    print("Downloading Processor...")
    processor = AutoImageProcessor.from_pretrained(ONLINE_MODEL_ID)
    
    # Save locally
    print(f"Saving to {MODEL_DIR}...")
    model.save_pretrained(MODEL_DIR)
    processor.save_pretrained(MODEL_DIR)
    
    print("Download and save complete!")

if __name__ == "__main__":
    download_and_save()
