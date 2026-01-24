import torch
import dino_train as dt
import clip_train as ct
from PIL import Image

def verify():
    print("Verifying DINOv2 shapes...")
    dino_model, dino_processor = dt.load_model()
    dino_model.eval()
    
    dummy_img = Image.new('RGB', (224, 224), color='red')
    
    inputs = dino_processor(images=dummy_img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = dino_model(**inputs)
        
    last_hidden_state = outputs.last_hidden_state
    print(f"DINO Last hidden state shape: {last_hidden_state.shape}")
    
    seq_len = last_hidden_state.shape[1]
    patch_emb_len = (224 // 14) ** 2
    registers = seq_len - 1 - patch_emb_len
    print(f"DINO Sequence Length: {seq_len}")
    print(f"Expected Patch Length (224/14)^2: {patch_emb_len}")
    print(f"Inferred Registers + CLS: {seq_len - patch_emb_len}")
    print(f"Inferred Registers: {registers}")
    
    print("-" * 20)
    
    print("Verifying CLIP shapes...")
    clip_model, clip_processor = ct.load_model()
    clip_model.eval()
    
    inputs = clip_processor(images=dummy_img, return_tensors="pt")
    
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(**inputs)
        
    last_hidden_state = vision_outputs.last_hidden_state
    print(f"CLIP Vision Last hidden state shape: {last_hidden_state.shape}")
    
    seq_len = last_hidden_state.shape[1]
    patch_emb_len = (224 // 32) ** 2
    print(f"CLIP Sequence Length: {seq_len}")
    print(f"Expected Patch Length (224/32)^2: {patch_emb_len}")
    print(f"Inferred Registers + CLS: {seq_len - patch_emb_len}")

if __name__ == "__main__":
    verify()
