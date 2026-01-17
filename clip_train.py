import os
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Constants
IMAGE_FOLDER = 'batch_1'
INDEX_FILENAME = 'clip_index.index'
PATHS_FILENAME = 'clip_image_paths.npy'
EMBEDDINGS_FILENAME = 'clip_embeddings.npy'
SIMILARITY_THRESHOLD = 0.92
MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_DIM = 512

# Load model
print(f"Loading CLIP {MODEL_ID}")
model = CLIPModel.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model.eval()


clip_index = faiss.IndexFlatIP(CLIP_DIM)

image_paths = []
embeddings_list = []


def get_clip_embedding(image_path):

    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    emb = image_features.detach().cpu().numpy().astype('float32')
    faiss.normalize_L2(emb)
    return emb


def add_image_to_faiss(image_path):
    
    emb = get_clip_embedding(image_path)
    clip_index.add(emb)
    embeddings_list.append(emb)
    image_paths.append(image_path)


image_files = [f for f in os.listdir(IMAGE_FOLDER) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
image_files.sort()


for filename in image_files:
    path = os.path.join(IMAGE_FOLDER, filename)
    try:
        add_image_to_faiss(path)
        print(f"Indexed: {path}")
    except Exception  as e:
        print(f"Skipped {path}: {e}")

print("Total indexed images:", clip_index.ntotal)


faiss.write_index(clip_index, INDEX_FILENAME)
np.save(PATHS_FILENAME, np.array(image_paths))
np.save(EMBEDDINGS_FILENAME, np.vstack(embeddings_list))


print("\n" + "=" * 50)
print(" ANALYSIS RESULT ")
print("=" * 50)

embeddings_matrix = np.vstack(embeddings_list)
D, I = clip_index.search(embeddings_matrix, k=2)

count_unique = 0
count_duplicates = 0

for i, path in enumerate(image_paths):
    if len(I[i]) < 2 or I[i][1] == -1:
        print(f"[{path}] : UNIQUE")
        count_unique += 1
        continue
    
    best_match_idx = I[i][1]
    similarity_score = D[i][1]
    best_match_path = image_paths[best_match_idx]
    
    if similarity_score >= SIMILARITY_THRESHOLD:
        print(f"[{path}] : DUPLICATE / EDITED VERSION")
        print(f"    -> Match: {best_match_path} (Similarity: {similarity_score:.4f})")
        count_duplicates += 1
    else:
        print(f"[{path}] : UNIQUE")
        count_unique += 1

print("=" * 50)
print(f"Finished. Found {count_unique} unique and {count_duplicates} duplicate/edited images.")


# Load
# clip_index = faiss.read_index(INDEX_FILENAME)
# image_paths = np.load(PATHS_FILENAME).tolist()
# embeddings = np.load(EMBEDDINGS_FILENAME)
