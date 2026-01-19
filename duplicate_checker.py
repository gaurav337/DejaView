import os
import faiss
import numpy as np
import imagehash
from PIL import Image
import torch

# Imports from existing files
try:
    import Final_preprocessing_hashing as fph
    import Faiss_implementation as fi
    import clip_train as ct
except ImportError as e:
    raise ImportError(f"Could not import helper modules. Ensure they are in the same directory. Error: {e}")

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHASH_BITS = 64
WHASH_BITS = 64
CLIP_DIM = 512

PHASH_THRESHOLD = 4
WHASH_THRESHOLD = 4
CLIP_THRESHOLD = 0.74

# Index paths
PHASH_INDEX_PATH = os.path.join(BASE_DIR, "phash.index")
WHASH_INDEX_PATH = os.path.join(BASE_DIR, "whash.index")
HASH_PATHS_FILE = os.path.join(BASE_DIR, "image_paths.npy")

CLIP_INDEX_PATH = os.path.join(BASE_DIR, "CLIP", "clip_index.index")
CLIP_PATHS_FILE = os.path.join(BASE_DIR, "CLIP", "clip_image_paths.npy")

# Initialize indices and paths
phash_index = None
whash_index = None
clip_index = None
hash_paths = []
clip_paths = []


def load_resources():
    global phash_index, whash_index, clip_index, hash_paths, clip_paths
    
    print("Loading indices...")
    
    if os.path.exists(PHASH_INDEX_PATH):
        phash_index = faiss.read_index_binary(PHASH_INDEX_PATH)
        print(f"Loaded phash index: {phash_index.ntotal} images")
    else:
        print("Warning: phash.index not found")
    

    if os.path.exists(WHASH_INDEX_PATH):
        whash_index = faiss.read_index_binary(WHASH_INDEX_PATH)
        print(f"Loaded whash index: {whash_index.ntotal} images")
    else:
        print("Warning: whash.index not found")
    

    if os.path.exists(HASH_PATHS_FILE):
        hash_paths = np.load(HASH_PATHS_FILE).tolist()
        print(f"Loaded hash paths: {len(hash_paths)} entries")
    else:
        print("Warning: image_paths.npy not found")
    

    if os.path.exists(CLIP_INDEX_PATH):
        clip_index = faiss.read_index(CLIP_INDEX_PATH)
        print(f"Loaded CLIP index: {clip_index.ntotal} images")
    else:
        print("Warning: clip_index.index not found")
    

    if os.path.exists(CLIP_PATHS_FILE):
        clip_paths = np.load(CLIP_PATHS_FILE).tolist()
        print(f"Loaded CLIP paths: {len(clip_paths)} entries")
    else:
        print("Warning: clip_image_paths.npy not found")
    
 
    if not hasattr(ct, 'model') or not hasattr(ct, 'processor'):
        print("Warning: CLIP model/processor not found in clip_train module")
    
    print("Resources loaded successfully")


def resolve_path(path_entry):
    
    path_entry = path_entry.replace("\\", "/")
    
    if "/" not in path_entry:
        return os.path.abspath(os.path.join(BASE_DIR, "images", path_entry))
    
    if path_entry.startswith("batch_1/"):
        new_path = path_entry.replace("batch_1/", "images/", 1)
        return os.path.abspath(os.path.join(BASE_DIR, new_path))
        
    return os.path.abspath(os.path.join(BASE_DIR, path_entry))


def check_phash(image_path):
    if phash_index is None or len(hash_paths) == 0:
        return False, 0.0, None
    
    p_hash, _ = fph.pw_hash(image_path)
    ph_str = str(p_hash)
    ph_vec = fi.hash_to_faiss_vector(ph_str).reshape(1, -1)
    
    D, I = phash_index.search(ph_vec, 1)
    dist = D[0][0]
    idx = I[0][0]
    
    if idx != -1 and dist <= PHASH_THRESHOLD:
        sim_pct = (1.0 - (dist / 64.0)) * 100.0
        return True, round(sim_pct, 2), resolve_path(hash_paths[idx])
    
    return False, 0.0, None


def check_whash(image_path):
    if whash_index is None or len(hash_paths) == 0:
        return False, 0.0, None
    
    _, w_hash = fph.pw_hash(image_path)
    wh_str = str(w_hash)
    wh_vec = fi.hash_to_faiss_vector(wh_str).reshape(1, -1)
    
    D, I = whash_index.search(wh_vec, 1)
    dist = D[0][0]
    idx = I[0][0]
    
    if idx != -1 and dist <= WHASH_THRESHOLD:
        sim_pct = (1.0 - (dist / 64.0)) * 100.0
        return True, round(sim_pct, 2), resolve_path(hash_paths[idx])
    
    return False, 0.0, None


def check_clip(image_path):
    if clip_index is None or len(clip_paths) == 0:
        return False, 0.0, None
    
    emb = ct.get_clip_embedding(image_path)
    
    if len(emb.shape) == 1:
        emb = emb.reshape(1, -1)
    
    D, I = clip_index.search(emb, 1)
    score = D[0][0]
    idx = I[0][0]
    
    if idx != -1 and score >= CLIP_THRESHOLD:
        sim_pct = round(score * 100.0, 2)
        return True, sim_pct, resolve_path(clip_paths[idx])
    
    return False, 0.0, None


def check_image_pipeline(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    result = {
        "status": "Unique",
        "similarity_percentage": 0.0,
        "matched_image_path": None,
        "source_image_path": image_path,
        "method": None
    }
    
    try:
        # PHASH Check
        is_match, sim_pct, matched_path = check_phash(image_path)
        if is_match:
            status = "Duplicate" if sim_pct == 100.0 else "Similar"
            result.update({
                "status": status,
                "similarity_percentage": sim_pct,
                "matched_image_path": matched_path,
                "method": "phash"
            })
            return result
        
        # WHASH Check
        is_match, sim_pct, matched_path = check_whash(image_path)
        if is_match:
            result.update({
                "status": "Similar",
                "similarity_percentage": sim_pct,
                "matched_image_path": matched_path,
                "method": "whash"
            })
            return result
        
        # CLIP Check
        is_match, sim_pct, matched_path = check_clip(image_path)
        if is_match:
            result.update({
                "status": "Similar",
                "similarity_percentage": sim_pct,
                "matched_image_path": matched_path,
                "method": "clip"
            })
        else:
            result.update({
                    "status": "Unique",
                    "similarity_percentage": sim_pct,
                    "matched_image_path": matched_path,
                    "method": "clip"
                })
            
            
    
    except Exception as e:
        print(f"Error processing image: {e}")
        result["error"] = str(e)
    
    return result


# def check_batch(image_paths_list):
#     results = []
    
#     for i, path in enumerate(image_paths_list):
#         try:
#             result = check_image(path)
#             results.append(result)
#             print(f"Checked {i + 1}/{len(image_paths_list)}: {path} -> {result['status']}")
#         except Exception as e:
#             print(f"Skipped {path}: {e}")
#             results.append({
#                 "status": "Error",
#                 "source_image_path": path,
#                 "error": str(e)
#             })
    
#     return results

load_resources()
