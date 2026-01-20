import os
import faiss
import numpy as np
import imagehash
from PIL import Image
import torch
from index_manager import IndexShardManager

try:
    import Final_preprocessing_hashing as fph
    import Faiss_implementation as fi
    import clip_train as ct
except ImportError as e:
    raise ImportError(f"Could not import helper modules. Ensure they are in the same directory. Error: {e}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "index")

PHASH_BITS = 64
WHASH_BITS = 64
CLIP_DIM = 512

PHASH_THRESHOLD = 4
WHASH_THRESHOLD = 4
CLIP_THRESHOLD = 0.85

phash_manager = None
whash_manager = None
clip_manager = None

def load_resources():
    global phash_manager, whash_manager, clip_manager
    
    print("Loading indices...")
    
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    phash_manager = IndexShardManager(INDEX_DIR, "phash", PHASH_BITS, index_type='binary')
    whash_manager = IndexShardManager(INDEX_DIR, "whash", WHASH_BITS, index_type='binary')
    
    clip_manager = IndexShardManager(INDEX_DIR, "clip", CLIP_DIM, index_type='flat')
    
    print("Resources loaded successfully")

    if not hasattr(ct, 'model') or not hasattr(ct, 'processor'):
        print("Warning: CLIP model/processor not found in clip_train module")


def resolve_path(path_entry):
    path_entry = path_entry.replace("\\", "/")
    
    if os.path.isabs(path_entry):
        return path_entry
        
    if "/" not in path_entry:
        return os.path.abspath(os.path.join(BASE_DIR, "images", path_entry))
    
    if path_entry.startswith("batch_1/"):
        new_path = path_entry.replace("batch_1/", "images/", 1)
        return os.path.abspath(os.path.join(BASE_DIR, new_path))
        
    return os.path.abspath(os.path.join(BASE_DIR, path_entry))


def add_to_indices(image_path):
    if phash_manager is None:
        load_resources()

    try:
        p_hash, w_hash = fph.pw_hash(image_path)

        ph_str = str(p_hash)
        wh_str = str(w_hash)
        
        phash_vec = fi.hash_to_faiss_vector(ph_str)
        whash_vec = fi.hash_to_faiss_vector(wh_str)
        
        phash_manager.add(phash_vec, image_path)
        whash_manager.add(whash_vec, image_path)
        
        clip_emb = ct.get_clip_embedding(image_path)
        clip_manager.add(clip_emb, image_path)

        phash_manager.persist()
        whash_manager.persist()
        clip_manager.persist()
        
        print(f"Successfully added image to indices: {image_path}")
        return True
    
    except Exception as e:
        print(f"Error adding to indices: {e}")
        return False


def check_phash(image_path):
    if phash_manager is None:
        return False, 0.0, None
    
    p_hash, _ = fph.pw_hash(image_path)
    ph_str = str(p_hash)
    ph_vec = fi.hash_to_faiss_vector(ph_str)
    
    results = phash_manager.search(ph_vec, 1)
    
    if not results:
        return False, 0.0, None

    dist, global_idx = results[0]
    
    if dist <= PHASH_THRESHOLD:
        sim_pct = (1.0 - (dist / 64.0)) * 100.0
        
        if global_idx < len(phash_manager.paths):
            matched_path = resolve_path(phash_manager.paths[global_idx])
            # Verify file exists
            if os.path.exists(matched_path):
                return True, round(sim_pct, 2), matched_path
            else:
                print(f"Warning: Index returned missing file: {matched_path}")
    
    return False, 0.0, None


def check_whash(image_path):
    if whash_manager is None:
        return False, 0.0, None
    
    _, w_hash = fph.pw_hash(image_path)
    wh_str = str(w_hash)
    wh_vec = fi.hash_to_faiss_vector(wh_str)
    
    results = whash_manager.search(wh_vec, 1)
    
    if not results:
        return False, 0.0, None
        
    dist, global_idx = results[0]
    
    if dist <= WHASH_THRESHOLD:
        sim_pct = (1.0 - (dist / 64.0)) * 100.0
        
        if global_idx < len(whash_manager.paths):
            matched_path = resolve_path(whash_manager.paths[global_idx])
            if os.path.exists(matched_path):
                return True, round(sim_pct, 2), matched_path
            else:
                print(f"Warning: Index returned missing file: {matched_path}")
    
    return False, 0.0, None


def check_clip(image_path):
    if clip_manager is None:
        return False, 0.0, None
    
    emb = ct.get_clip_embedding(image_path)
    
    results = clip_manager.search(emb, 1)
    
    if not results:
        return False, 0.0, None
        
    score, global_idx = results[0]
    
    if score >= CLIP_THRESHOLD:
        sim_pct = round(score * 100.0, 2)
        
        if global_idx < len(clip_manager.paths):
            matched_path = resolve_path(clip_manager.paths[global_idx])
            if os.path.exists(matched_path):
                return True, sim_pct, matched_path
            else:
                print(f"Warning: Index returned missing file: {matched_path}")
    
    return False, 0.0, None


def check_image_pipeline(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if phash_manager is None:
        load_resources()
    
    result = {
        "status": "Unique",
        "similarity_percentage": 0.0,
        "matched_image_path": None,
        "source_image_path": image_path,
        "method": None
    }
    
    try:
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
        
        is_match, sim_pct, matched_path = check_whash(image_path)
        if is_match:
            result.update({
                "status": "Similar",
                "similarity_percentage": sim_pct,
                "matched_image_path": matched_path,
                "method": "whash"
            })
            return result
        
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

load_resources()
