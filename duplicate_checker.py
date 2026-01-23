import os
import sys
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
    import dino_train as dt
    from final_hist.hist_matching import hist_match, get_feature_count
except ImportError as e:
    raise ImportError(f"Could not import helper modules. Ensure they are in the same directory. Error: {e}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "index")

PHASH_BITS = 64
WHASH_BITS = 64
CLIP_DIM = 512
DINO_DIM = 768

PHASH_THRESHOLD = 4
WHASH_THRESHOLD = 4
CLIP_THRESHOLD = 0.69
DINO_THRESHOLD = 0.82
HIST_THRESHOLD = 0.80   

phash_manager = None
whash_manager = None
clip_manager = None
dino_manager = None

def load_resources():
    global phash_manager, whash_manager, clip_manager, dino_manager
    
    print("Loading indices ")
    
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    phash_manager = IndexShardManager(INDEX_DIR, "phash", PHASH_BITS, index_type='binary')
    whash_manager = IndexShardManager(INDEX_DIR, "whash", WHASH_BITS, index_type='binary')
    
    clip_manager = IndexShardManager(INDEX_DIR, "clip", CLIP_DIM, index_type='flat')
    dino_manager = IndexShardManager(INDEX_DIR, "dino", DINO_DIM, index_type='flat')
    
    print("Resources loaded successfully")

    if not hasattr(ct, 'model') or not hasattr(ct, 'processor'):
        print("Warning: CLIP model/processor not found in clip_train module")
    
    if not hasattr(dt, 'model') or not hasattr(dt, 'processor'):
        print("Warning: DINO model/processor not found in dino_train module")


def resolve_path(path_entry):
    path_entry = path_entry.replace("\\", "/")
    if os.path.exists(path_entry):
        return os.path.abspath(path_entry)
        
    if "/DejaView/" in path_entry:
        relative_part = path_entry.split("/DejaView/", 1)[1]
        candidate = os.path.join(BASE_DIR, relative_part)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
            
    basename = os.path.basename(path_entry)
    
    candidate_images = os.path.join(BASE_DIR, "images", basename)
    if os.path.exists(candidate_images):
        return os.path.abspath(candidate_images)
        
    candidate_root = os.path.join(BASE_DIR, basename)
    if os.path.exists(candidate_root):
        return os.path.abspath(candidate_root)
        
    candidate_clip = os.path.join(BASE_DIR, "CLIP", basename)
    if os.path.exists(candidate_clip):
        return os.path.abspath(candidate_clip)

    if os.path.isabs(path_entry):
        pass

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
        augmented_hashes = fph.get_augmented_hashes(image_path)
        
        for p_hash, w_hash, aug_name in augmented_hashes:
            ph_str = str(p_hash)
            wh_str = str(w_hash)
            
            phash_vec = fi.hash_to_faiss_vector(ph_str)
            whash_vec = fi.hash_to_faiss_vector(wh_str)
            
            phash_manager.add(phash_vec, image_path)
            whash_manager.add(whash_vec, image_path)
        
        clip_emb = ct.get_clip_embedding(image_path)
        clip_manager.add(clip_emb, image_path)

        dino_emb = dt.get_dino_embedding(image_path)
        dino_manager.add(dino_emb, image_path)

        # phash_manager.persist()
        # whash_manager.persist()
        # clip_manager.persist()
        # dino_manager.persist()
        
        # print(f"Successfully added image to indices: {image_path}")
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


def check_dino(image_path):
    if dino_manager is None:
        return False, 0.0, None
    
    emb = dt.get_dino_embedding(image_path)
    if emb is None:
        return False, 0.0, None
    
    results = dino_manager.search(emb, 1)
    
    if not results:
        return False, 0.0, None
        
    score, global_idx = results[0]
    
    if score >= DINO_THRESHOLD:
        sim_pct = round(score * 100.0, 2)
        
        if global_idx < len(dino_manager.paths):
            matched_path = resolve_path(dino_manager.paths[global_idx])
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
        feature_count = get_feature_count(image_path)
        if feature_count < 20:
             result.update({
                "status": "Rejected",
                "similarity_percentage": 0.0,
                "matched_image_path": None,
                "method": "insufficient_features"
            })
             return result

        is_match1, sim_pct1, matched_path1 = check_phash(image_path)
        
        is_match2, sim_pct2, matched_path2 = check_whash(image_path)

        if(sim_pct1>92 and sim_pct2>92):
            sim_pct=(sim_pct1+sim_pct2)/2
            result.update({
                "status": "Similar (pHash & wHash)",
                "similarity_percentage": sim_pct,
                "matched_image_path": matched_path1,
                "method": "phash & whash"
            })
            return result
        
        # hist_score, hist_details = hist_match(image_path, matched_path1)
        # if hist_score > HIST_THRESHOLD:
        #     result.update({
        #         "status": "Similar",
        #         "similarity_percentage": sim_pct    ,
        #         "matched_image_path": matched_path1,
        #         "method": f"phash & whash (Histogram Matching) ({hist_details.get('orientation', 'Original')})"
        #     })
        #     return result
            

        is_match, sim_pct, matched_path = check_dino(image_path)

        if(sim_pct >= DINO_THRESHOLD*100):
            result.update({
                "status": "Similar",
                "similarity_percentage": sim_pct,
                "matched_image_path": matched_path,
                "method": "DINO"
            })
            return result
        elif(sim_pct < 50):
            return result
        else:
            is_match, sim_pct, matched_path = check_clip(image_path)
            if is_match:
                result.update({
                    "status": "Similar",
                    "similarity_percentage": sim_pct,
                    "matched_image_path": matched_path,
                    "method": "CLIP"
                })
            return result

    except Exception as e:
        print(f"Error processing image: {e}")
        result["error"] = str(e)
    
    return result

load_resources()