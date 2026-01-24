import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src import config
from src.core.index_manager import IndexShardManager
from src.utils import hasher as fph
from src.utils import faiss_ops as fi
from src.models import clip_engine as ct
from src.models import dino_engine as dt
from src.utils.verification import hist_match, get_feature_count

phash_manager = None
whash_manager = None
clip_manager = None
dino_manager = None

def load_resources():
    global phash_manager, whash_manager, clip_manager, dino_manager
    
    print("Loading indices...")
    
    if not config.INDEX_DIR.exists():
        config.INDEX_DIR.mkdir(parents=True)
        
    phash_manager = IndexShardManager(str(config.INDEX_DIR), "phash", config.PHASH_BITS, index_type='binary')
    whash_manager = IndexShardManager(str(config.INDEX_DIR), "whash", config.WHASH_BITS, index_type='binary')
    
    clip_manager = IndexShardManager(str(config.INDEX_DIR), "clip", config.CLIP_DIM, index_type='flat')
    dino_manager = IndexShardManager(str(config.INDEX_DIR), "dino", config.DINO_DIM, index_type='flat')
    
    print("Resources loaded successfully")

    if not hasattr(ct, 'model') or not hasattr(ct, 'processor'):
        print("Warning: CLIP model/processor not found in clip_engine")
    
    if not hasattr(dt, 'model') or not hasattr(dt, 'processor'):
        print("Warning: DINO model/processor not found in dino_engine")


def resolve_path(path_entry):
    path_entry = str(path_entry).replace("\\", "/")
    
    if os.path.exists(path_entry):
        return os.path.abspath(path_entry)

    candidate_images = config.IMAGE_DIR / os.path.basename(path_entry)
    if candidate_images.exists():
        return str(candidate_images.resolve())

    if "/DejaView/" in path_entry:
        relative_part = path_entry.split("/DejaView/", 1)[1]
        candidate = config.PROJECT_ROOT / relative_part
        if candidate.exists():
            return str(candidate.resolve())

    return os.path.abspath(path_entry)


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
    
    if dist <= config.PHASH_THRESHOLD:
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
    
    if dist <= config.WHASH_THRESHOLD:
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
    
    if score >= config.CLIP_THRESHOLD:
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
    
    if score >= config.DINO_THRESHOLD:
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
        if feature_count < config.STRUCTURE_CHECK_THRESHOLD:
             result.update({
                "status": "Rejected",
                "similarity_percentage": 0.0,
                "matched_image_path": None,
                "method": "insufficient_features"
            })
             return result

        is_match1, sim_pct1, matched_path1 = check_phash(image_path)
        is_match2, sim_pct2, matched_path2 = check_whash(image_path)

        if(sim_pct1 > 92 and sim_pct2 > 92):
            sim_pct=(sim_pct1+sim_pct2)/2
            result.update({
                "status": "Similar (pHash & wHash)",
                "similarity_percentage": sim_pct,
                "matched_image_path": matched_path1,
                "method": "phash & whash"
            })
            return result
        
        is_match, sim_pct, matched_path = check_dino(image_path)

        if(sim_pct >= config.DINO_THRESHOLD*100):
            result.update({
                "status": "Similar",
                "similarity_percentage": sim_pct,
                "matched_image_path": matched_path,
                "method": "DINO"
            })
            return result
        elif(sim_pct < 20):
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