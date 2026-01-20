import os
import sys
import cv2
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
    
    # Add final_hist to path for importing hist_matching
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_hist'))
    import hist_matching as hm
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
HISTOGRAM_VERIFICATION_THRESHOLD = 0.60
STRUCTURE_THRESHOLD = 10  

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
        source_img = cv2.imread(image_path)
        if source_img is None:
             result.update({
                "status": "Rejected",
                "message": "Image unnecessary to upload: Could not load image"
            })
             return result
        
        source_data = hm.preprocess_image(source_img)
        if source_data is None:
             result.update({
                "status": "Rejected",
                "message": "Image unnecessary to upload: Preprocessing failed"
             })
             return result
             
        _, gray_check, _ = source_data
        orb_check = cv2.ORB_create(nfeatures=500)
        kp_check = orb_check.detect(gray_check, None)
        
        if len(kp_check) < STRUCTURE_THRESHOLD:
            result.update({
                "status": "Rejected",
                "message": f"Image unnecessary to upload: Insufficient structure (Keypoints: {len(kp_check)})"
            })
            return result
            
    except Exception as e_struct:
        print(f"Error checking structure: {e_struct}")
        result.update({
            "status": "Rejected",
            "message": f"Image unnecessary to upload: Error {e_struct}"
        })
        return result
    
    def run_verification(matched_path_in):
        try:
            matched_img = cv2.imread(matched_path_in)
            if matched_img is None: return False, 0.0
            
            matched_data = hm.preprocess_image(matched_img)
            if matched_data is None: 
                return False, 0.0
            res_normal = hm.compare_image_data(source_data, matched_data)
            
            matched_img_flipped = cv2.flip(matched_img, 1)
            matched_data_flipped = hm.preprocess_image(matched_img_flipped)
            res_flipped = hm.compare_image_data(source_data, matched_data_flipped)
            
            best_res = res_flipped if res_flipped['total'] > res_normal['total'] else res_normal
            
            score = best_res['total']
            print(f"  [Verification] Score: {score:.3f} ({'Flipped' if best_res == res_flipped else 'Normal'})")
            return score >= HISTOGRAM_VERIFICATION_THRESHOLD, score
        except Exception as e_ver:
            print(f"  [Verification Error] {e_ver}")
            return False, 0.0

    try:
        is_match, sim_pct, matched_path = check_phash(image_path)
        if is_match:
            verified, v_score = run_verification(matched_path)
            if verified:
                status = "Duplicate" if sim_pct == 100.0 else "Similar"
                result.update({
                    "status": f"{status} (pHash Verified)",
                    "similarity_percentage": sim_pct,
                    "matched_image_path": matched_path,
                    "method": "phash",
                    "verification_score": round(v_score, 3)
                })
                return result
            else:
                print(f"pHash match rejected by verification. Score: {v_score:.3f}. Moving to wHash...")

        is_match, sim_pct, matched_path = check_whash(image_path)

        if is_match:
            verified, v_score = run_verification(matched_path)
            if verified:
                result.update({
                    "status": "Similar (wHash Verified)",
                    "similarity_percentage": sim_pct,
                    "matched_image_path": matched_path,
                    "method": "whash",
                    "verification_score": round(v_score, 3)
                })
                return result
            else:
                 print(f"wHash match rejected by verification. Score: {v_score:.3f}. Moving to CLIP...")
        
        is_match, sim_pct, matched_path = check_clip(image_path)
        
        if is_match:
            verified, v_score = run_verification(matched_path)
            if verified:
                result.update({
                    "status": "Similar (CLIP Verified)",
                    "similarity_percentage": sim_pct,
                    "matched_image_path": matched_path,
                    "method": "clip",
                    "verification_score": round(v_score, 3)
                })
            else:
                print(f"CLIP match rejected by verification. Score: {v_score:.3f}.")
                result.update({
                    "status": "Unique",
                    "similarity_percentage": 0.0,
                    "matched_image_path": None,
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
