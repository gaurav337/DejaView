import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src import config
import src.core.pipeline as duplicate_checker
from src.utils import hasher as fph
from src.utils import faiss_ops as fi
from src.models import clip_engine as ct
from src.models import dino_engine as dt

def add_augmented_hashes_to_indices(image_path):
    try:
        augmented_hashes = fph.get_augmented_hashes(image_path)
        
        for p_hash, w_hash, aug_name in augmented_hashes:
            p_str = str(p_hash)
            w_str = str(w_hash)

            phash_vec = fi.hash_to_faiss_vector(p_str)
            whash_vec = fi.hash_to_faiss_vector(w_str)
            
            duplicate_checker.phash_manager.add(phash_vec, image_path)
            duplicate_checker.whash_manager.add(whash_vec, image_path)
        
        return True
    except Exception as e:
        print(f"Error adding augmented hashes: {e}")
        return False

def main():
    IMAGES_DIR = config.IMAGE_DIR
    
    if not IMAGES_DIR.exists():
        print(f"Error: directory not found {IMAGES_DIR}")
        return

    print("Loading resources (models and indices)...")
    duplicate_checker.load_resources()
    
    print(f"Scanning directory: {IMAGES_DIR}")
    
    processed_count = 0
    all_image_paths = []
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    for root, dirs, files in os.walk(IMAGES_DIR):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_path = os.path.join(root, file)
                image_path = image_path.replace('\\', '/')
                
                print(f"Processing ({processed_count + 1}): {image_path}")
                
                hash_success = add_augmented_hashes_to_indices(image_path)
                
                dl_success = False
                try:
                    clip_emb = ct.get_clip_embedding(image_path)
                    if clip_emb is not None:
                        duplicate_checker.clip_manager.add(clip_emb, image_path)
                    else:
                        print(f"Skipping CLIP for {image_path} (embedding failed)")

                    dino_emb = dt.get_dino_embedding(image_path)
                    if dino_emb is not None:
                        duplicate_checker.dino_manager.add(dino_emb, image_path)
                    else:
                         print(f"Skipping DINO for {image_path} (embedding failed)")
                    
                    dl_success = True
                except Exception as e:
                    import traceback
                    print(f"Error adding CLIP/DINO: {e}")
                    traceback.print_exc()
                
                if hash_success or dl_success:
                    all_image_paths.append(image_path)
                    processed_count += 1
                else:
                    print(f"Failed to index: {image_path}")

    print(f"\nProcessed {processed_count} images.")
    print("Persisting indices to disk...")
    
    if duplicate_checker.phash_manager:
        duplicate_checker.phash_manager.persist()
    
    if duplicate_checker.whash_manager:
        duplicate_checker.whash_manager.persist()
        
    if duplicate_checker.clip_manager:
        duplicate_checker.clip_manager.persist()
        
    if duplicate_checker.dino_manager:
        duplicate_checker.dino_manager.persist()
        
    paths_file = config.INDEX_DIR / "image_paths.npy"
    np.save(paths_file, np.array(all_image_paths))
    print(f"Saved consolidated image paths to: {paths_file}")
    
    print("\nAll operations completed successfully.")

if __name__ == "__main__":
    main()
