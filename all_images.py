import os
import numpy as np
import duplicate_checker
import Final_preprocessing_hashing as fph
import Faiss_implementation as fi


def add_augmented_hashes_to_indices(image_path):
    try:
        augmented_hashes = fph.get_augmented_hashes(image_path)
        
        for p_hash, w_hash, aug_name in augmented_hashes:
            phash_vec = fi.hash_to_faiss_vector(p_hash)
            whash_vec = fi.hash_to_faiss_vector(w_hash)
            
            duplicate_checker.phash_manager.add(phash_vec, image_path)
            duplicate_checker.whash_manager.add(whash_vec, image_path)
        
        return True
    except Exception as e:
        print(f"Error adding augmented hashes: {e}")
        return False


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGES_DIR = os.path.join(BASE_DIR, 'images')
    INDEX_DIR = os.path.join(BASE_DIR, 'index')
    
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Images directory not found at {IMAGES_DIR}")
        return

    print("Loading resources (models and indices)...")
    duplicate_checker.load_resources()
    
    print(f"Scanning directory: {IMAGES_DIR}")
    
    processed_count = 0
    all_image_paths = []
    
    for root, dirs, files in os.walk(IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                image_path = os.path.join(root, file)
                
                image_path = image_path.replace('\\', '/')
                
                print(f"Processing ({processed_count + 1}): {image_path}")
                
                hash_success = add_augmented_hashes_to_indices(image_path)
                
                try:
                    import clip_train as ct
                    import dino_train as dt
                    
                    clip_emb = ct.get_clip_embedding(image_path)
                    duplicate_checker.clip_manager.add(clip_emb, image_path)
                    
                    dino_emb = dt.get_dino_embedding(image_path)
                    duplicate_checker.dino_manager.add(dino_emb, image_path)
                    
                    dl_success = True
                except Exception as e:
                    print(f"Error adding CLIP/DINO: {e}")
                    dl_success = False
                
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
        
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
        
    paths_file = os.path.join(INDEX_DIR, "image_paths.npy")
    np.save(paths_file, np.array(all_image_paths))
    print(f"Saved consolidated image paths to: {paths_file}")
    
    print("\nAll operations completed successfully.")

if __name__ == "__main__":
    main()
