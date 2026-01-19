import imagehash
import faiss
import numpy as np
import json

PHASH_BITS = 64
WHASH_BITS = 64

phash_index = faiss.IndexBinaryFlat(PHASH_BITS)
whash_index = faiss.IndexBinaryFlat(WHASH_BITS)

CLIP_DIM = 512
clip_index = faiss.IndexFlatIP(CLIP_DIM)

image_paths = []

def hash_to_faiss_vector(hex_str):
    return np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)


def add_image_to_faiss(image_name,p_hash,w_hash):
    
    # convert to FAISS format
    ph_vec = hash_to_faiss_vector(p_hash).reshape(1, -1)
    wh_vec = hash_to_faiss_vector(w_hash).reshape(1, -1)

    # add to indexes
    phash_index.add(ph_vec)
    whash_index.add(wh_vec)

    # keep mapping
    image_paths.append(image_name)

if __name__ == "__main__":

    image_hash_path=r"D:\DejaView-main\image_hashes.json"

    with open(image_hash_path, 'r') as f:
        data = json.load(f)
    print(f"Successfully loaded {len(data)} images.\n")

    counter=1
    for entry in data:
        image_name = entry.get('name')
        p_hash = entry.get('phash')
        w_hash = entry.get('whash')
        try:
            add_image_to_faiss(image_name,p_hash,w_hash)
            print(f"Indexed {counter} : {image_name}")
            counter+=1
            
        except Exception as e:
            print(f"Skipped {image_name}: {e}")
        


    print("Total indexed images:", phash_index.ntotal)

    faiss.write_index_binary(phash_index, "phash.index")
    faiss.write_index_binary(whash_index, "whash.index")

    np.save("image_paths.npy", np.array(image_paths))


    phash_index = faiss.read_index_binary("phash.index")
    whash_index = faiss.read_index_binary("whash.index")
    image_paths = np.load("image_paths.npy").tolist()
