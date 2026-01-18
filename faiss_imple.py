import imagehash
import faiss
import numpy as np

PHASH_BITS = 64
WHASH_BITS = 64

phash_index = faiss.IndexBinaryFlat(PHASH_BITS)
whash_index = faiss.IndexBinaryFlat(WHASH_BITS)

CLIP_DIM = 512
clip_index = faiss.IndexFlatIP(CLIP_DIM)

image_paths = []

def hash_to_faiss_vector(hash_obj):
    """
    Converts imagehash to FAISS binary vector
    """
    # hash_obj.hash is a 2D boolean numpy array
    bits = np.array(hash_obj.hash, dtype=np.uint8)
    return np.packbits(bits).astype(np.uint8)


def add_image_to_faiss(image_path):
    img = hash_preprocessing(image_path)

    # compute hashes
    ph = imagehash.phash(img)
    wh = imagehash.whash(img)

    # convert to FAISS format
    ph_vec = hash_to_faiss_vector(ph).reshape(1, -1)
    wh_vec = hash_to_faiss_vector(wh).reshape(1, -1)

    # add to indexes
    phash_index.add(ph_vec)
    whash_index.add(wh_vec)

    # keep mapping
    image_paths.append(image_path)

for path in image_path:
    try:
        add_image_to_faiss(path)
        print(f"Indexed: {path}")
    except Exception as e:
        print(f"Skipped {path}: {e}")

print("Total indexed images:", phash_index.ntotal)

faiss.write_index_binary(phash_index, "phash.index")
faiss.write_index_binary(whash_index, "whash.index")

np.save("image_paths.npy", np.array(image_paths))


phash_index = faiss.read_index_binary("phash.index")
whash_index = faiss.read_index_binary("whash.index")
image_paths = np.load("image_paths.npy").tolist()
