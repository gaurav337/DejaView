import numpy as np
import faiss

def hash_to_faiss_vector(hash_input):
    if hasattr(hash_input, 'hash'):
        packed_arr = np.packbits(hash_input.hash.flatten())
        return packed_arr.reshape(1, -1)
    
    try:
        if isinstance(hash_input, str):
            return np.frombuffer(bytes.fromhex(hash_input), dtype=np.uint8).reshape(1, -1)
    except Exception:
        pass
        
    return np.array([[]])

def create_binary_index(dimension):
    return faiss.IndexBinaryFlat(dimension)

def create_flat_ip_index(dimension):
    return faiss.IndexFlatIP(dimension)

def read_index(filepath, is_binary=False):
    if is_binary:
        return faiss.read_index_binary(filepath)
    else:
        return faiss.read_index(filepath)

def write_index(index, filepath, is_binary=False):
    if is_binary:
        faiss.write_index_binary(index, filepath)
    else:
        faiss.write_index(index, filepath)

def normalize_l2(vector):
    faiss.normalize_L2(vector)

