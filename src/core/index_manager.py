import os
import src.utils.faiss_ops as fi
import numpy as np
import heapq
import glob


class IndexShardManager:
    def __init__(self, base_dir, prefix, dimension, index_type="flat", max_vectors=1000000):
        self.base_dir = base_dir
        self.prefix = prefix
        self.dimension = dimension
        self.index_type = index_type
        self.max_vectors = max_vectors
        
        self.indices = []
        self.active_index = None
        self.active_suffix_id = 0
        self.paths = []
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            print(f"Created directory: {self.base_dir}")
        
        self.load_indices()
        self.load_paths()
        
        print(f"Initialized {self.prefix}: {self.get_total_vectors()} total vectors across {len(self.indices)} shards")
    
    def get_index_filename(self, suffix_id):
        return os.path.join(self.base_dir, f"{self.prefix}_{suffix_id}.index")

    def get_paths_filename(self):
        return os.path.join(self.base_dir, f"{self.prefix}_paths.npy")

    def create_new_index(self):
        if self.index_type == "binary":
            return fi.create_binary_index(self.dimension)
        else:
            return fi.create_flat_ip_index(self.dimension)

    def load_indices(self):
        pattern = os.path.join(self.base_dir, f"{self.prefix}_*.index")
        files = glob.glob(pattern)
    
        valid_files = []
        for f in files:
            try:
                base = os.path.basename(f)
                name_no_ext = os.path.splitext(base)[0]
                parts = name_no_ext.rpartition('_')
                suffix_id = int(parts[2])
                valid_files.append((suffix_id, f))
            except (ValueError, IndexError):
                continue
    
        valid_files.sort()
    
        self.indices = []
    
        if not valid_files:
            self.active_suffix_id = 0
            self.active_index = self.create_new_index()
            self.indices.append((self.active_index, 0))
            print(f"Created new shard: {self.prefix}_0")
        else:
            for suffix_id, filepath in valid_files:
                print(f"Loading shard: {filepath}")
                if self.index_type == "binary":
                    idx = fi.read_index(filepath, is_binary=True)
                else:
                    idx = fi.read_index(filepath, is_binary=False)
                
                if idx.d != self.dimension:
                    print(f"Warning: Index {filepath} has dimension {idx.d}, expected {self.dimension}. Skipping/Overwriting.")
                    continue

                self.indices.append((idx, suffix_id))
                print(f"Loaded shard {suffix_id}: {idx.ntotal} vectors")
        
            self.active_suffix_id = valid_files[-1][0]
            self.active_index = self.indices[-1][0]

    def load_paths(self):
        paths_file = self.get_paths_filename()
        if os.path.exists(paths_file):
            self.paths = np.load(paths_file, allow_pickle=True).tolist()
            print(f"Loaded {self.prefix} paths: {len(self.paths)} entries")
        else:
            self.paths = []
            print(f"No existing {self.prefix} paths found, starting fresh")

    def save_active_index(self):
        filename = self.get_index_filename(self.active_suffix_id)
        if self.index_type == "binary":
            fi.write_index(self.active_index, filename, is_binary=True)
        else:
            fi.write_index(self.active_index, filename, is_binary=False)
        print(f"Saved shard: {filename}")

    def save_paths(self):
        paths_file = self.get_paths_filename()
        np.save(paths_file, np.array(self.paths, dtype=object))
        print(f"Saved paths: {paths_file}")

    def persist(self):
        self.save_active_index()
        self.save_paths()
        print(f"Persisted {self.prefix} data")

    def rotate_shard(self):
        print(f"Shard {self.active_suffix_id} full ({self.active_index.ntotal} items). Creating new shard.")
        self.save_active_index()
        
        self.active_suffix_id += 1
        self.active_index = self.create_new_index()
        self.indices.append((self.active_index, self.active_suffix_id))
        print(f"Created new shard: {self.prefix}_{self.active_suffix_id}")

    def add(self, vector, image_path):
        if self.active_index.ntotal >= self.max_vectors:
            self.rotate_shard()
        
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
        
        if self.index_type == "binary":
            vector = vector.astype(np.uint8)
        else:
            vector = vector.astype(np.float32)
        
        self.active_index.add(vector)
        self.paths.append(image_path)


    def search(self, query_vector, k=1):
        all_results = []
        
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if self.index_type == "binary":
            query_vector = query_vector.astype(np.uint8)
        else:
            query_vector = query_vector.astype(np.float32)
        
        current_offset = 0
        
        for idx_obj, suffix_id in self.indices:
            if idx_obj.ntotal == 0:
                continue
            
            D, I = idx_obj.search(query_vector, min(k, idx_obj.ntotal))
            
            distances = D[0]
            local_indices = I[0]
            
            for dist, local_idx in zip(distances, local_indices):
                if local_idx != -1:
                    global_idx = current_offset + local_idx
                    if global_idx < len(self.paths):
                        all_results.append((dist, global_idx))
            
            current_offset += idx_obj.ntotal
        
        if not all_results:
            return []
        
        if self.index_type == "binary":
            best_k = heapq.nsmallest(k, all_results, key=lambda x: x[0])
        else:
            best_k = heapq.nlargest(k, all_results, key=lambda x: x[0])
        
        return best_k

    def get_total_vectors(self):
        total = 0
        for idx_obj, _ in self.indices:
            total += idx_obj.ntotal
        return total