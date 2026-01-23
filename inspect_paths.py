import duplicate_checker
import os

duplicate_checker.load_resources()

print("\n--- Sample Paths in Index ---")
if duplicate_checker.phash_manager and duplicate_checker.phash_manager.paths:
    print(f"Total paths: {len(duplicate_checker.phash_manager.paths)}")
    for i in range(min(5, len(duplicate_checker.phash_manager.paths))):
        print(f"Path {i}: {duplicate_checker.phash_manager.paths[i]}")
        
    target_files = [
        "Screenshot 2026-01-20 142815.png",
        "Screenshot 2026-01-20 142852.png",
        "WhatsApp Image 2026-01-20 at 10.19.29 AM.jpeg",
        "WhatsApp Image 2026-01-20 at 10.19.37 AM.jpeg"
    ]
    
    found = 0
    for p in duplicate_checker.phash_manager.paths:
        base = os.path.basename(p)
        if base in target_files:
            print(f"Found match: {p}")
            found += 1
            
    print(f"Found {found} target filenames in pHash manager.")
else:
    print("No paths found in index.")
