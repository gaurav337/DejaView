import os
import csv
import string
from PIL import Image, ImageDraw, ImageFont

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NON_MATCH_DIR = os.path.join(CURRENT_DIR, "evaluation", "test_images", "should_not_match")
EVAL_DIR = os.path.join(CURRENT_DIR, "evaluation")

os.makedirs(NON_MATCH_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

def create_non_match_ground_truth():
#     existing_images = [f for f in os.listdir(NON_MATCH_DIR) 
#                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    
    non_match_images = [f for f in os.listdir(NON_MATCH_DIR) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not non_match_images:
        print("No images found in should_not_match folder.")
        return

    ground_truth_data = []
    for img_name in non_match_images:
        ground_truth_data.append({
            "original": "NONE",
            "test_image": os.path.join(NON_MATCH_DIR, img_name),
            "transform": "unique",
            "expected": 0
        })
    
    csv_path = os.path.join(EVAL_DIR, "ground_truth_non_matches.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original", "test_image", "transform", "expected"])
        writer.writeheader()
        writer.writerows(ground_truth_data)
    
    print(f"Created ground truth for {len(ground_truth_data)} non-matching images at {csv_path}")

if __name__ == "__main__":
    create_non_match_ground_truth()
