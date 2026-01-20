import os
import csv
import random
import string
from PIL import Image, ImageDraw, ImageFont

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NON_MATCH_DIR = os.path.join(CURRENT_DIR, "evaluation", "test_images", "should_not_match")
EVAL_DIR = os.path.join(CURRENT_DIR, "evaluation")

os.makedirs(NON_MATCH_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

def generate_random_image(path):
    width, height = 300, 300
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img_type = random.choice(["solid", "noise", "text", "gradient"])
    
    if img_type == "solid":
        img = Image.new('RGB', (width, height), color)
    elif img_type == "noise":
        img = Image.effect_noise((width, height), random.randint(10, 100))
        img = img.convert('RGB')
    elif img_type == "text":
        img = Image.new('RGB', (width, height), (255, 255, 255))
        d = ImageDraw.Draw(img)
        text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        d.text((50, 150), text, fill=(0, 0, 0))
    else:
        img = Image.new('RGB', (width, height), color)
        d = ImageDraw.Draw(img)
        for _ in range(5):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            d.rectangle(
                [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            )

    img.save(path)
    return "synthetic"

def create_non_match_ground_truth():
    existing_images = [f for f in os.listdir(NON_MATCH_DIR) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(existing_images) < 20:
        print(f"Generating {20 - len(existing_images)} synthetic non-matching images...")
        for i in range(len(existing_images), 20):
            name = f"synthetic_{i:03d}.jpg"
            generate_random_image(os.path.join(NON_MATCH_DIR, name))
    
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
