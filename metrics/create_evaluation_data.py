from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
import os
import random
import shutil
import csv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
SOURCE_DIR = os.path.join(PARENT_DIR, "images")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "evaluation", "test_images", "should_match")
EVAL_DIR = os.path.join(CURRENT_DIR, "evaluation")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

def create_transformations(input_path, output_dir, base_name):
    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return []
    
    created = []
    
    try:
        resized = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        path = os.path.join(output_dir, f"{base_name}_resized.jpg")
        resized.save(path, quality=90)
        created.append(("resized", path))
    except: pass
    
    try:
        w, h = img.size
        left = int(w * 0.15)
        top = int(h * 0.15)
        right = int(w * 0.85)
        bottom = int(h * 0.85)
        cropped = img.crop((left, top, right, bottom))
        path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
        cropped.save(path, quality=90)
        created.append(("cropped", path))
    except: pass
    
    try:
        path = os.path.join(output_dir, f"{base_name}_compressed.jpg")
        img.save(path, quality=15)
        created.append(("compressed", path))
    except: pass
    
    try:
        bright = ImageEnhance.Brightness(img).enhance(1.4)
        contrast = ImageEnhance.Contrast(bright).enhance(1.3)
        path = os.path.join(output_dir, f"{base_name}_color.jpg")
        contrast.save(path, quality=90)
        created.append(("color", path))
    except: pass
    
    try:
        watermarked = img.copy()
        draw = ImageDraw.Draw(watermarked)
        text = "DEJAVIEW"
        w, h = img.size
        
        draw.text((w//4, h//2), text, fill=(255, 255, 255))
        draw.text((w//4 + 2, h//2 + 2), text, fill=(0, 0, 0))
        
        path = os.path.join(output_dir, f"{base_name}_watermark.jpg")
        watermarked.save(path, quality=90)
        created.append(("watermark", path))
    except: pass
    
    return created

if not os.path.exists(SOURCE_DIR):
    print(f"Error: Source directory not found at {SOURCE_DIR}")
    exit(1)

all_images = [f for f in os.listdir(SOURCE_DIR) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

if not all_images:
    print(f"No images found in {SOURCE_DIR}")
    exit(1)

selected = random.sample(all_images, min(30, len(all_images)))

print(f"Creating transformations for {len(selected)} images...")
ground_truth_data = []

for img_name in selected:
    input_path = os.path.join(SOURCE_DIR, img_name)
    base_name = os.path.splitext(img_name)[0]
    
    created = create_transformations(input_path, OUTPUT_DIR, base_name)
    
    for transform_type, output_path in created:
        ground_truth_data.append({
            "original": input_path,
            "test_image": output_path,
            "transform": transform_type,
            "expected": 1
        })
    
    print(f"  OK {img_name}: {len(created)} transformations")

print(f"\nTotal test images (should match): {len(ground_truth_data)}")

csv_path = os.path.join(EVAL_DIR, "ground_truth_matches.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["original", "test_image", "transform", "expected"])
    writer.writeheader()
    writer.writerows(ground_truth_data)

print(f"Saved: {csv_path}")
