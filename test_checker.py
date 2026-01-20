import os
from PIL import Image
from duplicate_checker import load_resources, check_image_pipeline
def run_model(image_file):
    
    img = Image.open(image_file)

    # Constants
    # BASE_DIR = r"E:\DejaView"
    # TEST_IMAGE = os.path.join(BASE_DIR, "similar4.png")

    # Load resources
    load_resources()

    # Check test image
    print(f"Testing with image: {image_file}")

    if os.path.exists(image_file):
        result = check_image_pipeline(image_file)
    else:
        print(f"Test image not found: {image_file}")

    return result
