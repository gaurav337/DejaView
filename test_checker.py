import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from duplicate_checker import load_resources, check_image_pipeline
def run_model():
    # Constants
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_IMAGE = os.path.join(BASE_DIR, "koala.png")

    # Load resources
    load_resources()

    # Check test image
    print(f"Testing with image: {TEST_IMAGE}")

    if os.path.exists(TEST_IMAGE):
        result = check_image_pipeline(TEST_IMAGE)
    else:
        print(f"Test image not found: {TEST_IMAGE}")

    return result

if __name__ == "__main__":
    print(run_model())