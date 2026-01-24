import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

try:
    from src import config
    from src.core.pipeline import add_to_indices
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    images_dir = config.IMAGE_DIR
    uploads_dir_path = config.UPLOAD_DIR
    
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return

    print(f"Scanning for images in: {images_dir}")
    print(f"Ignoring uploads folder: {uploads_dir_path}")
    
    count = 0
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}

    for root, dirs, files in os.walk(images_dir):
        if 'uploads' in dirs:
            dirs.remove('uploads')
            
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                file_path = os.path.join(root, file)
                success = add_to_indices(file_path)
                
                if success:
                    count += 1

    print(f"\nFinished. Total images added/processed: {count}")

if __name__ == "__main__":
    main()
