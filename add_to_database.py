import os
import sys

sys.path.append(os.getcwd())

try:
    from duplicate_checker import add_to_indices
except ImportError:
    print("Error: Could not import 'duplicate_checker'. Make sure you are running this script from the project root directory.")
    sys.exit(1)

def main():
    base_dir = os.getcwd()
    images_dir = os.path.join(base_dir, 'images')
    uploads_dir_name = 'uploads'
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return

    print(f"Scanning for images in: {images_dir}")
    print(f"ignoring folder: {uploads_dir_name}")
    
    count = 0
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    for root, dirs, files in os.walk(images_dir):
        if uploads_dir_name in dirs:
            dirs.remove(uploads_dir_name)
            
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                success = add_to_indices(file_path)
                
                if success:
                    count += 1
                else:
                    print(f"Failed to add: {file_path}")

    print(f"\nFinished. Total images added/processed: {count}")

if __name__ == "__main__":
    main()
