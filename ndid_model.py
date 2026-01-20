import os
import shutil
import tempfile

try:
    from duplicate_checker import check_image_pipeline, add_to_indices, BASE_DIR
except ImportError as e:
    raise ImportError(f"Could not import duplicate_checker. Ensure duplicate_checker.py exists in the same directory. Error: {e}")

UPLOADS_DIR = os.path.join(BASE_DIR, "images", "uploads")
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

def run_ndid(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        shutil.copyfileobj(uploaded_file, tmp_file)
        tmp_path = tmp_file.name

    try:
        result = check_image_pipeline(tmp_path)
        
        if result['status'] == "Unique":
            filename = uploaded_file.name
            base, ext = os.path.splitext(filename)
            counter = 1
            perm_path = os.path.join(UPLOADS_DIR, filename)
            while os.path.exists(perm_path):
                perm_path = os.path.join(UPLOADS_DIR, f"{base}_{counter}{ext}")
                counter += 1
            
            shutil.move(tmp_path, perm_path)
            
            print(f"Unique image detected. Adding {perm_path} to indices...")
            add_success = add_to_indices(perm_path)
            
            if add_success:
                print("Successfully updated indices.")
            else:
                print("Failed to update indices.")
                
        else:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"Error in run_ndid: {e}")
        if 'result' not in locals():
            result = {"status": "Error", "error": str(e)}

    return result
