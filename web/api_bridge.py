import os
import shutil
import tempfile
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

try:
    from src import config
    from src.core.pipeline import check_image_pipeline, add_to_indices
except ImportError as e:
    raise ImportError(f"Could not import backend modules. Error: {e}")

if not config.UPLOAD_DIR.exists():
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
            perm_path = config.UPLOAD_DIR / filename
            
            while perm_path.exists():
                perm_path = config.UPLOAD_DIR / f"{base}_{counter}{ext}"
                counter += 1
            
            shutil.move(tmp_path, str(perm_path))
            
            print(f"Unique image detected. Adding {perm_path} to indices...")
            add_success = add_to_indices(str(perm_path))
            
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
            result = {
                "status": "Error", 
                "error": str(e),
                "method": "N/A",
                "similarity_percentage": 0.0,
                "matched_image_path": None
            }

    return result
