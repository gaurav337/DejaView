
import os
import sys
import shutil
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from duplicate_checker import check_image_pipeline
except ImportError as e:
    raise ImportError(f"Could not import duplicate_checker. Ensure you are running from the correct directory or that duplicate_checker.py exists in parent. Error: {e}")

def run_ndid(uploaded_file):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        shutil.copyfileobj(uploaded_file, tmp_file)
        tmp_path = tmp_file.name

    try:
        result = check_image_pipeline(tmp_path)
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result
