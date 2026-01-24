import os
import subprocess
import sys

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(project_root, 'web', 'app.py')
    if not os.path.exists(app_path):
        print(f"Error: Could not find Streamlit app at {app_path}")
        print("Please use the DejaView root directory.")
        sys.exit(1)
        
    print(f"Starting DejaView UI ")
    print(f"App path: {app_path}")
    
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\nStreamlit exited with error code {e.returncode}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
