import json
import shutil
import os

def copy_files_from_json(json_file, destination_folder):
    # Read file paths from JSON
    with open(json_file, 'r') as f:
        file_list = json.load(f)
    
    # Ensure destination exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Copy files
    for file_path in file_list:
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(destination_folder, file_name)
            shutil.copy2(file_path, destination_path)
            print(f"Copied: {file_path} -> {destination_path}")
        else:
            print(f"Missing: {file_path}")

if __name__ == "__main__":
    json_file_path = os.environ.get("TEST_LIST_PATH", "test_list.json")
    destination_folder = os.environ.get("TESTSET_DEST", "data/esm_msa/test")

    copy_files_from_json(json_file_path, destination_folder)
