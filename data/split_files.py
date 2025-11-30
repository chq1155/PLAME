import os
import random
import shutil

def split_files(source_folder, train_folder, valid_folder, test_folder, train_ratio=0.9, valid_ratio=0.05, test_ratio=0.05):
    # Ensure ratios sum to 1
    assert round(train_ratio + valid_ratio + test_ratio, 5) == 1.0, "Ratios must sum to 1"

    # Create destination folders if needed
    for folder in (train_folder, valid_folder, test_folder):
        os.makedirs(folder, exist_ok=True)

    # Collect all files in the source folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Shuffle the file list
    random.shuffle(all_files)

    # Compute how many files go to each split
    total_files = len(all_files)
    train_count = int(total_files * train_ratio)
    valid_count = int(total_files * valid_ratio)
    # test_count = total_files - train_count - valid_count

    # Slice file lists
    train_files = all_files[:train_count]
    valid_files = all_files[train_count:train_count+valid_count]
    test_files = all_files[train_count+valid_count:]

    # Move files to the proper split folder
    for file_list, dest_folder in [(train_files, train_folder), 
                                   (valid_files, valid_folder), 
                                   (test_files, test_folder)]:
        for file in file_list:
            shutil.move(os.path.join(source_folder, file), os.path.join(dest_folder, file))

    print(f"Split complete. \nTrain: {len(train_files)} files\nValid: {len(valid_files)} files\nTest: {len(test_files)} files")

if __name__ == "__main__":
    source_folder = os.environ.get("MSA_SOURCE", "data/esm_msa")
    train_folder = os.environ.get("MSA_TRAIN", "data/esm_msa/train")
    valid_folder = os.environ.get("MSA_VALID", "data/esm_msa/valid")
    test_folder = os.environ.get("MSA_TEST", "data/esm_msa/test")

    split_files(source_folder, train_folder, valid_folder, test_folder)
