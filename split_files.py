import os
import random
import shutil

def split_files(source_folder, train_folder, valid_folder, test_folder, train_ratio=0.9, valid_ratio=0.05, test_ratio=0.05):
    # 确保比例之和为1
    assert round(train_ratio + valid_ratio + test_ratio, 5) == 1.0, "Ratios must sum to 1"

    # 创建目标文件夹（如果不存在）
    for folder in (train_folder, valid_folder, test_folder):
        os.makedirs(folder, exist_ok=True)

    # 获取源文件夹中的所有文件
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # 随机打乱文件列表
    random.shuffle(all_files)

    # 计算每个集合应该包含的文件数量
    total_files = len(all_files)
    train_count = int(total_files * train_ratio)
    valid_count = int(total_files * valid_ratio)
    # test_count = total_files - train_count - valid_count

    # 分割文件列表
    train_files = all_files[:train_count]
    valid_files = all_files[train_count:train_count+valid_count]
    test_files = all_files[train_count+valid_count:]

    # 移动文件到对应文件夹
    for file_list, dest_folder in [(train_files, train_folder), 
                                   (valid_files, valid_folder), 
                                   (test_files, test_folder)]:
        for file in file_list:
            shutil.move(os.path.join(source_folder, file), os.path.join(dest_folder, file))

    print(f"Split complete. \nTrain: {len(train_files)} files\nValid: {len(valid_files)} files\nTest: {len(test_files)} files")

# 使用示例
source_folder = "/uac/gds/hqcao23/hqcao/openfold/esm_msa"
train_folder = "/uac/gds/hqcao23/hqcao/openfold/esm_msa/train"
valid_folder = "/uac/gds/hqcao23/hqcao/openfold/esm_msa/valid"
test_folder = "/uac/gds/hqcao23/hqcao/openfold/esm_msa/test"

split_files(source_folder, train_folder, valid_folder, test_folder)