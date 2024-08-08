import json
import shutil
import os

def copy_files_from_json(json_file, destination_folder):
    # 读取JSON文件
    with open(json_file, 'r') as f:
        file_list = json.load(f)
    
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # 复制文件
    for file_path in file_list:
        if os.path.exists(file_path):
            # 获取文件名
            file_name = os.path.basename(file_path)
            # 构建目标路径
            destination_path = os.path.join(destination_folder, file_name)
            # 复制文件
            shutil.copy2(file_path, destination_path)
            print(f"已复制: {file_path} -> {destination_path}")
        else:
            print(f"文件不存在: {file_path}")

# 使用示例
json_file_path = 'test_list.json'
destination_folder = '/uac/gds/hqcao23/hqcao/openfold/esm_msa/test'

copy_files_from_json(json_file_path, destination_folder)