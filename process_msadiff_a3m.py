import os
import shutil

def rename_and_move_a3m_files(source_dir, target_dir):
    """
    遍历 source_dir 中的所有子文件夹，找到其中的 non_pairing.a3m 文件，
    并将其重命名为子文件夹的名称，再移动到 target_dir 目录中。

    :param source_dir: 源文件夹，包含多个子文件夹
    :param target_dir: 目标文件夹，用于存放重命名后的 .a3m 文件
    """
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)

    # 遍历 source_dir 目录下的所有子文件夹
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        
        # 确保是一个目录
        if os.path.isdir(subdir_path):
            a3m_file_path = os.path.join(subdir_path, "non_pairing.a3m")
            
            # 检查 non_pairing.a3m 是否存在
            if os.path.exists(a3m_file_path):
                new_filename = f"{subdir}.a3m"
                new_file_path = os.path.join(target_dir, new_filename)
                
                # 复制并重命名文件
                shutil.copy(a3m_file_path, new_file_path)
                print(f"已处理: {a3m_file_path} -> {new_file_path}")
            else:
                print(f"警告: {a3m_file_path} 不存在，跳过该文件夹。")

# 示例用法
source_directory = "./scarce_epoch_16"  # 请替换为你的实际路径
target_directory = "./ani_tes"  # 请替换为你的实际路径

rename_and_move_a3m_files(source_directory, target_directory)