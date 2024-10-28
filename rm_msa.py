import os

def clean_folders(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file in filenames:
            if file != 'merged.a3m':  # 正确的文件名
                file_path = os.path.join(dirpath, file)
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除 {file_path} 时出错: {e}")

# 使用示例
root_folder = "/uac/gds/hqcao23/hqcao/gx/Protein_MSA_Fold/casp14/nogap1/artificial/A1T8R1.0Temp0.3"
clean_folders(root_folder)