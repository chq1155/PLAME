import os
import glob
from tqdm import tqdm
import shutil

def collect_and_move_files(a3m_folder, pkl_folder, destination_folder):
    # 获取所有a3m文件名
    print("Collecting a3m files...")
    a3m_files = glob.glob(os.path.join(a3m_folder, "*.a3m"))
    a3m_basenames = [os.path.splitext(os.path.basename(f))[0] for f in a3m_files]
    
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # 首先获取pkl文件夹中所有文件的集合，加快搜索速度
    print("Indexing pkl files...")
    pkl_files = set(os.path.splitext(f)[0] for f in os.listdir(pkl_folder) if f.endswith('.pkl'))
    
    # 搜索对应的pkl文件并移动
    moved_count = 0
    print("Moving files...")
    for name in tqdm(a3m_basenames, desc="Processing"):
        print(name)
        if name in pkl_files:
            pkl_path = os.path.join(pkl_folder, name + ".pkl")
            print(pkl_path)
            dest_path = os.path.join(destination_folder, name + ".pkl")
            # 使用shutil.move代替os.rename，可以跨文件系统移动
            shutil.move(pkl_path, dest_path)
            moved_count += 1
    
    print(f"\nTotal files moved: {moved_count}")
    print(f"Total a3m files found: {len(a3m_basenames)}")

# 使用示例
a3m_folder = "./msagpt_gt"
pkl_folder = "/uac/gds/hqcao23/hqcao/openfold/esm_msa_short"
destination_folder = "/uac/gds/hqcao23/hqcao/openfold/esm_msa_testscarse"
collect_and_move_files(a3m_folder, pkl_folder, destination_folder)