import h5py
import pickle
import os
from tqdm import tqdm

def convert_pkl_to_hdf5(pkl_dir, output_hdf5):
    with h5py.File(output_hdf5, 'w') as hf:
        for i, filename in enumerate(tqdm(os.listdir(pkl_dir))):
            if filename.endswith('.pkl'):
                with open(os.path.join(pkl_dir, filename), 'rb') as f:
                    data = pickle.load(f)
                group = hf.create_group(f'sample_{i}')
                group.create_dataset('msa', data=data['msa'])
                group.create_dataset('seq', data=data['seq'])
                # 添加其他需要的字段

# 使用示例
convert_pkl_to_hdf5('/uac/gds/hqcao23/hqcao/openfold/esm_msa/train', 'output.hdf5')