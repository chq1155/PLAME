import os
from pathlib import Path

def process_a3m_files(folder_path, output_folder=None):
    """
    遍历文件夹下的所有 a3m 文件，标准化序列长度
    
    Args:
        folder_path: 包含 a3m 文件的文件夹路径
        output_folder: 输出文件夹路径，如果为 None 则覆盖原文件
    """
    folder_path = Path(folder_path)
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
    
    # 遍历所有 a3m 文件
    for a3m_file in folder_path.glob("*.a3m"):
        print(f"处理文件: {a3m_file}")
        
        # 读取文件内容
        with open(a3m_file, 'r') as f:
            lines = f.readlines()
        
        # 解析序列
        sequences = []
        current_seq = ""
        current_header = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    sequences.append((current_header, current_seq))
                current_header = line
                current_seq = ""
            else:
                current_seq += line
        
        # 添加最后一个序列
        if current_header:
            sequences.append((current_header, current_seq))
        
        if not sequences:
            continue
            
        # 获取第一条序列的长度作为标准长度
        reference_length = len(sequences[0][1])
        print(f"参考序列长度: {reference_length}")
        
        # 处理序列长度标准化
        processed_sequences = [sequences[0]]  # 保持第一条序列不变
        
        for header, seq in sequences[1:]:
            if len(seq) < reference_length:
                # 用 gap (-) 补全
                padded_seq = seq + '-' * (reference_length - len(seq))
                processed_sequences.append((header, padded_seq))
            elif len(seq) > reference_length:
                # 截断到参考长度
                truncated_seq = seq[:reference_length]
                processed_sequences.append((header, truncated_seq))
            else:
                # 长度相等，保持不变
                processed_sequences.append((header, seq))
        
        # 写入处理后的文件
        output_file = output_folder / a3m_file.name if output_folder else a3m_file
        
        with open(output_file, 'w') as f:
            for header, seq in processed_sequences:
                f.write(f"{header}\n{seq}\n")
        
        print(f"处理完成，输出到: {output_file}")

# 使用示例
if __name__ == "__main__":
    # 处理当前目录下的 a3m 文件，覆盖原文件
    process_a3m_files("./evodiff_msa_aug_hifiad")
    
    # 或者指定输入和输出文件夹
    # process_a3m_files("./input_folder", "./output_folder")