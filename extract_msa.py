import os

def process_msa_entry(entry):
    # 分割名称和序列
    name, sequence_data = entry.split(':')
    
    # 创建输出文件
    output_file = f"./msagpt_gt/{name}.a3m"
    
    with open(output_file, 'w') as f:
        if '<M>' in sequence_data:
            # 有MSA的情况
            sequences = sequence_data.split('<M>')
            # 写入query sequence
            f.write(">query\n")
            f.write(sequences[0].strip() + "\n")
            
            # 写入MSA sequences
            for i, seq in enumerate(sequences[1:], 1):
                if seq.strip():  # 确保序列不是空的
                    f.write(f">msa_{i}\n")
                    f.write(seq.strip() + "\n")
        else:
            # 没有MSA的情况
            f.write(">query\n")
            f.write(sequence_data.strip() + "\n")

# 读取输入文件
with open('natural-msa-scarce-cases.txt', 'r') as file:
    entries = file.readlines()

# 处理每个条目
for entry in entries:
    if entry.strip():  # 跳过空行
        process_msa_entry(entry)