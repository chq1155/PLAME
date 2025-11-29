import os

def parse_and_save_msa(input_file, output_folder):
    """
    解析输入的 txt 文件并提取 MSA 信息，将其保存到指定文件夹中。
    即使没有 MSA 信息，也会保存 query sequence。
    
    Args:
        input_file (str): 包含蛋白质序列和 MSA 信息的 txt 文件路径。
        output_folder (str): 保存 .a3m 文件的输出文件夹路径。
    """
    # 如果输出文件夹不存在，则创建
    os.makedirs(output_folder, exist_ok=True)
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:  # 跳过空行
            continue
        
        # 按照描述的格式分割蛋白质 ID、query sequence 和 MSA 信息
        try:
            # 以冒号分隔，提取蛋白质 ID 和其余内容
            protein_id, msa_data = line.split(':', 1)
            # 按 <M> 分割，获取 query sequence 和其他 MSA 信息
            sequences = msa_data.split('<M>')
        except ValueError:
            print(f"行格式错误，跳过：{line}")
            continue
        
        query_sequence = sequences[0].strip()  # query sequence
        msa_sequences = [seq.strip() for seq in sequences[1:] if seq.strip()]  # MSA 信息
        
        # 创建 .a3m 文件
        output_file = os.path.join(output_folder, f"{protein_id}.a3m")
        with open(output_file, 'w') as out_f:
            # 写入 query sequence
            out_f.write(f">{protein_id}\n{query_sequence}\n")
            
            # 写入 MSA 信息（如果有）
            for msa_seq in msa_sequences:
                out_f.write(f">{protein_id}_msa\n{msa_seq}\n")

        # 打印保存信息
        if msa_sequences:
            print(f"蛋白质 {protein_id} 的 MSA 信息已保存到 {output_file}")
        else:
            print(f"蛋白质 {protein_id} 没有 MSA 信息，仅保存了 query sequence 到 {output_file}")

# 示例用法
input_txt_file = "natural-msa-scarce-cases.txt"  # 替换为你的输入文件路径
output_directory = "af2_msa"  # 替换为你的输出文件夹路径
parse_and_save_msa(input_txt_file, output_directory)