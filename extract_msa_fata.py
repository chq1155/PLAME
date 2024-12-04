def process_proteins_to_fasta(input_file, output_file):
    # 打开输出的fasta文件
    with open(output_file, 'w') as out_f:
        # 读取输入文件
        with open(input_file, 'r') as in_f:
            for line in in_f:
                if line.strip():  # 跳过空行
                    # 分割蛋白质名称和序列数据
                    name, sequence_data = line.strip().split(':')
                    
                    # 提取query sequence
                    if '<M>' in sequence_data:
                        query_seq = sequence_data.split('<M>')[0].strip()
                    else:
                        query_seq = sequence_data.strip()
                    
                    # 写入fasta格式
                    out_f.write(f">{name}\n")
                    out_f.write(f"{query_seq}\n")

# 执行函数
process_proteins_to_fasta('natural-msa-scarce-cases.txt', 'query_sequences.fasta')