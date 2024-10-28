import os
import glob

def merge_a3m_files(root_folder):
    for subdir, dirs, files in os.walk(root_folder):
        a3m_files = glob.glob(os.path.join(subdir, '*.a3m'))
        if a3m_files:
            unique_sequences = {}  # 使用字典来存储唯一序列及其标识符
            output_file = os.path.join(subdir, 'merged.a3m')
            
            with open(output_file, 'w') as outfile:
                for a3m_file in a3m_files:
                    try:
                        with open(a3m_file, 'r') as infile:
                            lines = infile.readlines()
                            for i in range(0, len(lines), 2):
                                if i+1 < len(lines):
                                    identifier = lines[i].strip()
                                    sequence = lines[i+1].strip()
                                    if sequence and sequence not in unique_sequences:
                                        # 确保标识符以 ">" 开头
                                        if not identifier.startswith('>'):
                                            identifier = '>' + identifier
                                        unique_sequences[sequence] = identifier
                    except Exception as e:
                        print(f"处理文件 {a3m_file} 时出错: {e}")

                # 写入合并后的序列
                for sequence, identifier in unique_sequences.items():
                    outfile.write(f"{identifier}\n{sequence}\n")
            
            print(f"合并完成: {output_file}")

# 使用示例
root_folder = '/uac/gds/hqcao23/hqcao/gx/Protein_MSA_Fold/casp14/nogap1/artificial/A1T8R1.0Temp0.3'
merge_a3m_files(root_folder)
