import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import Counter
from scipy.stats import entropy
import pandas as pd
import os

class MSASequenceScorer:
    def __init__(self):
        self.amino_acids = set('ARNDCQEGHILKMFPSTWYV-')
        
    def read_a3m(self, file_path):
        sequences = []
        with open(file_path, 'r') as f:
            current_seq = ''
            current_id = ''
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        processed_seq = ''.join(c for c in current_seq if c.isupper() or c == '-')
                        sequences.append({
                            'id': current_id,
                            'sequence': processed_seq
                        })
                    current_id = line[1:]
                    current_seq = ''
                else:
                    current_seq += line
            
            if current_seq:
                processed_seq = ''.join(c for c in current_seq if c.isupper() or c == '-')
                sequences.append({
                    'id': current_id,
                    'sequence': processed_seq
                })
        return sequences

    def calculate_sequence_scores(self, query_seq, sequence, all_sequences):
        scores = {}
        
        identity = sum(a == b for a, b in zip(query_seq, sequence)) / len(query_seq)
        scores['query_similarity'] = identity
        
        gap_ratio = sequence.count('-') / len(sequence)
        scores['gap_score'] = 1 - gap_ratio
        
        aligned_positions = sum(1 for a, b in zip(query_seq, sequence) 
                              if a != '-' and b != '-')
        query_length = len(query_seq.replace('-', ''))
        scores['coverage_score'] = aligned_positions / query_length
        
        conserved_positions = self.get_conserved_positions(all_sequences)
        conserved_matches = sum(1 for i, aa in enumerate(sequence)
                              if i in conserved_positions and aa == query_seq[i])
        scores['conservation_score'] = (conserved_matches / len(conserved_positions)) if conserved_positions else 0
        
        valid_aa_ratio = sum(1 for aa in sequence if aa in self.amino_acids) / len(sequence)
        scores['sequence_validity'] = valid_aa_ratio
        
        return scores
    
    def get_conserved_positions(self, sequences, conservation_threshold=0.8):
        length = len(sequences[0]['sequence'])
        conserved_positions = set()
        
        for i in range(length):
            column = [seq['sequence'][i] for seq in sequences]
            counts = Counter(column)
            most_common = counts.most_common(1)[0]
            if most_common[1] / len(sequences) >= conservation_threshold:
                conserved_positions.add(i)
                
        return conserved_positions
    
    def evaluate_msa(self, a3m_file, msa_type):
        """
        增加msa_type参数来标识MSA来源
        """
        sequences = self.read_a3m(a3m_file)
        if not sequences:
            return pd.DataFrame()
        
        query_seq = sequences[0]['sequence']
        results = []
        
        for seq_data in sequences:
            scores = self.calculate_sequence_scores(
                query_seq, 
                seq_data['sequence'],
                sequences
            )
            
            weights = {
                'query_similarity': 0.25,
                'gap_score': 0.2,
                'coverage_score': 0.25,
                'conservation_score': 0.2,
                'sequence_validity': 0.1
            }
            
            final_score = sum(score * weights[metric] 
                            for metric, score in scores.items())
            
            result = {
                'sequence_id': seq_data['id'],
                'msa_type': msa_type,
                'final_score': final_score,
                'sequence_length': len(seq_data['sequence']),
                **scores
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        df = df.sort_values('final_score', ascending=False)
        return df

def filter_sequences(df, threshold=0.6):
    return df[df['final_score'] >= threshold]

def compare_msa_versions(seq_id):
    """
    比较两个版本MSA的函数
    """
    # 定义文件路径模式
    original_path = f"/uac/gds/hqcao23/hqcao/gx/Protein_MSA_Fold/casp14/a3m/{seq_id}.a3m"
    generated_path = f"/uac/gds/hqcao23/hqcao/gx/Protein_MSA_Fold/casp14/generation/artificial/A2T2R1.0/{seq_id}/generation_0.a3m"
    
    scorer = MSASequenceScorer()
    results = []
    
    # 检查文件是否存在
    if not os.path.exists(original_path):
        print(f"Warning: Original MSA file not found for {seq_id}")
        return None
    if not os.path.exists(generated_path):
        print(f"Warning: Generated MSA file not found for {seq_id}")
        return None

    # 评估原始MSA
    original_results = scorer.evaluate_msa(original_path, "original")
    results.append(original_results)

    # 评估生成的MSA
    generated_results = scorer.evaluate_msa(generated_path, "generated")
    results.append(generated_results)

    # 合并结果
    combined_results = pd.concat(results, ignore_index=True)
    
    # 添加sequence_id列
    combined_results['target_id'] = seq_id
    
    return combined_results

def batch_process_sequences(seq_id_list, output_file="msa_comparison_results.csv"):
    """
    批量处理多个序列ID
    """
    all_results = []
    
    for seq_id in seq_id_list:
        print(f"Processing {seq_id}...")
        results = compare_msa_versions(seq_id)
        if results is not None:
            all_results.append(results)
    
    # 合并所有结果
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        
        # 重新组织列的顺序
        columns_order = ['target_id', 'msa_type', 'sequence_id', 'final_score', 
                        'sequence_length', 'query_similarity', 'gap_score', 
                        'coverage_score', 'conservation_score', 'sequence_validity']
        final_results = final_results[columns_order]
        
        # 保存结果
        final_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        return final_results
    else:
        print("No results to save")
        return None

if __name__ == "__main__":
    # 示例序列ID列表
    # seq_ids = ["T1024", "T1025", "T1026", "T1027", "T1028", "T1029"]  # 在这里添加更多序列ID
    seq_ids = ["T1030", "T1031", "T1033", "T1035", "T1037", "T1039", "T1040", "T1041", "T1042", "T1043", "T1044", "T1052", "T1064", "T1070", "T1074", "T1082", "T1087", "T1096", "T1099"]  # 在这里添加更多序列ID
    
    # 批量处理所有序列
    results = batch_process_sequences(seq_ids)
    
    # 打印结果摘要
    if results is not None:
        print("\nResults summary:")
        print(results.groupby(['target_id', 'msa_type'])['final_score'].agg(['mean', 'min', 'max']))