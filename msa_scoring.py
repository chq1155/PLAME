import os
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from collections import Counter
import numpy as np
from scipy.stats import entropy
import pandas as pd

class MSAQualityChecker:
    def __init__(self):
        pass
        
    def read_a3m(self, file_path):
        """
        读取a3m格式文件并转换为标准MSA格式
        移除小写字母（插入），保留大写字母和gap
        """
        with open(file_path, 'r') as f:
            sequences = []
            current_seq = ''
            current_id = ''
            
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(SeqRecord(Seq(current_seq), id=current_id))
                    current_id = line[1:]
                    current_seq = ''
                else:
                    # 移除小写字母，保留大写字母和gap
                    current_seq += ''.join(c for c in line if c.isupper() or c == '-')
            
            if current_seq:
                sequences.append(SeqRecord(Seq(current_seq), id=current_id))
                
        return MultipleSeqAlignment(sequences)

    def calculate_conservation_score(self, alignment):
        """计算保守性得分"""
        length = alignment.get_alignment_length()
        num_sequences = len(alignment)
        conservation_per_pos = []
        
        for i in range(length):
            column = alignment[:, i]
            counts = Counter(column)
            freq = [count/num_sequences for count in counts.values()]
            conservation = 1 - entropy(freq, base=20)
            conservation_per_pos.append(conservation)
            
        return np.mean(conservation_per_pos)

    def calculate_gap_score(self, alignment):
        """计算gap得分"""
        length = alignment.get_alignment_length()
        num_sequences = len(alignment)
        gap_per_pos = []
        
        for i in range(length):
            column = alignment[:, i]
            gap_ratio = column.count('-') / num_sequences
            gap_per_pos.append(gap_ratio)
            
        return 1 - np.mean(gap_per_pos)

    def calculate_mismatch_score(self, alignment):
        """计算错配得分"""
        reference = alignment[0]
        mismatch_scores = []
        
        for record in alignment[1:]:
            matches = sum(1 for a, b in zip(reference, record) 
                         if a != '-' and b != '-' and a != b)
            length = sum(1 for a, b in zip(reference, record) 
                        if a != '-' and b != '-')
            mismatch_scores.append(matches/length if length > 0 else 0)
            
        return 1 - np.mean(mismatch_scores)

    def calculate_diversity_score(self, alignment):
        """计算序列多样性得分"""
        num_sequences = len(alignment)
        length = alignment.get_alignment_length()
        distances = []
        
        for i in range(num_sequences):
            for j in range(i+1, num_sequences):
                seq1 = str(alignment[i].seq)
                seq2 = str(alignment[j].seq)
                identity = sum(a == b for a, b in zip(seq1, seq2)) / length
                distances.append(identity)
                
        mean_identity = np.mean(distances)
        if 0.3 <= mean_identity <= 0.9:
            diversity_score = 1.0
        else:
            diversity_score = 1 - min(abs(mean_identity-0.6), 1.0)
            
        return diversity_score

    def calculate_coverage_score(self, alignment):
        """计算序列覆盖度得分"""
        query = str(alignment[0].seq)
        query_length = len(query.replace('-', ''))
        
        coverage_scores = []
        for record in alignment[1:]:
            seq = str(record.seq)
            aligned_positions = sum(1 for a, b in zip(query, seq) 
                                 if a != '-' and b != '-')
            coverage = aligned_positions / query_length
            coverage_scores.append(coverage)
            
        return np.mean(coverage_scores)

    def evaluate_msa(self, a3m_file):
        """评估单个MSA文件"""
        try:
            alignment = self.read_a3m(a3m_file)
            
            if len(alignment) < 2:
                return {
                    'file': a3m_file,
                    'error': 'Too few sequences',
                    'scores': None,
                    'final_score': 0,
                    'quality_label': 0
                }

            scores = {
                'conservation': self.calculate_conservation_score(alignment),
                'gap': self.calculate_gap_score(alignment),
                'mismatch': self.calculate_mismatch_score(alignment),
                'diversity': self.calculate_diversity_score(alignment),
                'coverage': self.calculate_coverage_score(alignment)
            }

            weights = {
                'conservation': 0.25,
                'gap': 0.2,
                'mismatch': 0.25,
                'diversity': 0.15,
                'coverage': 0.15
            }

            final_score = sum(score * weights[metric] 
                            for metric, score in scores.items())

            return {
                'file': a3m_file,
                'error': None,
                'scores': scores,
                'final_score': final_score,
                'quality_label': 1 if final_score > 0.6 else 0
            }
            
        except Exception as e:
            return {
                'file': a3m_file,
                'error': str(e),
                'scores': None,
                'final_score': 0,
                'quality_label': 0
            }

    def batch_evaluate(self, directory, output_file='msa_quality_scores.csv'):
        """批量评估目录下所有的a3m文件"""
        results = []
        
        for file in os.listdir(directory):
            if file.endswith('.a3m'):
                file_path = os.path.join(directory, file)
                result = self.evaluate_msa(file_path)
                results.append(result)

        # 转换为DataFrame并保存
        df_rows = []
        for result in results:
            row = {
                'file': result['file'],
                'final_score': result['final_score'],
                'quality_label': result['quality_label'],
                'error': result['error']
            }
            
            if result['scores']:
                row.update(result['scores'])
            
            df_rows.append(row)

        df = pd.DataFrame(df_rows)
        df.to_csv(output_file, index=False)
        return df

# 单个文件评估
file_path = "/uac/gds/hqcao23/hqcao/gx/Protein_MSA_Fold/casp14/generation/artificial/A1T5R1.2/T1024/generation_4.a3m"
checker = MSAQualityChecker()
result = checker.evaluate_msa(file_path)
print(f"File: {result['file']}")
print(f"Scores: {result['scores']}")
print(f"Final score: {result['final_score']:.3f}")
print(f"Quality label: {result['quality_label']}")

# 批量评估
results_df = checker.batch_evaluate("/uac/gds/hqcao23/hqcao/gx/Protein_MSA_Fold/casp14/generation/artificial/A1T5R1.2/T1024/")
print(results_df.head())
