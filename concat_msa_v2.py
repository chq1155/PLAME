from msa_scoring_single import MSASequenceScorer
import os
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
import shutil

def ensure_ends_with_newline(content):
    """确保内容以换行符结束"""
    return content if content.endswith('\n') else content + '\n'

def safe_combine_msa(target_file, source_file):
    """安全地将源MSA文件内容添加到目标文件"""
    with open(source_file, 'r') as infile:
        lines = []
        current_line = ""
        
        for line in infile:
            # 如果发现行中间有'>'，需要分割
            if '>' in line and not line.startswith('>'):
                parts = line.split('>')
                # 处理第一部分
                lines.append(ensure_ends_with_newline(parts[0]))
                # 处理剩余部分
                for part in parts[1:]:
                    if part.strip():
                        lines.append(ensure_ends_with_newline('>' + part))
            else:
                lines.append(ensure_ends_with_newline(line.rstrip('\n')))
    
    # 写入处理后的内容
    with open(target_file, 'a') as outfile:
        outfile.writelines(lines)

def process_msas(original_msa_dir, generated_folders, output_dir, scorer, mode="score"):
    """
    Process MSAs from multiple folders and combine them based on criteria
    
    Args:
        original_msa_dir: Directory containing original MSA files
        generated_folders: List of folders containing generated MSAs
        output_dir: Directory to save combined MSAs
        scorer: MSASequenceScorer instance
        mode: "score" or "direct" - whether to use scoring for filtering or directly combine
    """
    
    os.makedirs(output_dir, exist_ok=True)
    original_proteins = {f.stem: f for f in Path(original_msa_dir).glob("*.a3m")}
    
    for protein, orig_msa_path in original_proteins.items():
        print(f"Processing protein: {protein}")
        
        if mode == "score":
            # Scoring mode - original implementation
            orig_scores_df = scorer.evaluate_msa(str(orig_msa_path), "original")
            orig_count = len(orig_scores_df)
            
            generated_scores = []
            for folder in generated_folders:
                protein_folder = Path(folder) / protein
                if not protein_folder.exists():
                    continue
                    
                for file_path in protein_folder.rglob("*.a3m"):
                    scores_df = scorer.evaluate_msa(str(file_path), "generated")
                    scores_df['file_path'] = str(file_path)
                    generated_scores.append(scores_df)
            
            if not generated_scores:
                print(f"No generated MSAs found for {protein}")
                continue
                
            all_generated_df = pd.concat(generated_scores)
            
            orig_dist = orig_scores_df['final_score'].values
            gen_dist = all_generated_df['final_score'].values
            dist_similarity = -wasserstein_distance(orig_dist, gen_dist)
            
            all_generated_df['dist_similarity'] = dist_similarity
            all_generated_df = all_generated_df.sort_values(['dist_similarity', 'final_score'], 
                                                          ascending=[False, False])
            
            orig_mean = orig_scores_df['final_score'].mean()
            orig_std = orig_scores_df['final_score'].std()
            
            filtered_generated = all_generated_df[
                (all_generated_df['final_score'] >= orig_mean - 2*orig_std) &
                (all_generated_df['final_score'] <= orig_mean + 2*orig_std)
            ]
            
            if orig_count < 32:
                sequences_to_add = 32 - orig_count
                selected_files = filtered_generated.head(sequences_to_add)['file_path'].unique()
            else:
                selected_files = filtered_generated['file_path'].unique()
                
        else:
            # Direct mode - just count sequences and combine
            def count_sequences(a3m_file):
                count = 0
                with open(a3m_file, 'r') as f:
                    for line in f:
                        if line.startswith('>'):
                            count += 1
                return count
            
            orig_count = count_sequences(orig_msa_path)
            
            # Collect all generated MSA files
            generated_files = []
            for folder in generated_folders:
                protein_folder = Path(folder) / protein
                if protein_folder.exists():
                    generated_files.extend(list(protein_folder.rglob("*.a3m")))
            
            if not generated_files:
                print(f"No generated MSAs found for {protein}")
                continue
                
            # If original count < 32, select files until we reach target
            # Otherwise include all files
            if orig_count < 32:
                sequences_needed = 32 - orig_count
                selected_files = []
                current_count = 0
                
                for gen_file in generated_files:
                    file_count = count_sequences(gen_file)
                    if current_count + file_count <= sequences_needed:
                        selected_files.append(gen_file)
                        current_count += file_count
                    if current_count >= sequences_needed:
                        break
            else:
                selected_files = generated_files
        
        # Combine MSAs
        output_path = Path(output_dir) / f"{protein}_combined.a3m"
        shutil.copy2(orig_msa_path, output_path)
        
        # 使用安全的文件合并方法
        for gen_file in selected_files:
            safe_combine_msa(output_path, gen_file)
        
        print(f"Completed {protein}: Combined original MSA with {len(selected_files)} generated MSA files")

# Usage example:
scorer = MSASequenceScorer()
original_msa_dir = "./casp14/a3m"
generated_folders = ["./casp14/generation/artificial/A2T1R1.2", "./casp14/generation/artificial/A2T2R1.0"]
output_dir = "./casp14/combined"

# Using scoring mode
# process_msas(original_msa_dir, generated_folders, output_dir, scorer, mode="score")

# Or using direct mode
process_msas(original_msa_dir, generated_folders, output_dir, scorer, mode="direct")