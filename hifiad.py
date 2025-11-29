import os
import glob
from typing import List, Tuple, Dict
import numpy as np

class HiFiAD_A3M:
    """
    Simplified HiFiAD for A3M file processing
    """
    
    def __init__(self):
        # Simplified BLOSUM62 matrix (most common amino acid pairs)
        self.blosum62 = {
            ('A','A'): 4, ('A','R'): -1, ('A','N'): -2, ('A','D'): -2, ('A','C'): 0,
            ('R','R'): 5, ('R','N'): 0, ('R','D'): -2, ('R','C'): -3, ('R','A'): -1,
            ('N','N'): 6, ('N','D'): 1, ('N','C'): -3, ('N','A'): -2, ('N','R'): 0,
            ('D','D'): 6, ('D','C'): -3, ('D','A'): -2, ('D','R'): -2, ('D','N'): 1,
            ('C','C'): 9, ('C','A'): 0, ('C','R'): -3, ('C','N'): -3, ('C','D'): -3,
            ('Q','Q'): 5, ('E','E'): 5, ('G','G'): 6, ('H','H'): 8, ('I','I'): 4,
            ('L','L'): 4, ('K','K'): 5, ('M','M'): 5, ('F','F'): 6, ('P','P'): 7,
            ('S','S'): 4, ('T','T'): 5, ('W','W'): 11, ('Y','Y'): 7, ('V','V'): 4
        }
        
    def _get_blosum_score(self, aa1: str, aa2: str) -> int:
        """Get BLOSUM62 score for amino acid pair"""
        if aa1 == '-' or aa2 == '-':
            return 0
        return self.blosum62.get((aa1.upper(), aa2.upper()), 
                                self.blosum62.get((aa2.upper(), aa1.upper()), -1))
    
    def calculate_scores(self, query_seq: str, target_seq: str) -> Tuple[float, float]:
        """Calculate BLOSUM score and recovery rate"""
        if len(query_seq) != len(target_seq):
            return 0.0, 0.0
            
        blosum_sum = 0
        matches = 0
        valid_pos = 0
        
        for i in range(len(query_seq)):
            if query_seq[i] != '-' and target_seq[i] != '-':
                blosum_sum += self._get_blosum_score(query_seq[i], target_seq[i])
                if query_seq[i].upper() == target_seq[i].upper():
                    matches += 1
                valid_pos += 1
        
        blosum_score = blosum_sum / valid_pos if valid_pos > 0 else 0
        recovery_rate = matches / valid_pos if valid_pos > 0 else 0
        
        return blosum_score, recovery_rate
    
    def select_sequences(self, query_seq: str, sequences: List[str], k: int = 16) -> List[int]:
        """
        Select sequence indices using HiFiAD strategy
        Returns indices of selected sequences
        """
        if len(sequences) <= k:
            return list(range(len(sequences)))
        
        # Calculate scores
        scores = []
        for i, seq in enumerate(sequences):
            blosum, recovery = self.calculate_scores(query_seq, seq)
            scores.append((i, blosum, recovery))
        
        # Sort by different criteria
        by_blosum = sorted(scores, key=lambda x: x[1], reverse=True)
        by_recovery = sorted(scores, key=lambda x: x[2], reverse=True)
        
        selected_indices = set()
        
        # Select top BLOSUM scores
        for i in range(min(k, len(by_blosum))):
            selected_indices.add(by_blosum[i][0])
        
        # Balance with recovery rates
        k_half = k // 2
        # High recovery rates
        for i in range(min(k_half, len(by_recovery))):
            selected_indices.add(by_recovery[i][0])
        
        # Low recovery rates for diversity
        for i in range(max(0, len(by_recovery) - k_half), len(by_recovery)):
            selected_indices.add(by_recovery[i][0])
        
        return list(selected_indices)
    
    def read_a3m(self, filepath: str) -> List[Tuple[str, str]]:
        """
        Read A3M file and return list of (description, sequence) tuples
        """
        sequences = []
        current_desc = ""
        current_seq = ""
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_desc and current_seq:
                            sequences.append((current_desc, current_seq))
                        current_desc = line[1:]  # Remove '>'
                        current_seq = ""
                    else:
                        current_seq += line
                
                # Add last sequence
                if current_desc and current_seq:
                    sequences.append((current_desc, current_seq))
                    
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return []
        
        return sequences
    
    def write_a3m(self, filepath: str, sequences: List[Tuple[str, str]]):
        """
        Write sequences to A3M file
        """
        try:
            with open(filepath, 'w') as f:
                for desc, seq in sequences:
                    f.write(f">{desc}\n{seq}\n")
        except Exception as e:
            print(f"Error writing {filepath}: {e}")
    
    def process_a3m_file(self, input_path: str, output_path: str, 
                        target_prefix: str = "XX", k: int = 16) -> bool:
        """
        Process single A3M file with HiFiAD filtering
        
        Args:
            input_path: Input A3M file path
            output_path: Output A3M file path  
            target_prefix: Prefix to identify sequences for filtering
            k: Number of sequences to keep after filtering
            
        Returns:
            True if processing succeeded, False otherwise
        """
        # Read A3M file
        sequences = self.read_a3m(input_path)
        if not sequences:
            print(f"No sequences found in {input_path}")
            return False
        
        # Get query sequence (first sequence)
        query_desc, query_seq = sequences[0]
        
        # Separate sequences by prefix
        target_sequences = []
        target_indices = []
        other_sequences = []
        
        for i, (desc, seq) in enumerate(sequences):
            if desc.startswith(target_prefix):
                target_sequences.append(seq)
                target_indices.append(i)
            else:
                other_sequences.append((desc, seq))
        
        # If no target sequences found, just copy the file
        if not target_sequences:
            print(f"No sequences with prefix '{target_prefix}' found in {input_path}")
            # Copy original file
            self.write_a3m(output_path, sequences)
            return True
        
        print(f"Found {len(target_sequences)} sequences with prefix '{target_prefix}'")
        
        # Apply HiFiAD selection
        selected_indices = self.select_sequences(query_seq, target_sequences, k)
        print(f"Selected {len(selected_indices)} sequences after HiFiAD filtering")
        
        # Build final sequence list
        final_sequences = []
        
        # Add other sequences (including query)
        final_sequences.extend(other_sequences)
        
        # Add selected target sequences
        for idx in selected_indices:
            original_idx = target_indices[idx]
            final_sequences.append(sequences[original_idx])
        
        # Write output
        self.write_a3m(output_path, final_sequences)
        print(f"Saved processed file to {output_path}")
        
        return True
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         target_prefix: str = "XX", k: int = 16):
        """
        Process all A3M files in a directory
        
        Args:
            input_dir: Input directory containing A3M files
            output_dir: Output directory for processed files
            target_prefix: Prefix to identify sequences for filtering
            k: Number of sequences to keep after filtering
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all A3M files
        a3m_files = glob.glob(os.path.join(input_dir, "*.a3m"))
        
        if not a3m_files:
            print(f"No A3M files found in {input_dir}")
            return
        
        print(f"Found {len(a3m_files)} A3M files to process")
        
        success_count = 0
        for input_path in a3m_files:
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, filename)
            
            print(f"\nProcessing: {filename}")
            
            if self.process_a3m_file(input_path, output_path, target_prefix, k):
                success_count += 1
            else:
                print(f"Failed to process {filename}")
        
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed: {success_count}/{len(a3m_files)} files")


def main():
    """
    Example usage of HiFiAD A3M processor
    """
    processor = HiFiAD_A3M()
    
    # Example 1: Process single file
    # processor.process_a3m_file(
    #     input_path="input.a3m",
    #     output_path="output.a3m", 
    #     target_prefix="XX",
    #     k=16
    # )
    
    # Example 2: Process entire directory
    processor.process_directory(
        input_dir="/uac/gds/hqcao23/hqcao/msa_projects/MSAGPT/outputs/all_dpo/a3m_files",
        output_dir="./masagpt_msa_aug_hifiad",
        target_prefix="GENERATED",
        k=12
    )

if __name__ == "__main__":
    main()


# # Additional utility functions for testing
# def create_test_a3m(filepath: str, num_xx_sequences: int = 20):
#     """
#     Create a test A3M file for demonstration
#     """
#     import random
#     amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
#     sequences = []
    
#     # Query sequence
#     query_seq = ''.join(random.choices(amino_acids, k=100))
#     sequences.append(("query_protein", query_seq))
    
#     # Regular sequences
#     for i in range(5):
#         # Create similar sequence
#         seq = list(query_seq)
#         for j in range(10):  # 10 mutations
#             pos = random.randint(0, len(seq)-1)
#             seq[pos] = random.choice(amino_acids)
#         sequences.append((f"regular_seq_{i}", ''.join(seq)))
    
#     # XX sequences (to be filtered)
#     for i in range(num_xx_sequences):
#         seq = list(query_seq)
#         for j in range(random.randint(5, 30)):  # Variable mutations
#             pos = random.randint(0, len(seq)-1)
#             seq[pos] = random.choice(amino_acids)
#         sequences.append((f"XX_generated_seq_{i}", ''.join(seq)))
    
#     # Write test file
#     with open(filepath, 'w') as f:
#         for desc, seq in sequences:
#             f.write(f">{desc}\n{seq}\n")
    
#     print(f"Created test A3M file: {filepath}")
#     print(f"Total sequences: {len(sequences)} (including {num_xx_sequences} XX sequences)")

# # Create test data if needed
# if __name__ == "__main__" and len(os.sys.argv) > 1 and os.sys.argv[1] == "test":
#     # Create test directory and files
#     os.makedirs("./input_a3m_files", exist_ok=True)
    
#     for i in range(3):
#         create_test_a3m(f"./input_a3m_files/test_{i}.a3m", num_xx_sequences=25)
    
#     print("Test files created. Now run: python script.py")