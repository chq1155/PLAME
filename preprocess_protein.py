import os
import pickle
import random
import esm
import torch
from tqdm import tqdm

def process_protein_data(train_path, num_msa): # TODO:小写需要全部换成大写
    folders = [f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))]

    total_msa_data = [] # get full data

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    for folder in tqdm(folders):
        msa_path = os.path.join(train_path, folder, 'a3m')
        if os.path.exists(msa_path):
            output_data = []
            files_to_process = ['bfd_uniclust_hits.a3m', 'mgnify_hits.a3m', 'uniref90_hits.a3m']
            # files_to_process = ['uniref90_hits.a3m']
            
            for name in files_to_process:
                path = os.path.join(msa_path, name)
                if os.path.exists(path):
                    with open(path, 'r') as file:
                        lines = file.readlines()
                        if len(lines) < 2:
                            continue
                        
                        query_label = lines[0].strip()
                        query_seq = lines[1].strip()
                        seq_length = len(query_seq)
                        # output_data.append(f"{query_label}\n{query_seq}\n")
                        output_data.append(query_seq.upper())
                        
                        for i in range(2, len(lines), 2):
                            if i + 1 < len(lines):
                                label = lines[i].strip()
                                seq = lines[i + 1].strip()
                                if seq_length - 6 <= len(seq) <= seq_length + 6:
                                    # output_data.append(f"{label}\n{seq}\n")
                                    output_data.append(seq.upper())
            
            # output_path = os.path.join(train_path, f"{folder}.fasta")
            # with open(output_path, 'w') as output_file: 
            #     output_file.writelines(output_data)
        
        select_msa = random.sample(output_data[1:], num_msa)
        
        # get ESM Embedding
        data = [("data", query_seq)]
        _, _, batch_tokens = batch_converter(data)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)

        token_representations = results["representations"][33]
        tk_len = token_representations.shape[1]
        esm_emb = token_representations[:, 1:tk_len-1, :] # 1, 1280, seq_length
        
        
        total_msa_data.append({'name':folder, 'msa':select_msa, 'emb': esm_emb, 'seq':query_seq})
    
    return total_msa_data
                
msa = process_protein_data('../pdb', 128) # change file path here

with open("protein.pkl", "wb") as f:
    pickle.dump(msa, f)

print('process success!')