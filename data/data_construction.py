import os
import torch
import esm
import pickle
import random
from tqdm import tqdm

def encode_seq_esm2(seq, model, alphabet, batch_converter, device):
    seq = [seq]
    model.eval().to(device)
    data = [(f"protein_{i}", seq[i]) for i in range(len(seq))]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    tk_len = token_representations.shape[1]
    esm_emb = token_representations[:,1:tk_len-1,:].reshape(-1, 1280).detach().cpu()

    return esm_emb

def process_protein_data(train_path, model, alphabet, batch_converter, device="cuda"):
    
    count = 0
    # folders = [f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))]
    folders = [f.name for f in os.scandir(train_path)]

    total_msa_data = [] # get full data
    for folder in tqdm(folders, desc='Processing folders'):
        # print(folder)
        # break
        msa_path = os.path.join(train_path, folder, 'a3m')
        if os.path.exists(msa_path):
            output_data = []
            # files_to_process = ['bfd_uniclust_hits.a3m', 'mgnify_hits.a3m', 'uniref90_hits.a3m'] # for pdb-related data
            files_to_process = ['uniclust30.a3m'] # for uniclust30 data
            dict = {
                'name': folder,
            }
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
                        output_data.append(query_seq.upper())
                        
                        for i in range(2, len(lines), 2):
                            if i + 1 < len(lines):
                                label = lines[i].strip()
                                seq = lines[i + 1].strip()
                                # if seq_length - 6 <= len(seq) <= seq_length + 6:
                                if seq_length == len(seq): 
                                    output_data.append(seq.upper())
                        
            esm_emb = encode_seq_esm2(query_seq, model, alphabet, batch_converter, device)
            
            if len(output_data) <= 64:
                dict['msa'] = output_data
                count += 1
            else:
                dict['msa'] = random.sample(output_data, 64)
            dict['emb'] = esm_emb
            dict['seq'] = query_seq
 
            output_path = "./uniclust_emb/" + folder + ".pkl"

            with open(output_path, 'wb') as file:
                pickle.dump(dict, file)

    print(f"total special count: {count}")

if __name__ == "__main__":
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    process_protein_data('./uniclust30', model, alphabet, batch_converter, "cuda") # change file path here
    print('process success!')
