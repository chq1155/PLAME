import pickle

import torch
import esm

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results


seq = 1 # input sequence

data = [("data", seq)]

batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)

token_representations = results["representations"][33]
tk_len = token_representations.shape[1]
esm_emb = token_representations[:,1:tk_len-1,:]
