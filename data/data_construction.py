#!/usr/bin/env python3
"""Legacy data construction script for building training pickles from
pre-organized MSA directories.

For the recommended pipeline, use build_dataset.py instead.

This script processes a directory structure where each subfolder contains
an a3m/ directory with alignment files, computes ESM2-650M embeddings,
and saves training pickles.

Usage:
    python data/data_construction.py --input_dir ./uniclust30 --output_dir ./uniclust_emb --device cuda:0
"""

import argparse
import os
import pickle
import random

import torch
import esm
from tqdm import tqdm


def encode_seq_esm2(seq, model, alphabet, batch_converter, device):
    """Compute ESM2-650M per-residue embedding. Returns (seq_len, 1280) tensor."""
    data = [("protein", seq)]
    model.eval().to(device)
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_repr = results["representations"][33]
    tk_len = token_repr.shape[1]
    emb = token_repr[:, 1 : tk_len - 1, :].reshape(-1, 1280).detach().cpu()
    return emb


def process_protein_data(input_dir, output_dir, model, alphabet, batch_converter, device="cuda"):
    """Process protein folders with a3m/ subdirectories into training pickles."""
    os.makedirs(output_dir, exist_ok=True)
    folders = [f.name for f in os.scandir(input_dir)]

    for folder in tqdm(folders, desc="Processing"):
        msa_path = os.path.join(input_dir, folder, "a3m")
        if not os.path.exists(msa_path):
            continue

        output_data = []
        query_seq = None

        for name in os.listdir(msa_path):
            if not name.endswith(".a3m"):
                continue
            path = os.path.join(msa_path, name)
            with open(path, "r") as f:
                lines = f.readlines()

            if len(lines) < 2:
                continue

            query_seq = lines[1].strip().upper()
            seq_length = len(query_seq)
            output_data.append(query_seq)

            for i in range(2, len(lines), 2):
                if i + 1 < len(lines):
                    seq = lines[i + 1].strip().upper()
                    if len(seq) == seq_length:
                        output_data.append(seq)

        if query_seq is None or len(output_data) < 2:
            continue

        emb = encode_seq_esm2(query_seq, model, alphabet, batch_converter, device)

        record = {
            "name": folder,
            "seq": query_seq,
            "emb": emb,
            "msa": output_data if len(output_data) <= 64 else random.sample(output_data, 64),
        }

        output_path = os.path.join(output_dir, f"{folder}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(record, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training pickles from MSA directories")
    parser.add_argument("--input_dir", required=True, help="Directory with protein folders")
    parser.add_argument("--output_dir", required=True, help="Output directory for pickle files")
    parser.add_argument("--device", default="cuda", help="Device for ESM2")
    args = parser.parse_args()

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    process_protein_data(args.input_dir, args.output_dir, model, alphabet, batch_converter, args.device)
    print("Processing complete.")
