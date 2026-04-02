#!/usr/bin/env python3
"""Convert A3M files to pickle format for PLAME inference.

Usage:
    # Single A3M file
    python data/prepare_a3m.py --input data/example.a3m --output_dir data/inference_input

    # Directory of A3M files
    python data/prepare_a3m.py --input data/a3m_dir/ --output_dir data/inference_input

    # Specify GPU device
    python data/prepare_a3m.py --input data/a3m_dir/ --output_dir data/inference_input --device cuda:0

Each output pickle contains:
    - name (str): protein identifier (filename stem)
    - seq  (str): query sequence (first sequence in A3M, uppercase, insertions removed)
    - emb  (Tensor): ESM2-650M per-residue embedding, shape (seq_len, 1280)
"""

import argparse
import os
import pickle
import re

import torch
import esm


def parse_a3m(path: str):
    """Parse an A3M file and return (name, query_seq, aligned_seqs).

    The query sequence is the first entry. Lowercase characters (insertions
    relative to the query) are stripped from all sequences so every sequence
    has the same length as the query.
    """
    headers, sequences = [], []
    with open(path) as f:
        header, buf = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    headers.append(header)
                    sequences.append("".join(buf))
                header = line[1:]
                buf = []
            else:
                buf.append(line)
        if header is not None:
            headers.append(header)
            sequences.append("".join(buf))

    if not sequences:
        raise ValueError(f"No sequences found in {path}")

    # Remove lowercase insertion characters to get aligned-length sequences
    query_seq = re.sub(r"[a-z]", "", sequences[0])
    aligned_seqs = [re.sub(r"[a-z]", "", s) for s in sequences]

    name = os.path.splitext(os.path.basename(path))[0]
    return name, query_seq, aligned_seqs


def encode_esm2(seq: str, model, alphabet, batch_converter, device: str):
    """Compute ESM2-650M per-residue embedding for a single sequence.

    Returns tensor of shape (seq_len, 1280).
    """
    data = [("query", seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    # Strip BOS and EOS tokens → (seq_len, 1280)
    token_repr = results["representations"][33]
    emb = token_repr[:, 1 : len(seq) + 1, :].reshape(-1, 1280).detach().cpu()
    return emb


def process_a3m(path: str, output_dir: str, model, alphabet, batch_converter, device: str):
    """Process one A3M file → one pickle."""
    name, query_seq, _ = parse_a3m(path)

    emb = encode_esm2(query_seq, model, alphabet, batch_converter, device)

    record = {
        "name": name,
        "seq": query_seq,
        "emb": emb,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(record, f)

    print(f"[OK] {name}  seq_len={len(query_seq)}  emb={tuple(emb.shape)}  → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert A3M files to PLAME inference pickles")
    parser.add_argument("--input", required=True, help="A3M file or directory of A3M files")
    parser.add_argument("--output_dir", required=True, help="Output directory for pickle files")
    parser.add_argument("--device", default="cuda:0", help="Device for ESM2 (default: cuda:0)")
    args = parser.parse_args()

    # Load ESM2-650M
    print("Loading ESM2-650M model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval().to(args.device)
    print("ESM2-650M loaded.")

    # Collect A3M files
    if os.path.isfile(args.input):
        a3m_files = [args.input]
    elif os.path.isdir(args.input):
        a3m_files = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith(".a3m")
        )
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")

    if not a3m_files:
        raise FileNotFoundError(f"No .a3m files found in {args.input}")

    print(f"Processing {len(a3m_files)} A3M file(s) ...")
    for path in a3m_files:
        process_a3m(path, args.output_dir, model, alphabet, batch_converter, args.device)

    print(f"Done. Pickles written to {args.output_dir}")


if __name__ == "__main__":
    main()
