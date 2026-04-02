#!/usr/bin/env python3
"""Build PLAME training dataset from OpenProteinSet MSA files.

This script processes A3M alignment files from OpenProteinSet into pickle files
suitable for PLAME training. For each protein, it:
  1. Parses the A3M file to extract aligned sequences
  2. Computes ESM2-650M per-residue embeddings for the query sequence
  3. Saves a pickle with keys: name, seq, emb, msa

Usage:
    # Process PDB subset
    python data/build_dataset.py \
        --input_dir data/openproteinset/pdb \
        --output_dir data/esm_msa/train \
        --device cuda:0

    # Process with filters matching the paper (min 64 MSA sequences, max 512 length)
    python data/build_dataset.py \
        --input_dir data/openproteinset/pdb \
        --output_dir data/esm_msa/train \
        --min_msa_depth 64 \
        --max_seq_length 512 \
        --device cuda:0

    # Resume interrupted processing
    python data/build_dataset.py \
        --input_dir data/openproteinset/pdb \
        --output_dir data/esm_msa/train \
        --device cuda:0 --resume
"""

import argparse
import os
import pickle
import re
import sys

import torch
from tqdm import tqdm


def parse_a3m(path):
    """Parse an A3M file and return (query_seq, aligned_sequences).

    Lowercase characters (insertions relative to the query) are removed so
    all sequences share the same length as the query.
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
        return None, []

    # Remove lowercase insertion characters for uniform length
    query_seq = re.sub(r"[a-z]", "", sequences[0]).upper()
    query_len = len(query_seq)

    aligned = []
    for seq in sequences:
        clean = re.sub(r"[a-z]", "", seq).upper()
        if len(clean) == query_len:
            aligned.append(clean)

    return query_seq, aligned


def find_a3m_files(input_dir):
    """Recursively find all .a3m files under input_dir."""
    a3m_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".a3m"):
                a3m_files.append(os.path.join(root, f))
    return sorted(a3m_files)


def find_protein_folders(input_dir):
    """Find OpenProteinSet-style protein folders with a3m/ subdirectories.

    Handles two layouts:
      - Flat: input_dir contains .a3m files directly
      - Nested: input_dir/<protein_id>/a3m/*.a3m (OpenProteinSet layout)
    """
    # Check for flat layout first
    flat_a3ms = [f for f in os.listdir(input_dir) if f.endswith(".a3m")]
    if flat_a3ms:
        return [(os.path.splitext(f)[0], os.path.join(input_dir, f)) for f in sorted(flat_a3ms)]

    # Nested OpenProteinSet layout
    results = []
    for entry in sorted(os.listdir(input_dir)):
        protein_dir = os.path.join(input_dir, entry)
        a3m_dir = os.path.join(protein_dir, "a3m")
        if os.path.isdir(a3m_dir):
            # Collect all a3m files for this protein
            for a3m_file in sorted(os.listdir(a3m_dir)):
                if a3m_file.endswith(".a3m"):
                    results.append((entry, os.path.join(a3m_dir, a3m_file)))
        elif os.path.isdir(protein_dir):
            # Maybe a3m files directly in the protein folder
            for a3m_file in sorted(os.listdir(protein_dir)):
                if a3m_file.endswith(".a3m"):
                    results.append((entry, os.path.join(protein_dir, a3m_file)))
    return results


def encode_esm2(seq, model, alphabet, batch_converter, device):
    """Compute ESM2-650M per-residue embedding. Returns tensor of shape (seq_len, 1280)."""
    data = [("query", seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_repr = results["representations"][33]
    # Strip BOS and EOS tokens
    emb = token_repr[:, 1 : len(seq) + 1, :].reshape(-1, 1280).detach().cpu()
    return emb


def main():
    parser = argparse.ArgumentParser(
        description="Build PLAME training dataset from OpenProteinSet A3M files"
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing A3M files or protein folders")
    parser.add_argument("--output_dir", required=True, help="Output directory for pickle files")
    parser.add_argument("--device", default="cuda:0", help="Device for ESM2 model")
    parser.add_argument("--min_msa_depth", type=int, default=64,
                        help="Minimum number of MSA sequences to keep a protein (default: 64)")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum query sequence length (default: 1024)")
    parser.add_argument("--max_msa_seqs", type=int, default=256,
                        help="Maximum MSA sequences to store per protein (default: 256)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip proteins that already have output pickle files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load ESM2-650M
    print("Loading ESM2-650M model...")
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval().to(args.device)
    print("ESM2-650M loaded.")

    # Find input files
    proteins = find_protein_folders(args.input_dir)
    if not proteins:
        # Fallback: find all a3m files recursively
        a3m_files = find_a3m_files(args.input_dir)
        proteins = [(os.path.splitext(os.path.basename(f))[0], f) for f in a3m_files]

    if not proteins:
        print(f"No A3M files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(proteins)} A3M entries to process")

    stats = {"processed": 0, "skipped_depth": 0, "skipped_length": 0, "skipped_exists": 0, "errors": 0}

    for protein_name, a3m_path in tqdm(proteins, desc="Building dataset"):
        output_path = os.path.join(args.output_dir, f"{protein_name}.pkl")

        if args.resume and os.path.exists(output_path):
            stats["skipped_exists"] += 1
            continue

        try:
            query_seq, aligned_seqs = parse_a3m(a3m_path)

            if query_seq is None:
                stats["errors"] += 1
                continue

            if len(query_seq) > args.max_seq_length:
                stats["skipped_length"] += 1
                continue

            if len(aligned_seqs) < args.min_msa_depth:
                stats["skipped_depth"] += 1
                continue

            # Limit MSA depth
            msa_seqs = aligned_seqs[: args.max_msa_seqs]

            # Compute ESM2 embedding
            emb = encode_esm2(query_seq, model, alphabet, batch_converter, args.device)

            record = {
                "name": protein_name,
                "seq": query_seq,
                "emb": emb,
                "msa": msa_seqs,
            }

            with open(output_path, "wb") as f:
                pickle.dump(record, f)

            stats["processed"] += 1

        except Exception as e:
            tqdm.write(f"Error processing {protein_name}: {e}")
            stats["errors"] += 1

    print("\n=== Dataset Build Complete ===")
    print(f"  Processed:          {stats['processed']}")
    print(f"  Skipped (depth):    {stats['skipped_depth']} (< {args.min_msa_depth} sequences)")
    print(f"  Skipped (length):   {stats['skipped_length']} (> {args.max_seq_length} residues)")
    if stats["skipped_exists"]:
        print(f"  Skipped (existing): {stats['skipped_exists']}")
    if stats["errors"]:
        print(f"  Errors:             {stats['errors']}")
    print(f"  Output directory:   {args.output_dir}")


if __name__ == "__main__":
    main()
