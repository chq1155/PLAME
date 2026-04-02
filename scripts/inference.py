#!/usr/bin/env python
# coding=utf-8
"""PLAME inference: generate MSA sequences from ESM2 embeddings."""

import argparse
import json
import logging
import os
import time
import warnings

import torch
from transformers import T5Config

from plame.data.msadata import Alphabet, MSABatchConverter, MSAInferenceDataSet
from plame.models.msa import MSAT5

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d %H:%M:%S",
    level=logging.INFO,
)


def adjust_sequences(sequence_list, target_length):
    """Pad or truncate sequences to target_length."""
    return [
        seq[:target_length] if len(seq) > target_length else seq.ljust(target_length, "-")
        for seq in sequence_list
    ]


def msa_generate(args, model, dataset, msa_collator, tokenizer):
    """Generate MSA sequences for each protein in the dataset."""
    protein_count = 0
    with torch.no_grad():
        output_dir = os.path.join(
            args.output_dir,
            args.mode,
            f"A{args.augmentation_times}T{args.trials_times}R{args.repetition_penalty}T{args.temperature}P{args.top_p}",
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "params.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        logger.info("Input proteins: %d", len(dataset))

        for protein_data in dataset:
            protein_count += 1
            logger.info("Processing protein %d/%d: %s", protein_count, len(dataset),
                        os.path.basename(protein_data["name"]).split(".")[0])
            infer_time_avg = 0.0

            msa_name = os.path.basename(protein_data["name"]).split(".")[0]
            original_seq = protein_data["seq"]
            esm = protein_data["emb"]
            input_ids = [original_seq, esm]
            esm, src_ids = msa_collator.infer_batch_convert(input_ids, args.num_alignments, plame=False)
            esm = esm.to(args.device)
            src_ids = src_ids.to(args.device)

            _, original_seq_num, original_seq_len = src_ids.size()

            for trial in range(args.trials_times):
                msa_output_dir = os.path.join(output_dir, msa_name)
                os.makedirs(msa_output_dir, exist_ok=True)
                a3m_file_name = os.path.join(msa_output_dir, f"generation_{trial}.a3m")
                if os.path.exists(a3m_file_name):
                    logger.info("File %s already exists, skip", a3m_file_name)
                    continue

                start = time.time()
                output = model.generate(
                    src_ids, esm,
                    do_sample=True, top_k=5, top_p=0.95,
                    repetition_penalty=args.repetition_penalty,
                    max_length=original_seq_len + 1,
                    gen_seq_num=original_seq_num * args.augmentation_times,
                )
                end = time.time()
                infer_time_avg += (end - start) / args.trials_times

                generate_seq = [
                    tokenizer.decode(seq_token, skip_special_tokens=True).replace(" ", "")
                    for seq_token in output[0]
                ]
                generate_seq = adjust_sequences(generate_seq, original_seq_len - 1)
                generate_seq = [
                    s for s in generate_seq
                    if len(set(s)) >= 4 and len(s) == len(original_seq)
                ]

                with open(a3m_file_name, "w") as fw:
                    lines = [">" + msa_name, original_seq]
                    for i, seq in enumerate(generate_seq):
                        seq_name = f"MSAT5_Generate_condition_on_{src_ids.size(1)}_seq_from_{msa_name}_{i}"
                        lines.append(">" + seq_name)
                        lines.append(seq)
                    fw.write("\n".join(lines))
                    logger.info("Generated %d sequences for %s trial %d", len(generate_seq), msa_name, trial)

            logger.info("Average inference time: %.2fs", infer_time_avg)


def inference(args):
    """Load model and run MSA generation."""
    config = T5Config.from_pretrained("./config/")
    tokenizer = Alphabet.from_architecture(name="msa_transformer")
    msadata_collator = MSABatchConverter(tokenizer)

    if args.checkpoints:
        logger.info("Loading model from %s", args.checkpoints)
        model = MSAT5.from_pretrained(args.checkpoints).to(args.device)
    else:
        logger.warning("Loading a random model")
        model = MSAT5(config).to(args.device)

    model = model.to(torch.bfloat16)
    dataset = MSAInferenceDataSet(args)

    msa_generate(
        args,
        model=model,
        msa_collator=msadata_collator,
        dataset=dataset,
        tokenizer=tokenizer,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="PLAME MSA generation")
    parser.add_argument("--do_predict", action="store_true", help="Run inference")
    parser.add_argument("--checkpoints", type=str, help="Path to model checkpoint directory")
    parser.add_argument("--data_path", type=str, help="Path to input data directory (pickle files)")
    parser.add_argument("-o", "--output_dir", type=str, default="./output/", help="Output directory")
    parser.add_argument("--num_alignments", type=int, default=32, help="Number of MSA sequences to generate")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device (e.g. cuda:0)")

    # Generation parameters
    parser.add_argument("--mode", type=str, choices=["orphan", "artificial"], required=True,
                        help="orphan: zero-shot (no existing MSA); artificial: augment existing MSA")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("-a", "--augmentation_times", type=int, default=1,
                        help="Multiplier for generated sequences (1x, 3x, 5x)")
    parser.add_argument("-t", "--trials_times", type=int, default=5,
                        help="Number of independent generation runs")
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_beam_groups", type=int, default=1)
    parser.add_argument("--diversity_penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    logger.info("Parameters: %s", json.dumps(vars(args), indent=4, sort_keys=True))
    inference(args)
