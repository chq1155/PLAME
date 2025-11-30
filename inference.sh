#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./openfold32/checkpoint-200000}"
DATA_PATH="${DATA_PATH:-data/enzyme}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/enzyme_plame}"

python scripts/inference.py --do_predict \
  --checkpoints "${CHECKPOINT_DIR}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --mode artificial \
  --num_alignments 100 \
  --augmentation_times 1 \
  --trials_times 1 \
  --repetition_penalty 1.0 \
  --temperature 1.0 \
  --top_p 0.95 \
  --do_sample True \
  --num_beams 1 \
  --num_beam_groups 1
