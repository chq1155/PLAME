#!/usr/bin/env bash
# Run PLAME inference to generate MSA sequences.
# Override paths via environment variables.

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoint-160000}"
DATA_PATH="${DATA_PATH:-data/inference_input}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/plame_output}"

PYTHONPATH=. python scripts/inference.py --do_predict \
  --checkpoints "${CHECKPOINT_DIR}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --mode orphan \
  --num_alignments 32 \
  --augmentation_times 1 \
  --trials_times 1 \
  --repetition_penalty 1.0 \
  --temperature 1.0 \
  --top_p 0.95 \
  --do_sample True \
  --num_beams 1 \
  --num_beam_groups 1
