#!/usr/bin/env bash
# Train PLAME on 4 GPUs (alternative to finetune_v1.sh which uses 8 GPUs).
# Override DATA_DIR and OUTPUT_DIR via environment variables.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

DATA_DIR="${DATA_DIR:-data/esm_msa/train}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/plame}"

PYTHONPATH=. accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  scripts/finetune.py \
  --output_dir "${OUTPUT_DIR}" \
  --dataset_name openfold \
  --train_file "${DATA_DIR}" \
  --remove_unused_columns False \
  --do_train True \
  --overwrite_output_dir True \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --num_train_epochs 200 \
  --max_steps 200000 \
  --learning_rate 6e-5 \
  --lr_scheduler_type polynomial \
  --warmup_ratio 0.01 \
  --weight_decay 1e-5 \
  --metric_for_best_model eval_loss \
  --load_best_model_at_end True \
  --evaluation_strategy steps \
  --eval_steps 10000 \
  --save_strategy steps \
  --save_steps 10000 \
  --save_total_limit 20 \
  --prediction_loss_only True \
  --num_alignments 32 \
  --threshold 512 \
  --gradient_accumulation_steps 4 \
  --bf16 True \
  --max_grad_norm 1.0
