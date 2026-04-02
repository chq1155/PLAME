# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PLAME (Protein MSA Fold) is a T5-based encoder-decoder model for generating protein multiple sequence alignments (MSAs) from ESM-2 evolutionary embeddings. It uses axial attention (row + column) to handle the 2D MSA structure and a conservation-diversity loss (PCE + DIRE) to balance conserved positions with sequence variation.

## Environment Setup

```bash
conda create -n plame python=3.10
pip install -r requirements.txt
```

Key dependencies: PyTorch >= 2.0, Transformers >= 4.21, fair-esm, Accelerate, Biopython.

All scripts require `PYTHONPATH=.` to find the `plame` package.

## Common Commands

### Data Preparation
```bash
# Download OpenProteinSet MSAs (requires AWS CLI)
bash data/download_openproteinset.sh data/openproteinset

# Build training dataset (A3M → pickle with ESM2 embeddings)
python data/build_dataset.py --input_dir data/openproteinset/pdb --output_dir data/esm_msa/train --device cuda:0

# Prepare inference input from A3M files
python data/prepare_a3m.py --input your_file.a3m --output_dir data/inference_input --device cuda:0
```

### Training
```bash
# 8-GPU (primary recipe)
DATA_DIR=data/esm_msa/train OUTPUT_DIR=outputs/plame bash finetune_v1.sh

# 4-GPU (alternative)
DATA_DIR=data/esm_msa/train OUTPUT_DIR=outputs/plame bash finetune.sh
```

### Inference
```bash
CHECKPOINT_DIR=./checkpoint-160000 DATA_PATH=data/inference_input OUTPUT_DIR=outputs/results bash inference.sh
```

## Architecture

### Model Stack (`plame/models/`)

- **`msa.py` — `MSAT5`**: Main model class (T5PreTrainedModel). Encoder processes ESM embeddings (1280-dim projected to d_model) with axial (tied row) attention. Decoder generates protein sequences autoregressively with optional column cross-attention. Contains the custom `generate()` method for MSA generation.
- **`model.py` — `MSA_AUGMENTOR`**: Training wrapper that adds `PSSMWeightedCELoss` (combined PCE + DIRE loss) on top of MSAT5. Same architecture as MSAT5 but with loss computation for training.

### Data Pipeline (`plame/data/`)

- **`msadata.py`**: `Alphabet` (33-token protein vocabulary), `MSABatchConverter` (handles ESM/MSA/inference batching), and dataset classes (`MSADataSet_v3` for training, `MSAInferenceDataSet` for inference).
- **`constant.py`**: Protein token definitions (26 amino acids + gap).
- Training data format: pickle files with `{name, seq, emb (L×1280 tensor), msa (list of aligned strings)}`.

### Training (`scripts/finetune.py`)

Uses `GradientClippingTrainer` (custom HuggingFace Trainer with gradient clipping and auxiliary loss logging). FSDP with auto-wrap on T5Block. Dataset split: 90/5/5 train/valid/test.

### Inference (`scripts/inference.py`)

Loads checkpoint, generates sequences via sampling, filters invalid outputs, writes A3M files. Two modes: `orphan` (zero-shot) and `artificial` (MSA augmentation).

## Model Configuration

Config in `config/`. Key dims: d_model=768 (checkpoint-160000), d_ff=2048, 12 encoder/decoder layers, 12 heads, vocab_size=33. Axial attention and decoder column attention enabled by default.

## Training Details

- Precision: bfloat16
- Distributed: Accelerate + FSDP (full_shard auto_wrap on T5Block)
- Loss: α·PCE + (1-α)·DIRE, α=0.9
- Hyperparams: lr=5e-5, polynomial schedule, 200k max steps, batch_size=4/GPU, num_alignments=32, max_seq_len=512
