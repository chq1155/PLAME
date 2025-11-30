# Protein MSA Fold (PLAME)

Utilities for training and inference on protein multiple sequence alignments.

## Environment
- Recommended: `conda create -n protab python=3.8`  
  Install dependencies with `pip install -r requirements.txt` or `conda env create -f environment.yml`.
- Models rely on PyTorch/Accelerate; ensure GPUs are visible via `CUDA_VISIBLE_DEVICES`.

## Layout
- `plame/`: core library (`models/` for architectures, `data/` for datasets and collators).
- `scripts/finetune.py`: training entry; launched by `finetune_v1.sh`.
- `scripts/inference.py`: generation entry; launched by `inference.sh`.

## Finetuning
- Main entry point: `finetune_v1.sh` (8 GPUs, cudagraphs enabled).  
  Override defaults with environment variables:
  - `DATA_DIR` (default `data/esm_msa/train`)
  - `OUTPUT_DIR` (default `outputs/plame-v2-pdb`)
- An alternative 4-GPU recipe is provided in `finetune.sh`. Adjust batch sizes and `DATA_DIR`/`OUTPUT_DIR` as needed.

## Inference
- Run `inference.sh` to generate sequences. Configure paths via:
  - `CHECKPOINT_DIR` (default `./openfold32/checkpoint-200000`)
  - `DATA_PATH` (default `data/enzyme`)
  - `OUTPUT_DIR` (default `outputs/enzyme_plame`)

## Notes
- Dataset and helper scripts now default to relative paths; override via environment variables when pointing to custom data.
- All comments and logs are in English for easier sharing and collaboration.
