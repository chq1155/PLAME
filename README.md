# PLAME: Lightweight MSA Design from Evolutionary Embeddings

PLAME is a lightweight framework for generating protein multiple sequence alignments (MSAs) from evolutionary embeddings. It uses a T5-based encoder-decoder architecture with axial attention to generate MSAs that improve downstream protein structure prediction, especially for low-homology and orphan proteins.

**Key features:**
- Generates MSAs de novo from ESM-2 embeddings — no existing homologs required
- Conservation-diversity loss (PCE + DIRE) balances conserved positions with sequence variation
- HiFiAD selection strategy filters high-quality generated MSA candidates
- Compatible with AlphaFold2, AlphaFold3, and ESMFold as downstream folding tools

> **Paper:** *Lightweight MSA Design Advances Protein Folding from Evolutionary Embeddings* (ICLR 2026)

---

## Installation

**Requirements:** Python 3.8+, CUDA-capable GPU (recommended)

```bash
# Clone the repository
git clone https://github.com/<your-org>/PLAME.git
cd PLAME

# Create conda environment
conda create -n plame python=3.10
conda activate plame

# Install dependencies
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "from plame import MSAT5, MSA_AUGMENTOR; print('OK')"
```

---

## Quick Start: Inference with Pretrained Model

**Download the pretrained checkpoint:**

| Checkpoint | Link |
|------------|------|
| `checkpoint-160000` (200K steps, d_model=768) | [Google Drive](https://drive.google.com/file/d/1mokB2dnxfm80QjtGEa2atgTw-YTxyqX9/view?usp=sharing) |

```bash
# Download and extract (or use gdown: pip install gdown)
gdown 1mokB2dnxfm80QjtGEa2atgTw-YTxyqX9
tar xzf checkpoint-160000.tar.gz
```

Generate MSAs for your proteins in three steps:

### Step 1: Prepare input data

Create A3M files for your query proteins. Each A3M file should contain at least the query sequence:

```
>protein_name
MKFLILLFNILCLFPVLAADNHGVS...
```

Convert A3M files to the pickle format expected by PLAME:

```bash
python data/prepare_a3m.py \
    --input your_proteins.a3m \
    --output_dir data/inference_input \
    --device cuda:0
```

This computes ESM2-650M embeddings and saves `{name, seq, emb}` pickle files.

### Step 2: Run inference

```bash
CHECKPOINT_DIR=./checkpoint-160000 \
DATA_PATH=data/inference_input \
OUTPUT_DIR=outputs/results \
    bash inference.sh
```

Or run directly with custom parameters:

```bash
PYTHONPATH=. python scripts/inference.py --do_predict \
    --checkpoints ./checkpoint-160000 \
    --data_path data/inference_input \
    --output_dir outputs/results \
    --device cuda:0 \
    --mode orphan \
    --num_alignments 32 \
    --temperature 1.0 \
    --top_p 0.95 \
    --trials_times 4 \
    --do_sample True
```

**Inference modes:**
| Parameter | Description |
|-----------|-------------|
| `--mode orphan` | Zero-shot: no existing MSA (for orphan proteins) |
| `--mode artificial` | Augmentation: supplement existing MSAs |
| `--num_alignments N` | Number of MSA sequences to generate |
| `--trials_times T` | Independent generation runs (produces T output files) |
| `--augmentation_times A` | Multiplier for generated sequences per trial |

### Step 3: Use generated MSAs

Output MSAs are saved as A3M files under:
```
outputs/results/<mode>/A<aug>T<trials>R<rep>T<temp>P<topp>/<protein_name>/generation_*.a3m
```

These can be directly used as input to AlphaFold2/3 or other structure prediction tools.

---

## Training

### Step 1: Download training data

PLAME is trained on MSAs from [OpenProteinSet](https://registry.opendata.aws/openfold/) (PDB + UniClust30 subsets):

```bash
# Requires AWS CLI (https://aws.amazon.com/cli/)
bash data/download_openproteinset.sh data/openproteinset
```

This downloads pre-computed A3M alignments. The PDB subset is ~200 GB; UniClust30 is ~1 TB.

### Step 2: Build training dataset

Process the raw A3M files into training pickles with ESM2-650M embeddings:

```bash
python data/build_dataset.py \
    --input_dir data/openproteinset/pdb \
    --output_dir data/esm_msa/train \
    --min_msa_depth 64 \
    --max_seq_length 512 \
    --device cuda:0 \
    --resume
```

Each output pickle contains:
| Key | Type | Description |
|-----|------|-------------|
| `name` | `str` | Protein identifier |
| `seq` | `str` | Query sequence |
| `emb` | `Tensor (L, 1280)` | ESM2-650M per-residue embedding |
| `msa` | `list[str]` | Aligned MSA sequences |

### Step 3: Launch training

**8-GPU training** (primary recipe from the paper):

```bash
DATA_DIR=data/esm_msa/train OUTPUT_DIR=outputs/plame bash finetune_v1.sh
```

**4-GPU training** (alternative with gradient accumulation):

```bash
DATA_DIR=data/esm_msa/train OUTPUT_DIR=outputs/plame bash finetune.sh
```

**Training hyperparameters** (from the paper):
| Parameter | Value |
|-----------|-------|
| Architecture | T5 encoder-decoder, 12+12 layers |
| Hidden dim | 768 (checkpoint) / 1024 (config) |
| Attention heads | 12 |
| Optimizer | AdamW, lr=5e-5, weight_decay=1e-5 |
| Schedule | Polynomial decay, 0.1% warmup |
| Precision | bfloat16 |
| Max steps | 200,000 |
| Batch size | 4 per GPU |
| MSA depth | 32 sequences per sample |

Training uses FSDP (Fully Sharded Data Parallel) with auto-wrapping on T5Block.

---

## Architecture

```
Input: ESM2-650M embedding (L x 1280)
  │
  ▼
┌──────────────────────────┐
│  Linear Projection       │  1280 → d_model
│  (esm_input layer)       │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Encoder (12 layers)     │  T5 blocks + row attention (axial)
│  Row attention: tied     │  Summarizes across MSA depth
│  across MSA rows         │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Decoder (12 layers)     │  T5 blocks + column cross-attention
│  Cross-attention to      │  Attends to homologous positions
│  encoder outputs         │  across sequences
└──────────┬───────────────┘
           ▼
  LM Head → 33-token vocabulary (20 AAs + special tokens + gap)
```

**Loss function** (conservation-diversity):
- **PCE Loss** (α=0.9): PSSM-weighted cross-entropy emphasizing conserved positions
- **DIRE Loss** (1-α=0.1): Entropy regularizer promoting sequence diversity

---

## Project Structure

```
PLAME/
├── plame/                      # Core library
│   ├── models/
│   │   ├── msa.py              # MSAT5: encoder-decoder with axial attention + generation
│   │   └── model.py            # MSA_AUGMENTOR: training wrapper with PCE+DIRE loss
│   └── data/
│       ├── msadata.py          # Dataset classes, tokenizer (Alphabet), batch converters
│       └── constant.py         # Amino acid token definitions
├── scripts/
│   ├── finetune.py             # Training entry point (HuggingFace Trainer)
│   └── inference.py            # Inference entry point (MSA generation)
├── data/
│   ├── prepare_a3m.py          # A3M → pickle for inference
│   ├── build_dataset.py        # OpenProteinSet → training pickles
│   └── download_openproteinset.sh  # Download raw MSA data
├── config/                     # T5 model config + tokenizer files
├── finetune_v1.sh              # 8-GPU training script
├── finetune.sh                 # 4-GPU training script
├── inference.sh                # Inference script (orphan mode)
└── inference-plame.sh          # Inference script (augmentation mode)
```

---

## Citation

```bibtex
@inproceedings{plame2026,
  title={Lightweight MSA Design Advances Protein Folding from Evolutionary Embeddings},
  author={Anonymous},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
