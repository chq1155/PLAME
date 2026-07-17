# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PLAME (Protein MSA Fold) is a T5-based encoder-decoder that generates protein multiple sequence alignments (MSAs) from ESM-2 (esm2_t33_650M_UR50D, 1280-dim) per-residue embeddings. The encoder uses axial attention (tied row + column) over the 2D MSA; the decoder generates sequences autoregressively with cross-attention plus an extra column attention over the MSA depth. Goal: produce MSAs that improve downstream folding (AlphaFold2/3, ESMFold) for low-homology / orphan proteins.

## Environment Setup

```bash
conda create -n plame python=3.10
pip install -r requirements.txt   # or: conda env create -f environment.yml
```

Key deps: PyTorch >= 2.0, fair-esm, Accelerate, Biopython, sentencepiece.

⚠️ **Transformers must be 4.x** (checkpoint trained on 4.42.3). The model code imports private 4.x generation internals (`transformers.generation.beam_constraints`, `beam_search`, many `logits_process` symbols) that were **removed/relocated in transformers 5.x** — importing `plame` under 5.x fails with `ModuleNotFoundError: No module named 'transformers.generation.beam_constraints'`. `requirements.txt` pins `transformers>=4.21`, which is too loose; pin `<5` in practice. The repo's base/system Python may have an incompatible transformers — always run inside the dedicated `plame` env.

**All scripts require `PYTHONPATH=.`** to import the `plame` package (the `*.sh` wrappers set it for you).

## Common Commands

```bash
# Build training data: A3M dirs -> pickles with ESM2-650M embeddings
python data/build_dataset.py --input_dir data/openproteinset/pdb --output_dir data/esm_msa/train --device cuda:0
#   key flags: --min_msa_depth 64  --max_seq_length 1024  --max_msa_seqs 256  --resume

# Prepare inference input: A3M -> {name, seq, emb} pickle
python data/prepare_a3m.py --input your_file.a3m --output_dir data/inference_input --device cuda:0

# Train (primary 8-GPU recipe). finetune.sh is the 4-GPU variant (bs=1, grad_accum=4, lr=6e-5).
DATA_DIR=data/esm_msa/train OUTPUT_DIR=outputs/plame bash finetune_v1.sh

# Inference. inference.sh -> --mode orphan; inference-plame.sh -> --mode artificial.
CHECKPOINT_DIR=./checkpoint-160000 DATA_PATH=data/inference_input OUTPUT_DIR=outputs/results bash inference.sh
```

There is **no test suite, linter, or CI**. Smoke-check changes inside the `plame` env (transformers 4.x):
```bash
python -c "from plame import MSAT5, MSA_AUGMENTOR; print('OK')"   # import sanity (fails on transformers 5.x)
```
then a small inference run on `data/inference_input/`. Don't invent a `pytest`/`make` target — none exists.

## Architecture

### ⚠️ Two parallel T5 forks — the most important thing to know

`plame/models/msa.py` (~2.8k lines) and `plame/models/model.py` (~2.1k lines) **each contain a near-complete, duplicated fork of HuggingFace T5** (`T5Attention`, `T5LayerAxialAttention`, `T5DecLayerColAttention`, `T5Block`, `T5Stack`, `T5PreTrainedModel`, etc.). They are not shared — `model.py` only imports `MSAT5` from `msa.py` for an `isinstance` check.

- **`msa.py` → `MSAT5`**: the **inference / generation** model. Owns the custom `generate()` and all `MSA*LogitsWarper/Processor` classes.
- **`model.py` → `MSA_AUGMENTOR`**: the **training** model. Same architecture, but adds `PSSMWeightedCELoss` and computes `loss` in `forward()`.

**Consequence: any change to the shared attention/block/stack logic must be mirrored in BOTH files**, or training and inference silently diverge. `MSAT5` and `MSA_AUGMENTOR` share identical parameter names (`shared`, `esm_input`, `encoder`, `decoder`, `lm_head`), so a checkpoint saved as `MSA_AUGMENTOR` loads cleanly into `MSAT5` for inference.

### Attention variants (selected by config flags, both files)

- `is_axial_attention` → encoder block uses `T5LayerAxialAttention` = tied **row** self-attention (`has_tied_row_attention`) + **column** self-attention (`enc_col_attention`). Decoder forbids axial (raises).
- `is_dec_col_attention` → decoder block uses `T5DecLayerColAttention` = encoder-decoder cross-attention + a **column** attention across MSA depth (instead of plain `T5LayerCrossAttention`). Cross-attention pools encoder memory across sequences per `msa_cross_model` (`"avg"` default, or `"proj"`).
- Config note: the JSON field is `is_axial_attention`, but the model copies it onto `encoder_config.axial_attention` at build time — both names exist.

### Encoder input fusion (`T5Stack.forward`)

Not a plain projection. The encoder builds `inputs_embeds = shared(input_ids) * w + esm_input(esm)`, where `w` is a learned **per-sequence** weight (`pool` → `weight_head` sigmoid) and `esm_input` is the `Linear(1280 → d_model)` ESM projection. Tensors carry an MSA axis throughout: shape `[batch, num_alignments, seq_len, d_model]`.

### Loss (`PSSMWeightedCELoss` in `model.py`)

Returns `(loss, weighted_ce_loss, ce_loss, diversity_loss)`. `loss = weighted_ce_loss + self.dire_weight * diversity_loss`, i.e. **`L = L_PCE + λ·L_DIRE` with `λ = dire_weight = 0.1`** (constructor arg on `PSSMWeightedCELoss`; set `dire_weight=0` to disable DIRE). This matches the published objective. The PCE term is the PSSM-conservation-weighted cross-entropy (per-position conservation score → weight clamped to `[0.5, 1.5]`); the DIRE term is a negative-entropy diversity regularizer.

⚠️ Historical note: releases between commits `365a4b9` (2025-11-29) and the 2026-07 fix hardcoded this weight to `0.0`, silently disabling DIRE and contradicting the paper. Fixed 2026-07-16. Do not reintroduce a hardcoded weight here.

### Data pipeline (`plame/data/`)

- **`msadata.py`**: `Alphabet.from_architecture("msa_transformer")` (33-token vocab), `MSABatchConverter` (collator: `msa/esm/infer` batch conversion), and dataset classes. **Training uses `MSADataSet_v3`**; the other `MSADataSet*` variants exported from `plame/__init__.py` are legacy/unused. Inference uses `MSAInferenceDataSet`.
- **`constant.py`**: 27 base protein tokens (25 AA-ish letters + `.` + `-`); special tokens bring the vocab to 33.
- Training pickle format: `{name, seq, emb: Tensor(L,1280), msa: list[str]}`.
- **`data_construction.py`** is the legacy builder; **`build_dataset.py`** supersedes it. `handle.py` holds FASTA/A3M parsing helpers.

### Training entry (`scripts/finetune.py`)

`GradientClippingTrainer` (HF `Trainer` subclass) wraps the optimizer to clip grads each step and logs `weight_ce_loss / ce_loss / diversity_loss`. **FSDP is forced in code** (`training_args.fsdp = "full_shard auto_wrap"`, wrap class `T5Block`) regardless of launch flags, so even single-process runs get FSDP wrapping. Model is built fresh from `./config` (not a pretrained path). Dataset split is 90/5/5 via `random_split` (test set evaluated at the end for perplexity).

### Inference entry (`scripts/inference.py`)

Loads `MSAT5.from_pretrained(checkpoint)` (config read from the **checkpoint dir**), casts to bf16, generates by sampling, filters outputs (`len(set(seq)) >= 4` and length match), writes A3M files to `OUTPUT_DIR/<mode>/A<aug>T<trials>R<rep>T<temp>P<topp>/<protein>/generation_<trial>.a3m`.

**Inference gotchas:**
- The `generate()` call **hardcodes** `do_sample=True, top_k=5, top_p=0.95`. CLI `--temperature`, `--top_p`, `--do_sample`, `--num_beams` only affect the **output directory name**, not the actual sampling. Only `--repetition_penalty`, `--num_alignments`, `--augmentation_times`, `--trials_times` change behavior.
- `--mode orphan|artificial` currently only changes the output path. Both call `infer_batch_convert(..., plame=False)` and run identical generation. The MSA-conditioned paths (`plame=True` / `zero_shot`) exist in `MSAT5.generate` but are **not** wired up from the CLI.

## Model Configuration

Config files: `config/` (training; **d_model=1024**) and `checkpoint-160000/config.json` (released checkpoint; **d_model=768**). Shared dims: `d_ff=2048`, `d_kv=64`, 12 encoder + 12 decoder layers, 12 heads, `vocab_size=33`, `feed_forward_proj=gated-gelu`. `is_axial_attention` and `is_dec_col_attention` enabled.

⚠️ **Do not load `checkpoint-160000` with `./config`** — the 1024 vs 768 `d_model` mismatch breaks weight loading. Inference correctly reads config from the checkpoint directory.

## Training Details

- Precision: bfloat16; distributed: Accelerate + FSDP (`full_shard auto_wrap` on `T5Block`).
- Loss: `L = L_PCE + 0.1·L_DIRE` (PSSM-conservation-weighted CE + entropy diversity regularizer — see Loss section).
- Hyperparams (finetune_v1.sh): lr=5e-5, polynomial schedule, warmup_ratio=0.001, weight_decay=1e-5, max_steps=200000, batch_size=4/GPU, num_alignments=32, threshold(max_seq_len)=512, max_grad_norm=1.0.
- W&B logging is on by default (project `PLAME`); set `WANDB_DISABLED=true` to turn it off.
