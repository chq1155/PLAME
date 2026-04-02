#!/usr/bin/env bash
# Download OpenProteinSet MSA data for PLAME training.
#
# OpenProteinSet (Ahdritz et al., 2023) provides pre-searched MSAs from
# OpenFold training. We download the PDB and UniClust30 subsets.
#
# Usage:
#   bash data/download_openproteinset.sh [OUTPUT_DIR]
#
# Requirements:
#   - AWS CLI v2 (install: https://aws.amazon.com/cli/)
#   - ~200 GB free disk space for PDB subset, ~1 TB for UniClust30

set -euo pipefail

OUTPUT_DIR="${1:-data/openproteinset}"
BUCKET="s3://openfold"

echo "=== PLAME Data Download ==="
echo "Output directory: ${OUTPUT_DIR}"

# Check AWS CLI
if ! command -v aws &>/dev/null; then
    echo "Error: AWS CLI is required. Install from https://aws.amazon.com/cli/"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# --- PDB alignments (smaller, recommended to start) ---
echo ""
echo "[1/2] Downloading PDB MSA alignments..."
echo "      This contains pre-computed MSAs for PDB training chains."
aws s3 sync \
    "${BUCKET}/pdb/" \
    "${OUTPUT_DIR}/pdb/" \
    --no-sign-request \
    --exclude "*" \
    --include "*/a3m/*"

# --- UniClust30 alignments (large, optional) ---
echo ""
echo "[2/2] Downloading UniClust30 MSA alignments..."
echo "      This is a large download (~1 TB). Press Ctrl+C to skip."
aws s3 sync \
    "${BUCKET}/uniclust30/" \
    "${OUTPUT_DIR}/uniclust30/" \
    --no-sign-request \
    --exclude "*" \
    --include "*/a3m/*"

echo ""
echo "=== Download complete ==="
echo "PDB data:        ${OUTPUT_DIR}/pdb/"
echo "UniClust30 data:  ${OUTPUT_DIR}/uniclust30/"
echo ""
echo "Next step: build training dataset with:"
echo "  python data/build_dataset.py --input_dir ${OUTPUT_DIR}/pdb --output_dir data/esm_msa/train"
