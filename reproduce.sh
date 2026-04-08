#!/usr/bin/env bash
# Reproduce all results for:
#   "A Deep Learning Search for Multiverse Signatures in the CMB"
#
# Requirements: Python 3.10+, CUDA-capable GPU recommended
# Expected runtime: ~24 hours on a single A100 GPU
set -euo pipefail

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Step 1/7: Downloading Planck data ==="
python run_pipeline.py --step download

echo "=== Step 2/7: Generating training data ==="
python run_pipeline.py --step generate --n-train 10000

echo "=== Step 3/7: Training S-CNN ensemble ==="
python run_pipeline.py --step train --device cuda

echo "=== Step 4/7: Calibrating probabilities ==="
python run_pipeline.py --step calibrate --device cuda

echo "=== Step 5/7: Running inference on Planck maps ==="
python run_pipeline.py --step infer --device cuda

echo "=== Step 6/7: Statistical analysis ==="
python run_pipeline.py --step analyze --device cuda

echo "=== Step 7/7: Generating figures ==="
python run_pipeline.py --step figures

echo "=== Done. Results in output/, figures in output/figures/ ==="
