#!/usr/bin/env bash
# Serve MedJev Flash 9B (MLX, Apple silicon) as a Jev-compatible endpoint: POST http://localhost:3000/v1/systemone
# Setup (once): see README.md -> "Setup". Calibration settings are the ones on the MLX model card.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL_DIR="${MODEL_DIR:-MedJev-Flash-9B-MLX}"
HELPER_DIR="${HELPER_DIR:-$MODEL_DIR/helper}"
PORT="${PORT:-3000}"

TOKENIZER="$MODEL_DIR" SHIM_MODEL="$MODEL_DIR" \
READOUT_T=1.07 READOUT_NOUL_T=1.074766 READOUT_NOUL_BIAS=0 \
READOUT_TARGETED=1 READOUT_INSTR_STYLE=pyrepr SHIM_STAGGER=1 \
  exec mlx/bin/python "$HELPER_DIR/shim_mlx.py" \
  --helper "$HELPER_DIR/shim.py" --model "$MODEL_DIR" --port "$PORT" "$@"
