#!/usr/bin/env bash
# Zero-shot answers for MedJev and every zero-shot LLM baseline (MLX), one model at a time. Stages:
#   medjev     MedJev on every evaluation set
#   train      MedJev on the training sets (answers used only for recalibration)
#   baselines  Qwen3.5-9B (base), Qwen3.5-2B and Pythia-2.8B on every evaluation set; MedGemma-1.5-4B and
#              Gemma-3-4B on the knowledge benchmarks
#   all        every stage in this order, stopping at the first failure
# Long runs, detached and awake:  nohup caffeinate -is src/scripts/run_variants_gpu.sh all > logs/gpu.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
X="mlx/bin/python src/scripts/extract_jev.py"
QA="medqa usmle4 diagnosisarena nejm_medqa"
EHR="ed_triage_admission ed_triage_disposition meds_ed_disposition meds_mortality meds_long_los meds_readmission_30d"
TRAIN="medqa $EHR"

stage() {
  echo "=== $1 $(date)"
  case "$1" in
    medjev)    $X $QA $EHR --set test --no-features ;;
    train)     $X $TRAIN --set train --no-features ;;
    baselines) $X $QA $EHR --set test --no-features --model-dir models/Qwen3.5-9B-8bit --variant qwen3.5-9b-base
               $X $QA $EHR --set test --no-features --model-dir models/Qwen3.5-2B-8bit --variant qwen3.5-2b
               $X $QA $EHR --set test --no-features --model-dir models/pythia-2.8b-8bit --variant pythia-2.8b \
                  --plain --max-tokens 2040
               $X $QA --set test --no-features --model-dir models/medgemma-1.5-4b-it-8bit --variant medgemma-1.5-4b
               $X $QA --set test --no-features --model-dir models/gemma-3-4b-it-8bit --variant gemma-3-4b ;;
    *) echo "unknown stage $1"; exit 1 ;;
  esac
  echo "=== $1 finished $(date)"
}

if [[ "${1:-all}" == all ]]; then
  for s in medjev train baselines; do stage "$s"; done
else
  stage "$1"
fi
