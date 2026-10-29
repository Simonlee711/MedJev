#!/usr/bin/env bash
# End-to-end Open MedJev evaluation on the full benchmark splits.
#   LIMIT=500 src/scripts/run_all.sh     # optional cap on examples per task, for a quick run
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LIM=${LIMIT:+--limit $LIMIT}
mkdir -p logs

$PY src/scripts/prepare_data.py $LIM                     # evaluation sets   -> data/prepared/
$PY src/scripts/prepare_data.py --set train $LIM         # training sets     -> data/prepared_train/
src/scripts/run_variants_gpu.sh all                       # zero-shot answers  -> runs/, runs_train/   (MLX)
$PY src/scripts/recalibrate.py                            # local recalibration -> runs/medjev-recal/
$PY src/scripts/run_baselines.py                          # XGBoost, TF-IDF    -> runs/xgboost/, runs/tfidf-lr/
$PY src/scripts/evaluate.py                               # scores + metrics   -> scores/, results/
$PY src/scripts/label_efficiency.py                       # label budgets      -> results/label_efficiency.json
$PY src/scripts/benchmark_stats.py                        # dataset statistics -> results/benchmark_stats.json
$PY src/scripts/make_figures.py                           # figures            -> results/figures/
