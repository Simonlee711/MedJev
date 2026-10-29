# Open MedJev

**How far do zero-shot System-One decisions go in medicine?**

Open MedJev evaluates MedJev Flash 9B, an open System-One model, on medical decisions, zero-shot
and running locally on Apple silicon. Every benchmark is cast as a [Jev](https://docs.typesafe.ai/introduction) request:
a **state** (text or JSON) plus typed questions, either a **Choice** over named options or a **Noul** (the probability
that a statement is true). The model answers each question with a full probability distribution from one forward pass,
without generating text.

| Benchmark | Question | Source |
|---|---|---|
| MedQA | USMLE-style vignettes, 5 options | MedQA test (BigBio) |
| USMLE | USMLE-style vignettes, 4 options | MedQA-USMLE-4-options test |
| NEJM | diagnostic vignettes (released open-ended) | NEJM-MedQA (Savage et al.) |
| DiagnosisArena | published case reports, 4 diagnoses | DiagnosisArena |
| ED Triage Disposition | ED disposition at triage, 6 classes | MIMIC-IV-ED |
| ED Triage Admission | hospital admission, decided at triage | MIMIC-IV-ED |
| ED Admission at 2 h | hospital admission, 2 h after ED arrival | MIMIC-IV (MEDS) |
| In-Hospital Mortality | in-hospital death, 24 h after admission | MIMIC-IV (MEDS) |
| Length of Stay > 7 Days | stay longer than 7 days, 24 h after admission | MIMIC-IV (MEDS) |
| 30-Day Readmission | readmission within 30 days, at discharge | MIMIC-IV (MEDS) |

MedJev is compared with zero-shot LLMs read out identically (Qwen3.5-9B, its base model; Qwen3.5-2B; Pythia-2.8B; and,
on the knowledge benchmarks, MedGemma-1.5-4B and Gemma-3-4B), with local recalibration of its own probabilities, and,
on the EHR benchmarks, with supervised XGBoost and TF-IDF + logistic regression models.

## Setup

Two Python environments: `.venv` for data preparation, baselines and evaluation (CPU), and `mlx` for the language
models (Apple silicon GPU through [MLX](https://github.com/ml-explore/mlx)).

```bash
# analysis environment
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .
brew install libomp                        # XGBoost needs OpenMP on macOS

# model environment
uv venv mlx --python 3.12
uv pip install --python mlx/bin/python -r requirements-mlx.txt
uv pip install --python mlx/bin/python --no-deps -e .

# MedJev weights (8-bit MLX) with its frozen helper in helper/
mlx/bin/hf download YOUR_HF_USERNAME/MedJev-Flash-9B-MLX --local-dir MedJev-Flash-9B-MLX

# zero-shot baselines (8-bit MLX builds)
mlx/bin/hf download mlx-community/Qwen3.5-9B-8bit         --local-dir models/Qwen3.5-9B-8bit
mlx/bin/hf download mlx-community/Qwen3.5-2B-MLX-8bit     --local-dir models/Qwen3.5-2B-8bit
mlx/bin/hf download mlx-community/medgemma-1.5-4b-it-8bit --local-dir models/medgemma-1.5-4b-it-8bit
mlx/bin/hf download mlx-community/gemma-3-4b-it-8bit      --local-dir models/gemma-3-4b-it-8bit
mlx/bin/hf download EleutherAI/pythia-2.8b --local-dir models/pythia-2.8b-hf --exclude "*.bin"
mlx/bin/python -m mlx_lm convert --hf-path models/pythia-2.8b-hf --mlx-path models/pythia-2.8b-8bit -q --q-bits 8
```

### Data

MedQA (BigBio parquet files) goes in `eval_datasets/bigbio:med_qa/`. USMLE, NEJM-MedQA and DiagnosisArena are downloaded
from the Hugging Face Hub on first use.

The EHR benchmarks need **credentialed PhysioNet access** to [MIMIC-IV](https://physionet.org/content/mimiciv/) and
[MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/):

- `eval_datasets/MIMIC_MEDS_Extract_v0.0.3_test/`: a [MEDS-Transforms](https://github.com/mmcdermott/MEDS_transforms)
  extract of MIMIC-IV v2.2 (`data/{train,tuning,held_out}/*.parquet`, `metadata/codes.parquet`, `splits.json`).
- `eval_datasets/mimic-iv-ed-2.2/ed/`: the MIMIC-IV-ED v2.2 tables (`edstays.csv.gz`, `triage.csv.gz`, ...).

Other locations can be set with `MEDJEV_MEDQA_DIR`, `MEDJEV_MEDS_DIR` and `MEDJEV_EVAL_DATASETS`.

## Run

```bash
src/scripts/run_all.sh                     # everything, on the full benchmark splits
LIMIT=500 src/scripts/run_all.sh           # optional cap on examples per task, for a quick run
```

Or step by step:

| Step | Command | Output |
|---|---|---|
| Jev requests (evaluation sets) | `.venv/bin/python src/scripts/prepare_data.py [--limit N]` | `data/prepared/` |
| Jev requests (training sets) | `.venv/bin/python src/scripts/prepare_data.py --set train [--limit N]` | `data/prepared_train/` |
| Zero-shot answers (MLX) | `src/scripts/run_variants_gpu.sh all` | `runs/`, `runs_train/` |
| Local recalibration | `.venv/bin/python src/scripts/recalibrate.py` | `runs/medjev-recal/`, `weights/calibration/` |
| Supervised baselines | `.venv/bin/python src/scripts/run_baselines.py` | `runs/xgboost/`, `runs/tfidf-lr/` |
| Scores and metrics | `.venv/bin/python src/scripts/evaluate.py` | `scores/`, `results/` |
| Label efficiency | `.venv/bin/python src/scripts/label_efficiency.py` | `results/label_efficiency.json` |
| Benchmark statistics | `.venv/bin/python src/scripts/benchmark_stats.py` | `results/benchmark_stats.json` |
| Figures (300 dpi) | `.venv/bin/python src/scripts/make_figures.py` | `results/figures/` |

Figures use Product Sans if its `.ttf` is placed in `fonts/` (not distributed here), otherwise matplotlib's default
sans-serif. All output locations can be redirected with `MEDJEV_{DATA,RUNS,TRAIN_RUNS,RESULTS,SCORES,WEIGHTS}_DIR`.

Every model output is cached on disk, keyed by the exact request (state and questions). Reruns only compute what is
missing or what changed, and every analysis step reads the cache without querying a model.

The model can also be served as a Jev-compatible HTTP endpoint (`src/scripts/serve_medjev.sh`, then
`src/scripts/run_jev.py`). The local MLX engine (`src/medjev/engine.py`) loads the same frozen helper and reproduces the
server's answers exactly.

## Layout

```
src/medjev/
  config.py                  paths (environment-overridable)
  jev.py                     Jev request format, examples, resumable HTTP client
  engine.py                  local MLX engine: the frozen MedJev helper's prompts and readout, for any MLX model
  metrics.py                 accuracy, AUROC/AUPRC, ECE, Brier, bootstrap CIs
  plotting.py                figure style
  tasks/                     benchmark builders (medqa, qa, meds, ed_triage)
  baselines/                 XGBoost and TF-IDF references
src/scripts/                 pipeline entry points (see the table above)
```

## Data use

MIMIC-IV and MIMIC-IV-ED are credentialed data. Run everything locally (the model server binds to the loopback
interface) and do not commit or share anything under `data/`, `runs/`, `runs_train/`, `scores/` or `weights/`: these
hold individual-level, MIMIC-derived content. The models evaluated here are research artefacts, not clinical tools.

## License

Code: MIT (see `LICENSE`). MedJev weights: CC BY-NC 4.0 (see the model card). Each benchmark and model keeps its own license.
