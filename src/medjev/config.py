"""Paths and defaults. Every path can be overridden with an environment variable."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EVAL_DATASETS = Path(os.environ.get("MEDJEV_EVAL_DATASETS", ROOT / "eval_datasets"))
MEDQA_DIR = Path(os.environ.get("MEDJEV_MEDQA_DIR", EVAL_DATASETS / "bigbio:med_qa"))
MEDS_DIR = Path(os.environ.get("MEDJEV_MEDS_DIR", EVAL_DATASETS / "MIMIC_MEDS_Extract_v0.0.3_test"))

DATA_DIR = Path(os.environ.get("MEDJEV_DATA_DIR", ROOT / "data"))  # prepared examples + cohort caches (git-ignored)
PREPARED_DIR = DATA_DIR / "prepared"
RUNS_DIR = Path(os.environ.get("MEDJEV_RUNS_DIR", ROOT / "runs"))  # raw model responses (git-ignored)
TRAIN_RUNS_DIR = Path(os.environ.get("MEDJEV_TRAIN_RUNS_DIR", ROOT / "runs_train"))  # answers on training sets
RESULTS_DIR = Path(os.environ.get("MEDJEV_RESULTS_DIR", ROOT / "results"))  # aggregate metrics only
SCORES_DIR = Path(os.environ.get("MEDJEV_SCORES_DIR", ROOT / "scores"))  # per-example scores (git-ignored: MIMIC-derived)
WEIGHTS_DIR = Path(os.environ.get("MEDJEV_WEIGHTS_DIR", ROOT / "weights"))  # fitted recalibration maps (git-ignored)

JEV_URL = os.environ.get("MEDJEV_JEV_URL", "http://localhost:3000/v1/systemone")
JEV_MODEL_NAME = os.environ.get("MEDJEV_JEV_MODEL", "medjev")

