#!/usr/bin/env python
"""Full statistics of every benchmark (all splits, complete cohorts) -> results/benchmark_stats.json. CPU only.

    python src/scripts/benchmark_stats.py
"""

import json
from collections import Counter

import pandas as pd

from medjev.config import MEDQA_DIR, RESULTS_DIR
from medjev.tasks import ed_triage, meds, qa

MEDS_SPLITS = {"train": "train", "val": "tuning", "test": "held_out"}


def letter_spread(golds) -> dict:
    c = Counter(golds)
    n = sum(c.values())
    return {k: c[k] / n for k in sorted(c)}


def main():
    out = {}
    # ---- MedQA (BigBio, 5 options)
    sizes, test_gold = {}, []
    for name, split in (("train", "train"), ("val", "validation"), ("test", "test")):
        df = pd.read_parquet(MEDQA_DIR / f"{split}-00000-of-00001.parquet")
        sizes[name] = len(df)
        if split == "test":
            test_gold = ["ABCDE"[list(c).index(a[0])] for c, a in zip(df["choices"], df["answer"])]
    out["medqa"] = {"unit": "questions", **sizes, "options": 5, "gold": letter_spread(test_gold)}
    # ---- USMLE (MedQA-USMLE-4-options)
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")
    out["usmle4"] = {"unit": "questions", "train": len(ds["train"]), "val": None, "test": len(ds["test"]), "options": 4,
                     "gold": letter_spread(ds["test"]["answer_idx"])}
    # ---- NEJM and DxArena (test only)
    nejm = qa.build_nejm()
    out["nejm_medqa"] = {"unit": "vignettes", "train": None, "val": None, "test": len(nejm), "options": 5,
                         "gold": letter_spread([e.gold["answer"] for e in nejm])}
    dx = qa.build_diagnosisarena()
    out["diagnosisarena"] = {"unit": "cases", "train": None, "val": None, "test": len(dx), "options": 4,
                             "gold": letter_spread([e.gold["answer"] for e in dx])}
    # ---- MIMIC-IV-ED triage
    t = ed_triage.table()
    by = {name: t[t["split"] == split] for name, split in MEDS_SPLITS.items()}
    common = {"unit": "ED stays", **{k: int(len(v)) for k, v in by.items()}, "patients": int(t["subject_id"].nunique())}
    out["ed_triage_admission"] = {**common, "positive_rate": float((t["disposition"] == "ADMITTED").mean())}
    dist = t["category"].value_counts(normalize=True).to_dict()
    out["ed_triage_disposition"] = {**common, "options": 6, "classes": {k: float(v) for k, v in dist.items()}}
    # ---- MIMIC-IV MEDS cohorts (all train shards, tuning, held-out)
    for task in ("meds_ed_disposition", "meds_mortality", "meds_long_los", "meds_readmission_30d"):
        parts = {name: meds.cohort(task, split) for name, split in MEDS_SPLITS.items()}
        n = {k: int(v.height) for k, v in parts.items()}
        patients = sum(int(v["patient_id"].n_unique()) for v in parts.values())
        pos = sum(float(v["label"].sum()) for v in parts.values()) / sum(n.values())
        out[task] = {"unit": "ED visits" if task == "meds_ed_disposition" else "admissions", **n, "patients": patients,
                     "positive_rate": pos}
        print(task, out[task], flush=True)
    (RESULTS_DIR / "benchmark_stats.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {a: b for a, b in v.items() if a not in ("gold", "classes")})


if __name__ == "__main__":
    main()
