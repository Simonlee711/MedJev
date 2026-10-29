#!/usr/bin/env python
"""Local recalibration of zero-shot MedJev: the only local step on top of the zero-shot model. CPU only.

  binary (Noul)    Platt scaling: logistic regression on logit(p_yes), fitted on MedJev's zero-shot answers to the
                   training split. Two parameters.
  multiple choice  temperature scaling of the option distribution (one parameter), fitted on training answers: MedQA
                   training questions for the knowledge benchmarks, the task's own training split otherwise.

Both maps are monotone, so rankings are unchanged; only the probability scale moves.
  answers -> runs/medjev-recal/<task>.jsonl     maps -> weights/calibration/<task>.json

    python src/scripts/recalibrate.py
"""

import argparse
import json
import math
import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from medjev.config import DATA_DIR, PREPARED_DIR, RUNS_DIR, TRAIN_RUNS_DIR, WEIGHTS_DIR
from medjev.jev import read_examples, read_responses

JEV = "medjev"
OUT = "medjev-recal"
WEIGHTS = WEIGHTS_DIR / "calibration"
BINARY = ["ed_triage_admission", "meds_ed_disposition", "meds_mortality", "meds_long_los", "meds_readmission_30d"]
CHOICE = {"medqa": "medqa", "usmle4": "medqa", "diagnosisarena": "medqa", "nejm_medqa": "medqa",
          "ed_triage_disposition": "ed_triage_disposition"}  # task -> training answers the temperature is fitted on


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def write(task, answers: dict) -> None:
    path = RUNS_DIR / OUT / f"{task}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for uid, a in answers.items():
            f.write(json.dumps({"uid": uid, "answers": a, "model": OUT}) + "\n")


def platt(task: str) -> None:
    tr = read_examples(DATA_DIR / "prepared_train" / f"{task}.jsonl")
    te = read_examples(PREPARED_DIR / f"{task}.jsonl")
    zs_tr = read_responses(TRAIN_RUNS_DIR / JEV / f"{task}.jsonl")
    zs_te = read_responses(RUNS_DIR / JEV / f"{task}.jsonl")
    head = next(iter(te[0].questions))
    tr = [e for e in tr if e.uid in zs_tr]
    x = logit([zs_tr[e.uid]["answers"][head]["noul"] for e in tr])[:, None]
    y = np.array([e.gold[head] for e in tr], dtype=int)
    if len(set(y)) < 2:
        print(f"{task}: training answers contain one class only, skipped")
        return
    m = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)
    a, b = float(m.coef_[0, 0]), float(m.intercept_[0])
    out = {}
    for e in te:
        z = logit([zs_te[e.uid]["answers"][head]["noul"]])[0]
        out[e.uid] = {head: {"type": "noul", "noul": float(1 / (1 + math.exp(-(a * z + b))))}}
    write(task, out)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    (WEIGHTS / f"{task}.json").write_text(json.dumps({"type": "platt", "a": a, "b": b, "n_train": len(tr)}, indent=1))
    print(f"{task}: Platt a={a:.3f} b={b:.3f} (n={len(tr)})", flush=True)


def fit_temperature(answers: list[dict], golds: list[str]) -> float:
    logp = [np.log(np.clip(np.array(list(a["probabilities"].values())), 1e-9, 1)) for a in answers]
    idx = [list(a["probabilities"]).index(g) for a, g in zip(answers, golds)]

    def nll(t):
        return -np.mean([(lp / t)[i] - np.log(np.exp(lp / t).sum()) for lp, i in zip(logp, idx)])
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def rescale(a: dict, t: float) -> dict:
    keys = list(a["probabilities"])
    lp = np.log(np.clip(np.array([a["probabilities"][k] for k in keys]), 1e-9, 1)) / t
    p = np.exp(lp - lp.max())
    p /= p.sum()
    best = keys[int(np.argmax(p))]
    return {"type": "choice", "choice": best, "probabilities": dict(zip(keys, p.tolist())),
            "confidence": float(max(0.0, (p.max() - 1 / len(p)) / (1 - 1 / len(p))))}


def temperature(task: str, source: str) -> None:
    tr = read_examples(DATA_DIR / "prepared_train" / f"{source}.jsonl")
    zs_tr = read_responses(TRAIN_RUNS_DIR / JEV / f"{source}.jsonl")
    tr = [e for e in tr if e.uid in zs_tr]
    t = fit_temperature([zs_tr[e.uid]["answers"]["answer"] for e in tr], [e.gold["answer"] for e in tr])
    te = read_examples(PREPARED_DIR / f"{task}.jsonl")
    zs_te = read_responses(RUNS_DIR / JEV / f"{task}.jsonl")
    write(task, {e.uid: {"answer": rescale(zs_te[e.uid]["answers"]["answer"], t)} for e in te if e.uid in zs_te})
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    (WEIGHTS / f"{task}.json").write_text(json.dumps({"type": "temperature", "T": t, "fitted_on": source, "n_train": len(tr)}, indent=1))
    print(f"{task}: temperature T={t:.3f} (fitted on {source} train, n={len(tr)})", flush=True)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*", default=BINARY + list(CHOICE))
    args = ap.parse_args()
    for task in args.tasks:
        src = CHOICE.get(task, task)
        if not (PREPARED_DIR / f"{task}.jsonl").is_file() or not (TRAIN_RUNS_DIR / JEV / f"{src}.jsonl").is_file():
            print(f"{task}: no evaluation set or training answers, skipped")
            continue
        if task in BINARY:
            platt(task)
        elif task in CHOICE:
            temperature(task, CHOICE[task])


if __name__ == "__main__":
    main()
