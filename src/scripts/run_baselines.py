#!/usr/bin/env python
"""Supervised reference baselines for every task, trained on the prepared training sets. CPU only.

  xgboost    MEDS: bag-of-codes counts and last values. ED triage: structured triage fields + chief-complaint tokens
             (the standard tabular approach).
  tfidf-lr   TF-IDF on the serialised state + logistic regression.

Hyperparameters are chosen by 5-fold cross-validation on the training examples only (no other labels), then the
model is refit on all of them. Answers -> runs/<model>/<task>.jsonl in Jev format; the EHR feature matrices are cached
in data/cache/baselines/ for the label-efficiency analysis.

    python src/scripts/run_baselines.py              # every task
    python src/scripts/run_baselines.py meds_mortality ed_triage_admission
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from run_baselines_triage import triage_matrices  # noqa: E402

from medjev.baselines.supervised import fit_xgb, text_of, tfidf, tune_lr, tune_xgb  # noqa: E402
from medjev.config import DATA_DIR, PREPARED_DIR, RESULTS_DIR, RUNS_DIR  # noqa: E402
from medjev.jev import read_examples  # noqa: E402

MEDS = ["meds_ed_disposition", "meds_mortality", "meds_long_los", "meds_readmission_30d"]
TRIAGE = ["ed_triage_admission", "ed_triage_disposition"]
CACHE = DATA_DIR / "cache" / "baselines"


def write(model: str, task: str, answers: dict[str, dict]) -> None:
    path = RUNS_DIR / model / f"{task}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for uid, a in answers.items():
            f.write(json.dumps({"uid": uid, "answers": a, "model": model}) + "\n")


def noul(p) -> dict:
    return {"label": {"type": "noul", "noul": float(p)}}


def choice_answer(keys, p) -> dict:
    best = keys[int(np.argmax(p))]
    k = len(keys)
    return {"answer": {"type": "choice", "choice": best, "probabilities": dict(zip(keys, map(float, p))),
                       "confidence": float(max(0.0, (max(p) - 1 / k) / (1 - 1 / k)))}}


def too_few(task, y) -> bool:
    """Five-fold CV needs at least five training examples of every class."""
    if len(y) and np.bincount(y).min(initial=len(y)) >= 5 and len(set(y)) > 1:
        return False
    print(f"{task}: fewer than 5 training examples of some class, skipped", flush=True)
    return True


def binary_xgb(task, x_tr, y_tr, x_te, uids, log):
    if too_few(task, np.asarray(y_tr, dtype=int)):
        return
    params, n, cv = tune_xgb(x_tr, y_tr)
    p = fit_xgb(x_tr, y_tr, params, n).predict_proba(x_te)[:, 1]
    write("xgboost", task, {u: noul(pi) for u, pi in zip(uids, p)})
    log.setdefault(task, {})["xgboost"] = {"params": params, "n_estimators": n, "cv_auroc": cv}
    print(f"{task}: xgboost {params} x{n} trees, CV AUROC {cv:.3f}", flush=True)


def tfidf_task(task, log):
    train = read_examples(DATA_DIR / "prepared_train" / f"{task}.jsonl")
    test = read_examples(PREPARED_DIR / f"{task}.jsonl")
    head = next(iter(test[0].questions))
    multiclass = test[0].questions[head]["type"] == "choice"
    keys = list(test[0].questions[head]["criteria"]) if multiclass else None
    y = np.array([keys.index(e.gold[head]) if multiclass else e.gold[head] for e in train], dtype=int)
    if not multiclass and too_few(task, y):
        return
    best = None
    for ng in ((1, 1), (1, 2)):
        vec = tfidf(ng).fit([text_of(e) for e in train])
        c, cv = tune_lr(vec.transform([text_of(e) for e in train]), y, multiclass)
        if best is None or cv > best[2]:
            best = (ng, c, cv, vec)
    ng, c, cv, vec = best
    m = LogisticRegression(C=c, max_iter=3000).fit(vec.transform([text_of(e) for e in train]), y)
    p = m.predict_proba(vec.transform([text_of(e) for e in test]))
    if multiclass:
        full = np.zeros((len(test), len(keys)))
        full[:, m.classes_] = p
        write("tfidf-lr", task, {e.uid: choice_answer(keys, pi) for e, pi in zip(test, full)})
    else:
        write("tfidf-lr", task, {e.uid: noul(pi[1]) for e, pi in zip(test, p)})
    log.setdefault(task, {})["tfidf-lr"] = {"ngram": list(ng), "C": c, "cv": cv}
    print(f"{task}: tfidf-lr ngram {ng} C={c} CV {cv:.3f}", flush=True)


def meds_task(task, log):
    """Evaluation and training samples follow the prepared Jev files (hash-ordered, so a smaller set is a prefix)."""
    from medjev.baselines.xgboost_meds import matrices
    want = {e.uid for e in read_examples(PREPARED_DIR / f"{task}.jsonl")}
    train_file = DATA_DIR / "prepared_train" / f"{task}.jsonl"
    n_train = sum(1 for _ in open(train_file)) if train_file.is_file() else None
    path = CACHE / f"{task}.npz"
    z = np.load(path, allow_pickle=True) if path.is_file() else None
    if z is not None and set(z["uids"].tolist()) == want and len(z["y_tr"]) == (n_train or len(z["y_tr"])):
        x_tr, y_tr, x_te, uids = z["x_tr"], z["y_tr"], z["x_te"], z["uids"].tolist()
    else:
        x_tr, y_tr, x_te, y_te, uids, _ = matrices(task, eval_limit=len(want), max_train=n_train)
        np.savez(path, x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te, uids=np.array(uids, dtype=object))
    assert set(uids) == want, "eval sample mismatch"
    binary_xgb(task, x_tr, y_tr, x_te, uids, log)


def triage_tasks(log, tasks):
    x_tr, x_te, tr, te_adm, te_dis, classes = triage_matrices()
    if "ed_triage_admission" in tasks and te_adm:
        y_adm = (tr["disposition"] == "ADMITTED").to_numpy().astype(int)
        np.savez(CACHE / "ed_triage_admission.npz", x_tr=x_tr, y_tr=y_adm, x_te=x_te,
                 y_te=np.array([e.gold["label"] for e in te_adm], dtype=int), uids=np.array([e.uid for e in te_adm], dtype=object))
        binary_xgb("ed_triage_admission", x_tr, y_adm, x_te, [e.uid for e in te_adm], log)
    if "ed_triage_disposition" in tasks and te_dis:
        # classes with < 10 training stays (e.g. expired) are folded into "other" for training; they get probability 0
        cat = tr["category"].where(tr["category"].map(tr["category"].value_counts()) >= 10, "other")
        y = cat.map(classes.index).to_numpy()
        present = sorted(set(y))
        remap = {c: i for i, c in enumerate(present)}
        yy = np.array([remap[c] for c in y])
        params, n, cv = tune_xgb(x_tr, yy, multiclass=True, n_classes=len(present))
        p = fit_xgb(x_tr, yy, params, n, multiclass=True, n_classes=len(present)).predict_proba(x_te)
        full = np.zeros((len(te_dis), len(classes)))
        for i, c in enumerate(present):
            full[:, c] = p[:, i]
        write("xgboost", "ed_triage_disposition", {e.uid: choice_answer(classes, pi) for e, pi in zip(te_dis, full)})
        log.setdefault("ed_triage_disposition", {})["xgboost"] = {"params": params, "n_estimators": n, "cv_neg_log_loss": cv}
        print(f"ed_triage_disposition: xgboost {params} x{n}, CV log-loss {-cv:.3f}", flush=True)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*", default=TRIAGE + MEDS, help="EHR tasks")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "baselines.json"
    log = json.loads(log_path.read_text()) if log_path.is_file() else {}
    if any(t in TRIAGE for t in args.tasks):
        triage_tasks(log, args.tasks)
    for t in args.tasks:
        if t in MEDS:
            meds_task(t, log)
        if t in MEDS + TRIAGE:
            tfidf_task(t, log)
    log_path.write_text(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()
