"""Metrics over Jev-format answers, ported from notes/jev_benchmarking/metrics.py, plus bootstrap CIs."""

import math

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

N_BINS = 15


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error with equal-width bins."""
    if len(conf) == 0:
        return math.nan
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.sum() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total / len(conf))


def _safe(fn, *args) -> float:
    try:
        return float(fn(*args))
    except ValueError:  # e.g. a single class present
        return math.nan


def bootstrap_ci(fn, *arrays, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of fn(*arrays) over example-level resamples."""
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        v = _safe(fn, *(a[idx] for a in arrays))
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return math.nan, math.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def choice_metrics(golds: list, answers: list[dict]) -> dict:
    preds = [a["choice"] for a in answers]
    correct = np.array([p == g for p, g in zip(preds, golds)], dtype=float)
    top_p = np.array([max(a["probabilities"].values()) for a in answers])
    brier = np.mean([sum((p - (k == g)) ** 2 for k, p in a["probabilities"].items()) for a, g in zip(answers, golds)])
    lo, hi = bootstrap_ci(np.mean, correct)
    return {
        "n": len(golds),
        "accuracy": float(correct.mean()),
        "accuracy_ci95": [lo, hi],
        "chance": float(np.mean([1 / len(a["probabilities"]) for a in answers])),
        "macro_f1": float(f1_score(golds, preds, average="macro", zero_division=0)),
        "ece": ece(top_p, correct),
        "brier": float(brier),
        "mean_top_prob": float(top_p.mean()),
    }


def binary_metrics(golds: list[bool], probs: list[float], threshold: float = 0.5) -> dict:
    y = np.array(golds, dtype=int)
    p = np.array(probs, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / max(1, (y == 1).sum())
    tnr = int(((pred == 0) & (y == 0)).sum()) / max(1, (y == 0).sum())
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "auroc": _safe(roc_auc_score, y, p),
        "auroc_ci95": list(bootstrap_ci(roc_auc_score, y, p)),
        "auprc": _safe(average_precision_score, y, p) if y.any() else math.nan,
        "auprc_ci95": list(bootstrap_ci(average_precision_score, y, p)),
        "accuracy@0.5": float((pred == y).mean()),
        "balanced_accuracy@0.5": float((tpr + tnr) / 2),
        "f1@0.5": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "precision@0.5": float(tp / max(1, tp + fp)),
        "recall@0.5": float(tpr),
        "predicted_positive_rate@0.5": float(pred.mean()),
        "mean_prob": float(p.mean()),
        "ece": ece(p, y.astype(float)),
        "brier": float(np.mean((p - y) ** 2)),
    }
