#!/usr/bin/env python
"""Step 4: score every model's responses against the gold labels.

Writes per-example scores to scores/<model>/<task>.csv and metrics to results/<task>.json + results/summary.md.

    python src/scripts/evaluate.py
"""

import argparse
import csv
import json
import math
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

from medjev.config import PREPARED_DIR, RESULTS_DIR, RUNS_DIR, SCORES_DIR
from medjev.jev import read_examples, read_responses
from medjev.metrics import binary_metrics, bootstrap_ci, choice_metrics

MEAN_KEYS = ("auroc", "auprc", "brier", "ece", "positive_rate", "mean_prob", "balanced_accuracy@0.5",
             "predicted_positive_rate@0.5")


def mean_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Mean AUROC over the columns (questions) that have both classes."""
    vals = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(vals)) if vals else math.nan


def write_scores(path, examples, responses) -> None:
    """One row per example and question: gold, prediction, and the model's probabilities."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "question", "gold", "prediction", "probability", "probabilities"])
        for e in examples:
            for head, q in e.questions.items():
                a = responses[e.uid]["answers"][head]
                if q["type"] == "noul":
                    w.writerow([e.uid, head, e.gold[head], a["noul"] >= 0.5, a["noul"], ""])
                elif q["type"] == "choice":
                    w.writerow([e.uid, head, e.gold[head], a["choice"], a["probabilities"][a["choice"]],
                                json.dumps(a["probabilities"])])
                else:
                    w.writerow([e.uid, head, e.gold[head], a.get("score"), "", json.dumps(a.get("probabilities"))])


def evaluate_task(task: str) -> dict:
    examples = read_examples(PREPARED_DIR / f"{task}.jsonl")
    out = {}
    for model_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        responses = read_responses(model_dir / f"{task}.jsonl")
        if not responses:
            continue
        have = [e for e in examples if e.uid in responses]
        write_scores(SCORES_DIR / model_dir.name / f"{task}.csv", have, responses)
        per_head = {}
        for head, q in examples[0].questions.items():
            golds = [e.gold[head] for e in have]
            answers = [responses[e.uid]["answers"][head] for e in have]
            if q["type"] == "choice":
                per_head[head] = choice_metrics(golds, answers)
            elif q["type"] == "noul":
                per_head[head] = binary_metrics(golds, [a["noul"] for a in answers])
        noul_heads = [h for h, q in examples[0].questions.items() if q["type"] == "noul"]
        if len(noul_heads) > 1:  # fan-out task: average over questions, bootstrap over examples
            agg = {k: float(np.nanmean([per_head[h][k] for h in noul_heads])) for k in MEAN_KEYS}
            y = np.array([[e.gold[h] for h in noul_heads] for e in have], dtype=int)
            pr = np.array([[responses[e.uid]["answers"][h]["noul"] for h in noul_heads] for e in have])
            agg["auroc_ci95"] = list(bootstrap_ci(mean_auroc, y, pr, n=500))
            agg["n"] = len(have)
            per_head["mean"] = agg
        out[model_dir.name] = {"coverage": f"{len(have)}/{len(examples)}", "heads": per_head}
    return out


def fmt(x: float, ci=None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    s = f"{x:.3f}"
    return s + (f" ({ci[0]:.3f}–{ci[1]:.3f})" if ci and not math.isnan(ci[0]) else "")


def summary_md(results: dict) -> str:
    lines = ["# Open MedJev results", "", "Zero-shot MedJev vs. supervised baselines. 95% bootstrap CIs in parentheses.", ""]
    mc = {t: r for t, r in results.items() if any("accuracy" in m["heads"].get("answer", {}) for m in r.values())}
    if mc:
        lines += ["## Multiple choice", "", "| task | model | n | accuracy | chance | ECE |", "|---|---|---|---|---|---|"]
        for task, r in mc.items():
            for model, m in r.items():
                h = m["heads"]["answer"]
                lines.append(f"| {task} | {model} | {m['coverage']} | {fmt(h['accuracy'], h['accuracy_ci95'])} | "
                             f"{fmt(h['chance'])} | {fmt(h['ece'])} |")
        lines.append("")
    binary = {t: r for t, r in results.items() if t not in mc}
    if binary:
        lines += ["## Binary prediction (EHR; fan-out tasks show the mean over questions)", "",
                  "| task | model | n | positive rate | AUROC | AUPRC | ECE | Brier |",
                  "|---|---|---|---|---|---|---|---|"]
        for task, r in binary.items():
            for model, m in r.items():
                h = m["heads"].get("label") or m["heads"].get("mean")
                lines.append(f"| {task} | {model} | {m['coverage']} | {fmt(h['positive_rate'])} | "
                             f"{fmt(h['auroc'], h.get('auroc_ci95'))} | {fmt(h['auprc'], h.get('auprc_ci95'))} | "
                             f"{fmt(h['ece'])} | {fmt(h['brier'])} |")
        lines.append("")
    return "\n".join(lines)


def main():
    warnings.filterwarnings("ignore", module="sklearn")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*", help="tasks (default: every file in data/prepared/)")
    args = ap.parse_args()
    tasks = args.tasks or sorted(p.stem for p in PREPARED_DIR.glob("*.jsonl"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for task in tasks:
        results[task] = evaluate_task(task)
        (RESULTS_DIR / f"{task}.json").write_text(json.dumps(results[task], indent=1))
    md = summary_md(results)
    (RESULTS_DIR / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
