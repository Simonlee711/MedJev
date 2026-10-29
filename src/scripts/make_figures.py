#!/usr/bin/env python
"""Figures from results/ and scores/ -> results/figures/*.pdf (+ 300 dpi .png)

    python src/scripts/make_figures.py

Variants that have not been run yet are left out, so the figures can be rebuilt at any point.
"""

import csv
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from medjev.config import RESULTS_DIR, ROOT, SCORES_DIR
from medjev.plotting import (BLUE, GREY_LIGHT, GREY_MID, GREY_TEXT, INK, MODEL_STYLE, TASK_NAMES, TEXT_WIDTH_IN, save,
                             setup)

OUT = RESULTS_DIR / "figures"
MC_TASKS = {t: TASK_NAMES[t] for t in ("medqa", "usmle4", "nejm_medqa", "diagnosisarena", "ed_triage_disposition")}
BIN_TASKS = {t: TASK_NAMES[t] for t in ("ed_triage_admission", "meds_ed_disposition", "meds_mortality", "meds_long_los",
                                         "meds_readmission_30d")}
MC_MODELS = ["medjev", "qwen3.5-9b-base", "medgemma-1.5-4b", "gemma-3-4b", "qwen3.5-2b", "pythia-2.8b", "xgboost",
             "tfidf-lr"]
BIN_MODELS = ["medjev", "qwen3.5-9b-base", "qwen3.5-2b", "pythia-2.8b", "xgboost", "tfidf-lr"]
CAL_MODELS = ["medjev", "medjev-recal", "qwen3.5-9b-base", "xgboost"]


def results(task: str) -> dict:
    p = RESULTS_DIR / f"{task}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def complete(m: dict) -> bool:
    a, b = m.get("coverage", "0/1").split("/")
    return a == b


def head_of(task: str, m: dict) -> dict | None:
    if not m or not complete(m):
        return None
    h = m["heads"]
    return h.get("answer") or h.get("label") or h.get("mean")


def read_scores(model: str, task: str) -> list[dict]:
    p = SCORES_DIR / model / f"{task}.csv"
    return list(csv.DictReader(p.open())) if p.is_file() else []


def mk(ax, x, y, model, ms=6.5, label=True):
    st = MODEL_STYLE[model]
    ax.plot(x, y, st["marker"], ms=ms, color=st["color"], mfc="white" if st.get("hollow") else st["color"],
            mec=st["color"] if st.get("hollow") else "white", mew=1.4 if st.get("hollow") else 1.0, zorder=3,
            label=st["short"] if label else None)


def row_labels(ax, ys, titles, subtitles):
    ax.set_yticks(ys)
    ax.set_yticklabels([""] * len(ys))
    for y, t, s in zip(ys, titles, subtitles):
        ax.annotate(t, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(-8, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=9, color=INK)
        ax.annotate(s, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(-8, 1), textcoords="offset points",
                    ha="right", va="top", fontsize=7.5, color=GREY_TEXT)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def legend_top(fig, axes, ncol=4, y=1.02, fixed_ncol=None):
    seen = {}
    for ax in np.atleast_1d(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    order = [st["short"] for st in MODEL_STYLE.values()] + [f"{m} (reported)" for m in ("GPT-4", "GPT-3.5", "GPT-4o", "o1")]
    labels = sorted(seen, key=lambda l: order.index(l) if l in order else len(order))
    ncol = fixed_ncol or (len(labels) if len(labels) <= 6 else (5 if len(labels) <= 10 else ncol))
    if fixed_ncol:  # matplotlib fills columns first; reorder so the legend reads row by row
        rows = -(-len(labels) // ncol)
        grid = [labels[i:i + ncol] for i in range(0, len(labels), ncol)]
        labels = [g[c] for c in range(ncol) for g in grid if c < len(g)]
    fig.legend([seen[l] for l in labels], labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.55, y),
               handlelength=1.2, columnspacing=1.4)


def forest(tasks: dict, models: list[str], metric: str, ci_key: str, xlabel: str, path, xlim, chance=None, refs=None,
           legend_ncol=None):
    rows = [t for t in tasks if any(head_of(t, results(t).get(m)) for m in models)]
    if not rows:
        return
    present = [m for m in models if any(head_of(t, results(t).get(m)) for t in rows)]
    n_slots = len(present) + (1 if refs else 0)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 0.4 + len(rows) * (0.25 + 0.085 * n_slots)))
    ys = np.arange(len(rows))[::-1].astype(float)
    offs = np.linspace(0.32, -0.32, n_slots) if n_slots > 1 else [0.0]
    subs = []
    for y, task in zip(ys, rows):
        r = results(task)
        n = None
        for j, m in enumerate(present):
            h = head_of(task, r.get(m))
            if not h:
                continue
            n = h.get("n", n)
            yy = y + offs[j]
            ci = h.get(ci_key)
            if ci and not any(isinstance(v, float) and math.isnan(v) for v in ci):
                ax.plot(ci, [yy, yy], color=MODEL_STYLE[m]["color"], lw=1.6, alpha=0.8, solid_capstyle="round", zorder=2)
            mk(ax, h[metric], yy, m, ms=5.5, label=(task == rows[0]) or True)
        if refs and task in refs:
            for name, val, (marker, color, filled) in refs[task]:
                ax.plot(val, y + offs[-1], marker, ms=7 if marker == "*" else 6, color=color,
                        mfc=color if filled else "white", mec=color, mew=1.2, zorder=4, label=name)
        if chance and task in chance:
            ax.plot([chance[task]] * 2, [y - 0.42, y + 0.42], color=GREY_MID, lw=1.3, zorder=1)
        subs.append("")
    ax.set_xlim(*xlim)
    ax.set_ylim(ys[-1] - 0.5, ys[0] + 0.5)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GREY_LIGHT, lw=0.8)
    ax.set_axisbelow(True)
    row_labels(ax, ys, [tasks[t] for t in rows], subs)
    fig.tight_layout()
    legend_top(fig, ax, ncol=4, y=0.99, fixed_ncol=legend_ncol)
    save(fig, OUT / path)


def fig_mc_accuracy():
    chance = {}
    for t in MC_TASKS:
        for m in results(t).values():
            h = head_of(t, m)
            if h:
                chance[t] = h["chance"]
                break
    refs = {}
    ref_path = RESULTS_DIR / "published_reference.json"
    markers = {"GPT-4": ("P", INK, True), "GPT-3.5": ("D", INK, False), "GPT-4o": ("*", GREY_TEXT, True), "o1": ("X", GREY_TEXT, True)}
    if ref_path.is_file():
        for task, rows in json.loads(ref_path.read_text()).items():
            refs[task] = [(f"{r['model']} (reported)", r["accuracy"], markers[r["model"]]) for r in rows]
    forest(MC_TASKS, MC_MODELS, "accuracy", "accuracy_ci95", "Top-1 accuracy (95% bootstrap CI); grey bar: chance",
           "fig_mc_accuracy.pdf", (0, 1), chance=chance, refs=refs)


def fig_binary_auroc():
    forest(BIN_TASKS, BIN_MODELS, "auroc", "auroc_ci95", "AUROC (95% bootstrap CI); grey bar: chance",
           "fig_binary_auroc.pdf", (0.35, 1.0), chance={t: 0.5 for t in BIN_TASKS}, legend_ncol=3)


def fig_ece():
    """ECE per task: zero-shot MedJev, recalibrated, base model and XGBoost."""
    tasks = {**{t: v for t, v in MC_TASKS.items()}, **BIN_TASKS}
    rows = [t for t in tasks if head_of(t, results(t).get("medjev"))]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 0.36 * len(rows) + 0.9))
    ys = np.arange(len(rows))[::-1].astype(float)
    for y, t in zip(ys, rows):
        r = results(t)
        vals = [(m, head_of(t, r.get(m))) for m in CAL_MODELS]
        vals = [(m, h["ece"]) for m, h in vals if h]
        if len(vals) > 1:
            ax.plot([min(v for _, v in vals), max(v for _, v in vals)], [y, y], color=GREY_LIGHT, lw=2.2, zorder=1)
        for m, v in vals:
            mk(ax, v, y, m, ms=6)
    ax.set_xlim(0, None)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    ax.set_xlabel("Expected calibration error (lower is better)")
    ax.grid(axis="x", color=GREY_LIGHT, lw=0.8)
    ax.set_axisbelow(True)
    row_labels(ax, ys, [tasks[t] for t in rows], [""] * len(rows))
    fig.tight_layout()
    legend_top(fig, ax, ncol=4, y=0.99)
    save(fig, OUT / "fig_ece.pdf")


def _bins(conf, outcome, n_bins=10):
    order = np.argsort(conf, kind="stable")
    return [(conf[g].mean(), outcome[g].mean()) for g in np.array_split(order, n_bins) if len(g)]


def fig_calibration_curves():
    panels = [(t, TASK_NAMES[t]) for t in ("medqa", "nejm_medqa", "diagnosisarena", "ed_triage_admission",
                                             "meds_mortality", "meds_readmission_30d")]
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 0.72))
    for ax, (task, title) in zip(axes.flat, panels):
        ax.plot([0, 1], [0, 1], color=GREY_MID, lw=1.0, ls=(0, (3, 3)), zorder=1)
        for m in CAL_MODELS:
            rows = read_scores(m, task)
            if not rows:
                continue
            p = np.array([float(r["probability"]) for r in rows])
            if task in MC_TASKS:
                y = np.array([r["gold"] == r["prediction"] for r in rows], dtype=float)
            else:
                y = np.array([r["gold"] == "True" for r in rows], dtype=float)
            pts = _bins(p, y)
            st = MODEL_STYLE[m]
            ax.plot([q[0] for q in pts], [q[1] for q in pts], ls="-", lw=1.3, color=st["color"], marker=st["marker"],
                    ms=4, mfc="white" if st.get("hollow") else st["color"], mec=st["color"], zorder=3, label=st["short"])
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.grid(color=GREY_LIGHT, lw=0.7)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("Mean predicted probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed frequency")
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    legend_top(fig, axes.flat, ncol=4, y=0.99)
    save(fig, OUT / "fig_calibration_curves.pdf")


def fig_label_efficiency():
    """ECE after Platt scaling, and supervised AUROC, against the number of local labels (from cached runs)."""
    p = RESULTS_DIR / "label_efficiency.json"
    if not p.is_file():
        return
    le = json.loads(p.read_text())
    tasks = [t for t in BIN_TASKS if t in le]
    if not tasks:
        return
    fig, axes = plt.subplots(2, len(tasks), figsize=(TEXT_WIDTH_IN, 3.6), sharex=True, squeeze=False)
    for j, t in enumerate(tasks):
        r = le[t]
        ns = [int(n) for n in r["budgets"]]
        for row, (metric, models, ylabel) in enumerate([("ece", ["platt"], "ECE"), ("auroc", ["xgboost", "tfidf-lr"], "AUROC")]):
            ax = axes[row, j]
            ax.axhline(r["zero_shot"][metric], color=BLUE, lw=1.2, ls=(0, (4, 2)), label="MedJev zero-shot")
            for m in models:
                key = {"platt": "medjev-recal", "xgboost": "xgboost", "tfidf-lr": "tfidf-lr"}[m]
                st = MODEL_STYLE[key]
                mean = np.array([r["budgets"][str(n)][m][metric]["mean"] for n in ns])
                sd = np.array([r["budgets"][str(n)][m][metric]["sd"] for n in ns])
                ax.fill_between(ns, mean - sd, mean + sd, color=st["color"], alpha=0.15, lw=0)
                ax.plot(ns, mean, "-", marker=st["marker"], ms=3.5, lw=1.4, color=st["color"],
                        mfc="white" if st.get("hollow") else st["color"], label=st["short"])
            ax.set_xscale("log")
            ax.set_xticks([25, 100, 1000])
            ax.set_xticklabels(["25", "100", "1k"], fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(color=GREY_LIGHT, lw=0.6)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(BIN_TASKS[t], fontsize=7.5, color=INK, pad=3, loc="left")
            if row == 1:
                ax.set_xlabel("Labels", fontsize=8)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    legend_top(fig, axes.flat, ncol=4, y=0.99)
    save(fig, OUT / "fig_label_efficiency.pdf")


def fig_selective():
    """Accuracy among the questions answered, as the most confident ones are kept (coverage curves)."""
    panels = [t for t in MC_TASKS if read_scores("medjev", t)]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH_IN, 2.2), sharey=False)
    for ax, t in zip(np.atleast_1d(axes), panels):
        for m in ["medjev", "qwen3.5-9b-base", "medgemma-1.5-4b", "gemma-3-4b", "qwen3.5-2b", "pythia-2.8b", "xgboost",
                  "tfidf-lr"]:
            rows = read_scores(m, t)
            if not rows:
                continue
            p = np.array([float(r["probability"]) for r in rows])
            c = np.array([r["gold"] == r["prediction"] for r in rows], dtype=float)
            order = np.argsort(-p, kind="stable")
            acc = np.cumsum(c[order]) / np.arange(1, len(c) + 1)
            cov = np.arange(1, len(c) + 1) / len(c)
            st = MODEL_STYLE[m]
            start = max(1, len(c) // 20)
            ax.plot(cov[start:], acc[start:], color=st["color"], lw=1.4, ls="--" if st.get("hollow") else "-", label=st["short"])
        ax.set_title(MC_TASKS[t], loc="left", fontsize=8, color=INK, pad=3)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.grid(color=GREY_LIGHT, lw=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("Coverage")
    np.atleast_1d(axes)[0].set_ylabel("Accuracy of answered")
    fig.tight_layout(w_pad=0.6)
    legend_top(fig, axes, ncol=6, y=0.98, fixed_ncol=3)
    save(fig, OUT / "fig_selective.pdf")


EXPECTED = {"fig_mc_accuracy": "accuracy on the multiple-choice tasks", "fig_binary_auroc": "AUROC on the binary tasks",
            "fig_ece": "calibration error per variant", "fig_calibration_curves": "calibration curves",
            
            "fig_label_efficiency": "label efficiency", "fig_selective": "selective accuracy"}


def placeholder(name: str, what: str) -> None:
    """A labelled stand-in so documents that include the figures build before a run has finished."""
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 1.2))
    ax.axis("off")
    ax.text(0.5, 0.5, f"Pending: {what} (run in progress)", ha="center", va="center", fontsize=10, color=GREY_TEXT,
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=GREY_LIGHT))
    save(fig, OUT / f"{name}.pdf")


def main():
    setup()
    for name in EXPECTED:
        for ext in (".pdf", ".png"):
            (OUT / f"{name}{ext}").unlink(missing_ok=True)
    for f in (fig_mc_accuracy, fig_binary_auroc, fig_ece, fig_calibration_curves, fig_label_efficiency, fig_selective):
        f()
    for name, what in EXPECTED.items():
        if not (OUT / f"{name}.pdf").is_file():
            placeholder(name, what)
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
