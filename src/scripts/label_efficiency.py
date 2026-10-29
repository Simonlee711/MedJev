#!/usr/bin/env python
"""How many local labels does it take (i) to recalibrate zero-shot MedJev and (ii) for a supervised model to reach its
zero-shot ranking? Computed from cached answers and feature matrices only (no model queries). CPU only.

For every binary EHR task and label budget n in {25, 50, 100, 200, 500, 1000, all}, draw 10 random subsets of the
prepared training examples (each with at least two examples of either class; one fit on the full set) and fit
  platt     Platt scaling of the zero-shot logit                 -> test ECE, Brier
  xgboost   XGBoost with the task's cross-validated settings      -> test AUROC, ECE
  tfidf-lr  TF-IDF + logistic regression (task's CV settings)     -> test AUROC, ECE
Zero-shot MedJev (no labels) is the reference. -> results/label_efficiency.json
"""

import json
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from medjev.baselines.supervised import fit_xgb, text_of, tfidf
from medjev.config import DATA_DIR, PREPARED_DIR, RESULTS_DIR, RUNS_DIR, TRAIN_RUNS_DIR
from medjev.jev import read_examples, read_responses
from medjev.metrics import ece

JEV = "medjev"
TASKS = ["ed_triage_admission", "meds_ed_disposition", "meds_mortality", "meds_long_los", "meds_readmission_30d"]
BUDGETS = [25, 50, 100, 200, 500, 1000]  # plus the full training set
REPEATS = 10


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    warnings.filterwarnings("ignore")
    settings = json.loads((RESULTS_DIR / "baselines.json").read_text())
    out = {}
    for task in TASKS:
        needed = [DATA_DIR / "cache" / "baselines" / f"{task}.npz", TRAIN_RUNS_DIR / JEV / f"{task}.jsonl"]
        if not all(p.is_file() for p in needed) or task not in settings:
            print(f"{task}: baselines or training answers missing, skipped")
            continue
        tr = read_examples(DATA_DIR / "prepared_train" / f"{task}.jsonl")
        te = read_examples(PREPARED_DIR / f"{task}.jsonl")
        zs_tr = read_responses(TRAIN_RUNS_DIR / JEV / f"{task}.jsonl")
        zs_te = read_responses(RUNS_DIR / JEV / f"{task}.jsonl")
        y_tr = np.array([e.gold["label"] for e in tr], dtype=int)
        y_te = np.array([e.gold["label"] for e in te], dtype=int)
        l_tr = logit([zs_tr[e.uid]["answers"]["label"]["noul"] for e in tr])
        p_zs = np.array([zs_te[e.uid]["answers"]["label"]["noul"] for e in te])
        l_te = logit(p_zs)
        # XGBoost matrices (cached by run_baselines.py) in the same example order as the prepared files
        z = np.load(DATA_DIR / "cache" / "baselines" / f"{task}.npz", allow_pickle=True)
        te_pos = {u: i for i, u in enumerate(z["uids"].tolist())}
        x_te = z["x_te"][[te_pos[e.uid] for e in te]]
        x_tr, yx_tr = z["x_tr"], z["y_tr"]
        assert len(x_tr) == len(tr) and (np.sort(yx_tr) == np.sort(y_tr)).all()
        xs = settings[task]["xgboost"]
        ts = settings[task]["tfidf-lr"]
        vec = tfidf(tuple(ts["ngram"])).fit([text_of(e) for e in tr])
        t_tr, t_te = vec.transform([text_of(e) for e in tr]), vec.transform([text_of(e) for e in te])
        res = {"zero_shot": {"auroc": float(roc_auc_score(y_te, p_zs)), "ece": ece(p_zs, y_te.astype(float))}, "budgets": {}}
        rng = np.random.default_rng(0)
        for n in [b for b in BUDGETS if b < len(tr)] + [len(tr)]:
            runs = {"platt": [], "xgboost": [], "tfidf-lr": []}
            for _ in range(REPEATS if n < len(tr) else 1):
                for _try in range(200):
                    idx = rng.choice(len(tr), size=min(n, len(tr)), replace=False)
                    if 2 <= y_tr[idx].sum() <= len(idx) - 2 and 2 <= yx_tr[idx].sum() <= len(idx) - 2:
                        break
                else:
                    continue
                pl = LogisticRegression(C=1e6, max_iter=1000).fit(l_tr[idx][:, None], y_tr[idx])
                p = pl.predict_proba(l_te[:, None])[:, 1]
                runs["platt"].append({"ece": ece(p, y_te.astype(float)), "brier": float(np.mean((p - y_te) ** 2))})
                p = fit_xgb(x_tr[idx], yx_tr[idx], xs["params"], xs["n_estimators"]).predict_proba(x_te)[:, 1]
                runs["xgboost"].append({"auroc": float(roc_auc_score(y_te, p)), "ece": ece(p, y_te.astype(float))})
                p = LogisticRegression(C=ts["C"], max_iter=3000).fit(t_tr[idx], y_tr[idx]).predict_proba(t_te)[:, 1]
                runs["tfidf-lr"].append({"auroc": float(roc_auc_score(y_te, p)), "ece": ece(p, y_te.astype(float))})
            res["budgets"][str(n)] = {m: {k: {"mean": float(np.mean([r[k] for r in v])), "sd": float(np.std([r[k] for r in v]))}
                                          for k in v[0]} for m, v in runs.items() if v}
            b = res["budgets"][str(n)]
            print(f"{task} n={n}: platt ECE {b['platt']['ece']['mean']:.3f} | xgb AUROC {b['xgboost']['auroc']['mean']:.3f} "
                  f"| tfidf AUROC {b['tfidf-lr']['auroc']['mean']:.3f} | zero-shot AUROC {res['zero_shot']['auroc']:.3f}", flush=True)
        out[task] = res
    (RESULTS_DIR / "label_efficiency.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
