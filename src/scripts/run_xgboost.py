#!/usr/bin/env python
"""Step 3: XGBoost baseline for the MEDS binary tasks -> runs/xgboost/<task>.jsonl (Jev answer format).

Evaluates on exactly the held-out examples in data/prepared/<task>.jsonl.

    python src/scripts/run_xgboost.py                                  # all MEDS tasks
    python src/scripts/run_xgboost.py meds_mortality --train-shards 2
"""

import argparse
import json
import time

from medjev.baselines.xgboost_meds import train_and_predict
from medjev.config import PREPARED_DIR, RESULTS_DIR, RUNS_DIR
from medjev.jev import read_examples
from medjev.tasks import meds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*", default=list(meds.TASKS))
    ap.add_argument("--train-shards", type=int, default=None, help="MEDS train shards to use (default: all 5)")
    ap.add_argument("--max-train", type=int, default=None, help="optional cap on training examples (default: all)")
    args = ap.parse_args()

    for task in args.tasks:
        prepared = PREPARED_DIR / f"{task}.jsonl"
        uids = [e.uid for e in read_examples(prepared)] if prepared.is_file() else None
        t0 = time.time()
        eval_s, probs, info = train_and_predict(task, eval_limit=len(uids) if uids else None,
                                                train_shards=args.train_shards, max_train=args.max_train)
        pred = dict(zip(eval_s["sample_id"].to_list(), probs.tolist()))
        if uids and set(uids) != set(pred):
            raise SystemExit(f"{task}: XGBoost eval sample differs from data/prepared/{task}.jsonl; re-run prepare_data.py")

        out = RUNS_DIR / "xgboost" / f"{task}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for uid, p in pred.items():
                f.write(json.dumps({"uid": uid, "answers": {"label": {"type": "noul", "noul": p}}, "model": "xgboost"}) + "\n")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"xgboost_{task}_training.json").write_text(json.dumps(info, indent=1))
        print(f"{task}: trained on {info['n_train']} admissions ({info['n_features']} features), "
              f"tuning AUROC {info['tune_auroc']:.3f} @ {info['best_iteration']} trees, {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
