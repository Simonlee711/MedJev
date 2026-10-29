#!/usr/bin/env python
"""Run MedJev (or a zero-shot LLM baseline) locally with MLX over prepared examples -> Jev answers.

Runs in the mlx environment:
    mlx/bin/python src/scripts/extract_jev.py medqa meds_mortality --set test --no-features
    mlx/bin/python src/scripts/extract_jev.py meds_mortality --set train --no-features        # for recalibration
    mlx/bin/python src/scripts/extract_jev.py medqa --model-dir models/Qwen3.5-9B-8bit --variant qwen3.5-9b-base --no-features
    mlx/bin/python src/scripts/extract_jev.py medqa --model-dir models/pythia-2.8b-8bit --variant pythia-2.8b --plain \
        --max-tokens 2040 --no-features

Outputs
    answers:  runs/<variant>/<task>.jsonl (test) or runs_train/<variant>/<task>.jsonl (train), Jev answer format
    features: data/features/<variant>/<task>.<set>.npz (optional last-token hidden states per question and layer)
Both are a cache keyed on the exact request (state + questions): a rerun skips every example whose request is
unchanged and recomputes the ones whose prompt or state changed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from medjev.config import DATA_DIR, PREPARED_DIR, ROOT, RUNS_DIR, TRAIN_RUNS_DIR
from medjev.jev import read_examples, read_responses, request_key



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--set", choices=["test", "train"], default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model-dir", default=str(ROOT / "MedJev-Flash-9B-MLX"))
    ap.add_argument("--adapter", default=None, help="optional mlx-lm adapter directory")
    ap.add_argument("--variant", default="medjev", help="runs/ subdirectory name")
    ap.add_argument("--no-features", action="store_true", help="do not record hidden states")
    ap.add_argument("--prefix-cache", action="store_true", help="reuse the state prefix across questions of a request")
    ap.add_argument("--plain", action="store_true", help="base model without a chat template (e.g. Pythia)")
    ap.add_argument("--max-tokens", type=int, default=None, help="context window; longer prompts are left-truncated")
    args = ap.parse_args()

    from medjev.engine import Engine

    engine = Engine(args.model_dir, adapter_path=args.adapter, capture=not args.no_features, plain=args.plain,
                    max_tokens=args.max_tokens)
    src_dir = PREPARED_DIR if args.set == "test" else DATA_DIR / "prepared_train"
    runs = (RUNS_DIR if args.set == "test" else TRAIN_RUNS_DIR) / args.variant
    feat_dir = DATA_DIR / "features" / args.variant
    runs.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        examples = read_examples(src_dir / f"{task}.jsonl")[: args.limit]
        engine.prefix_cache = args.prefix_cache
        out_path = runs / f"{task}.jsonl"
        keys = {e.uid: request_key(e.state, e.questions) for e in examples}
        done = {u: r for u, r in read_responses(out_path).items() if r.get("key") in (None, keys.get(u))}
        stale = [u for u, r in read_responses(out_path).items() if u in keys and r.get("key") not in (None, keys[u])]
        if stale:  # the request changed: drop the cached answers so they are recomputed
            kept = [l for l in out_path.read_text().splitlines() if l.strip() and json.loads(l)["uid"] not in set(stale)]
            out_path.write_text("\n".join(kept) + ("\n" if kept else ""))
            print(f"  {task}: {len(stale)} cached answers are stale (request changed) and will be recomputed", flush=True)
        feat_path = feat_dir / f"{task}.{args.set}.npz"
        have_feats, feat_keys = set(), {}
        if not args.no_features and feat_path.is_file():
            z = np.load(feat_path, allow_pickle=True)
            stored = z["uids"].tolist()
            feat_keys = dict(zip(stored, z["keys"].tolist())) if "keys" in z.files else {u: keys.get(u) for u in stored}
            have_feats = {u for u in stored if feat_keys.get(u) == keys.get(u)}
        todo = [e for e in examples if e.uid not in done or (not args.no_features and e.uid not in have_feats)]
        print(f"{task} [{args.set}]: {len(todo)} of {len(examples)} to run (prefix cache: {engine.prefix_cache})", flush=True)
        feats: dict[str, list] = {}
        uids: list[str] = []
        if have_feats:  # resume: keep the stored features whose request is unchanged
            z = np.load(feat_path, allow_pickle=True)
            keep = [i for i, u in enumerate(z["uids"].tolist()) if u in have_feats]
            uids = [z["uids"][i] for i in keep]
            feats = {k: [z[k][i] for i in keep] for k in z.files if k not in ("uids", "keys")}
        t0 = time.time()
        with out_path.open("a") as out:
            for i, e in enumerate(todo):
                answers, hidden = engine.answer(e.state, e.questions)
                if e.uid not in done:
                    out.write(json.dumps({"uid": e.uid, "key": request_key(e.state, e.questions), "answers": answers,
                                          "model": args.variant}, ensure_ascii=False) + "\n")
                    out.flush()
                if not args.no_features and e.uid not in have_feats:
                    uids.append(e.uid)
                    for qid, layers in hidden.items():
                        for layer, vec in layers.items():
                            feats.setdefault(f"{qid}|{layer}", []).append(vec)
                if (i + 1) % 50 == 0:
                    import mlx.core as mx
                    rate = (time.time() - t0) / (i + 1)
                    print(f"  {task}: {i + 1}/{len(todo)}  {rate:.2f}s/example  eta {rate * (len(todo) - i - 1) / 60:.0f} min"
                          f"  (GPU memory {mx.get_active_memory() / 1e9:.1f} GB active, {mx.get_cache_memory() / 1e9:.1f} GB cache)", flush=True)
                if not args.no_features and (i + 1) % 250 == 0:
                    _save(feat_path, uids, feats, keys)
        if not args.no_features and uids:
            _save(feat_path, uids, feats, keys)
        print(f"{task} [{args.set}]: done in {(time.time() - t0) / 60:.1f} min", flush=True)


def _save(path: Path, uids: list[str], feats: dict[str, list], keys: dict[str, str]) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, uids=np.array(uids, dtype=object), keys=np.array([keys[u] for u in uids], dtype=object),
             **{k: np.stack(v) for k, v in feats.items()})
    tmp.replace(path)


if __name__ == "__main__":
    main()
