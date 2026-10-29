#!/usr/bin/env python
"""Step 1: turn datasets into Jev requests with gold labels -> data/prepared/<task>.jsonl

    python src/scripts/prepare_data.py                       # all tasks, full test splits
    python src/scripts/prepare_data.py medqa --limit 200 --show 1
    python src/scripts/prepare_data.py --set train           # training sets -> data/prepared_train/
"""

import argparse
import json
from collections import Counter

from medjev.config import DATA_DIR, PREPARED_DIR
from medjev.jev import write_examples
from medjev.tasks import BUILDERS, TRAIN_BUILDERS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*", default=list(BUILDERS), help=f"tasks (default: all): {', '.join(BUILDERS)}")
    ap.add_argument("--limit", type=int, default=None, help="optional cap on examples per task (default: full split)")
    ap.add_argument("--split", default=None, help="dataset split (default: medqa=test, meds_*=held_out)")
    ap.add_argument("--show", type=int, default=0, help="print this many example requests")
    ap.add_argument("--set", choices=["test", "train"], default="test", help="evaluation or training examples")
    args = ap.parse_args()
    limit = args.limit
    builders = BUILDERS if args.set == "test" else TRAIN_BUILDERS
    out_dir = PREPARED_DIR if args.set == "test" else DATA_DIR / "prepared_train"

    for task in args.tasks if args.tasks != list(BUILDERS) else list(builders):
        if task not in builders:
            raise SystemExit(f"unknown task {task!r}; known: {', '.join(builders)}")
        if args.set == "train":
            examples = builders[task](limit)
        else:
            split = args.split or ("test" if task == "medqa" else None)
            examples = builders[task](split, limit)
        path = out_dir / f"{task}.jsonl"
        write_examples(path, examples)
        golds = Counter(str(g) for e in examples for g in e.gold.values())
        chars = sorted(len(json.dumps(e.state, ensure_ascii=False)) for e in examples)
        print(f"{task}: {len(examples)} examples -> {path}  gold={dict(golds)}  "
              f"state chars median={chars[len(chars) // 2]} max={chars[-1]}")
        for e in examples[: args.show]:
            print(json.dumps({"state": e.state, "questions": e.questions, "gold": e.gold}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
