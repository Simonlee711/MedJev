#!/usr/bin/env python
"""Step 2: send prepared requests to the MedJev server -> runs/<model>/<task>.jsonl (resumable).

Start the server first: src/scripts/serve_medjev.sh

    python src/scripts/run_jev.py                 # every prepared task
    python src/scripts/run_jev.py medqa --limit 20
"""

import argparse

from medjev.config import JEV_MODEL_NAME, JEV_URL, PREPARED_DIR, RUNS_DIR
from medjev.jev import JevClient, read_examples, run_examples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*", help="tasks (default: every file in data/prepared/)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N prepared examples")
    ap.add_argument("--url", default=JEV_URL)
    ap.add_argument("--model-name", default=JEV_MODEL_NAME, help="name of the runs/ subdirectory")
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()

    tasks = args.tasks or sorted(p.stem for p in PREPARED_DIR.glob("*.jsonl"))
    if not tasks:
        raise SystemExit("nothing prepared; run src/scripts/prepare_data.py first")
    client = JevClient(args.url)
    try:
        print("server:", client.version())
    except Exception as e:
        raise SystemExit(f"cannot reach {args.url} ({e}); start it with src/scripts/serve_medjev.sh")

    for task in tasks:
        examples = read_examples(PREPARED_DIR / f"{task}.jsonl")[: args.limit]
        new = run_examples(client, examples, RUNS_DIR / args.model_name / f"{task}.jsonl", args.concurrency)
        print(f"{task}: {new} new responses ({len(examples)} examples)")


if __name__ == "__main__":
    main()
