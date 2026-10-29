"""The Jev request format and a small resumable client for a Jev-compatible endpoint.

A Jev request is one `state` (text or JSON) plus a map of typed questions:

- ``choice``: pick one option key from ``criteria`` (a map key -> description or None)
- ``noul``:   probability that a yes/no statement is true (``criteria`` = optional true/false descriptions)
- ``score``:  expected level on an ordered scale (``criteria`` = list of level descriptions)

The response carries one answer per question id. Request construction mirrors
notes/jev_benchmarking/tasks/base.py so tasks stay compatible with the benchmarking harness.
"""

import hashlib
import json
import string
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm


@dataclass
class Example:
    task: str
    uid: str  # stable id within the task
    state: Any
    questions: dict[str, dict]
    gold: dict[str, Any]  # question id -> option key (choice) | bool (noul) | number (score)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)

    @staticmethod
    def from_json(line: str) -> "Example":
        return Example(**json.loads(line))


# ----------------------------------------------------------------------------------------------------
# Question constructors


def choice(instructions: Any, criteria: dict[str, Any]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul(instructions: Any, true: str | None = None, false: str | None = None) -> dict:
    q: dict = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {k: v for k, v in (("true", true), ("false", false)) if v}
    return q


def score(instructions: Any, levels: list[Any]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def option_keys(n: int) -> list[str]:
    letters = string.ascii_uppercase
    return list(letters[:n]) if n <= 26 else [a + b for a in letters for b in letters][:n]


def mc_criteria(options: list[str]) -> dict[str, str]:
    return dict(zip(option_keys(len(options)), options))


def stable_rank(*parts: Any) -> str:
    """Deterministic sort key: a smaller sample is always a subset of a larger one."""
    return hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------
# I/O


def write_examples(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in examples:
            f.write(e.to_json() + "\n")


def read_examples(path: Path) -> list[Example]:
    return [Example.from_json(line) for line in path.read_text().splitlines() if line.strip()]


def request_key(state: Any, questions: dict) -> str:
    payload = json.dumps({"state": state, "questions": questions}, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_responses(path: Path) -> dict[str, dict]:
    """uid -> response record. Lines cut off by an interrupted run are ignored (and redone)."""
    out = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["uid"]] = rec
    return out


# ----------------------------------------------------------------------------------------------------
# Client


class JevClient:
    def __init__(self, url: str, timeout: float = 600.0):
        self.url = url
        self.http = httpx.Client(timeout=timeout)

    def version(self) -> dict:
        r = self.http.get(self.url.rsplit("/", 1)[0] + "/version")
        r.raise_for_status()
        return r.json()

    def ask(self, state: Any, questions: dict) -> dict:
        r = self.http.post(self.url, json={"state": state, "questions": questions})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()


def run_examples(client: JevClient, examples: list[Example], out_path: Path, concurrency: int = 2) -> int:
    """Send every example that has no response in `out_path` yet; append responses as they arrive.
    Safe to interrupt: a rerun only sends what is missing. Returns the number of new responses."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_responses(out_path)
    todo = [e for e in examples if e.uid not in done or done[e.uid]["key"] != request_key(e.state, e.questions)]
    if not todo:
        return 0
    lock = threading.Lock()
    failures = 0
    with out_path.open("a") as out, ThreadPoolExecutor(concurrency) as pool:
        futures = {pool.submit(client.ask, e.state, e.questions): e for e in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=out_path.stem):
            e = futures[fut]
            try:
                resp = fut.result()
            except Exception as err:  # keep going; the example is retried on the next run
                failures += 1
                tqdm.write(f"{e.uid}: {err}")
                continue
            rec = {"uid": e.uid, "key": request_key(e.state, e.questions), "answers": resp["answers"],
                   "model": resp.get("model"), "usage": resp.get("usage")}
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
    if failures:
        print(f"{failures} request(s) failed; rerun to retry them")
    return len(todo) - failures
