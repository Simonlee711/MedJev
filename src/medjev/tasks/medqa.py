"""MedQA (USMLE, 5 options) as a Jev Choice question. Same layout as MMLU in the benchmarking harness:
the question goes into `state`, the options become Choice criteria A-E."""

import pandas as pd

from medjev.config import MEDQA_DIR
from medjev.jev import Example, choice, mc_criteria, option_keys, stable_rank

INSTRUCTION = "Which option correctly answers `question`?"


def build(split: str = "test", limit: int | None = None) -> list[Example]:
    df = pd.read_parquet(MEDQA_DIR / f"{split}-00000-of-00001.parquet")
    rows = sorted(df.itertuples(index=False), key=lambda r: stable_rank("medqa", split, r.id))
    out = []
    for r in rows[:limit]:
        options = [str(c) for c in r.choices]
        gold = option_keys(len(options))[options.index(r.answer[0])]
        state = {"subject": "medicine (USMLE)", "question": r.question}
        out.append(Example("medqa", f"{split}/{r.id}", state, {"answer": choice(INSTRUCTION, mc_criteria(options))},
                           {"answer": gold}))
    return out


def build_train(limit: int | None = None) -> list[Example]:
    """MedQA *train* questions (for recalibration), minus every vignette that also appears in NEJM-MedQA (so the NEJM
    evaluation stays clean)."""
    from medjev.tasks.qa import _norm, nejm_table
    nejm_keys = {_norm(q)[:150] for q in nejm_table()["question"]}
    out = [e for e in build("train", None) if _norm(e.state["question"])[:150] not in nejm_keys]
    return out[:limit] if limit else out
