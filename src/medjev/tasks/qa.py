"""Diagnostic multiple-choice QA beyond MedQA, as Jev Choice questions.

- DiagnosisArena (shzyk/DiagnosisArena): 915 published case reports (case, examination, tests) with four
  candidate final diagnoses.
- NEJM-MedQA diagnostic reasoning (katielink/nejm-medqa-diagnostic-reasoning-dataset, medqa_test; Savage et
  al.): 512 MedQA-style vignettes released *open-ended* (no options) with graded GPT-3.5/GPT-4 free-text answers.
  Jev cannot generate, so every question becomes a Choice: vignettes found in MedQA keep their original five
  options and answer; the rest get the gold diagnosis plus up to four of the dataset's GPT answers that were graded
  wrong, padded to five options with other questions' gold diagnoses. All questions are kept.
"""

import re

import pandas as pd

from medjev.config import MEDQA_DIR
from medjev.jev import Example, choice, mc_criteria, option_keys, stable_rank

DX_INSTRUCTION = "Which option is the most likely final diagnosis for the patient described in the state?"
MC_INSTRUCTION = "Which option correctly answers `question`?"

# NEJM-MedQA free-text answer columns and their grade columns (first block GPT-3.5, second GPT-4).
NEJM_GPT35 = [("new_IR", "IR Grade"), ("new_AR", "AR Grade"), ("new_BR", "BR Grade"), ("new_DR", "DR Grade")]
NEJM_GPT4 = [("CoT.2", "CoT Grade"), ("DR ", "DR Grade.1"), ("IR", "IR Grade.1"), ("AR", "AR Grade.1"), ("BR", "BR Grade.1")]


def build_diagnosisarena(split: str | None = None, limit: int | None = None) -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset("shzyk/DiagnosisArena")["test"]
    rows = sorted(ds, key=lambda r: stable_rank("diagnosisarena", r["id"]))
    out = []
    for r in rows[:limit]:
        keys = sorted(r["Options"])
        options = [r["Options"][k] for k in keys]
        state = {"case_information": r["Case Information"], "physical_examination": r["Physical Examination"],
                 "diagnostic_tests": r["Diagnostic Tests"]}
        gold = option_keys(len(options))[keys.index(r["Right Option"])]
        out.append(Example("diagnosisarena", str(r["id"]), state,
                           {"answer": choice(DX_INSTRUCTION, mc_criteria(options))}, {"answer": gold}))
    return out


def _norm(s) -> str:
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def nejm_table() -> pd.DataFrame:
    from datasets import load_dataset
    df = load_dataset("katielink/nejm-medqa-diagnostic-reasoning-dataset", "medqa_test")["train"].to_pandas()
    return df[df["question"].notna()].reset_index(drop=True)


def medqa_index() -> pd.DataFrame:
    mq = pd.concat([pd.read_parquet(MEDQA_DIR / f"{s}-00000-of-00001.parquet").assign(split=s)
                    for s in ("train", "validation", "test")])
    mq["k"] = mq["question"].map(lambda s: _norm(s)[:150])
    return mq.drop_duplicates("k")


def build_nejm(split: str | None = None, limit: int | None = None) -> list[Example]:
    df = nejm_table()
    df["row"] = range(len(df))  # the dataset's own `index` column repeats values for different questions
    df["k"] = df["question"].map(lambda s: _norm(s)[:150])
    m = df.merge(medqa_index()[["k", "choices", "answer", "split", "id"]], on="k", how="left", suffixes=("", "_mq"))
    out = []
    for r in m.to_dict("records"):
        uid = str(int(r["row"]))
        if isinstance(r["id"], str):  # found in MedQA: original five options and answer
            options = [str(c) for c in r["choices"]]
            gold_text = str(r["answer_mq"][0])
            source = f"medqa_{r['split']}"
        else:  # gold + distinct GPT answers graded wrong, in a deterministic order
            gold_text = str(r["answer"]).strip()
            wrong = []
            for col, grade in NEJM_GPT35 + NEJM_GPT4:
                ans = r[col]
                usable = isinstance(ans, str) and len(ans) <= 120 and not ans.lower().startswith("rationale")
                if usable and r[grade] == "N" and _norm(ans) not in {_norm(x) for x in wrong + [gold_text]}:
                    wrong.append(ans.strip().rstrip("."))
            # Pad to five options with other questions' gold diagnoses, so chance is 20% as for MedQA items.
            pool = sorted({str(a).strip() for a in m["answer"]} - {gold_text}, key=lambda o: stable_rank("pad", uid, o))
            for o in pool:
                if len(wrong) >= 4:
                    break
                if _norm(o) not in {_norm(x) for x in wrong + [gold_text]}:
                    wrong.append(o)
            options = sorted([gold_text] + wrong[:4], key=lambda o: stable_rank("nejm", uid, o))
            source = "gpt_distractors"
        gold = option_keys(len(options))[options.index(gold_text)]
        out.append(Example("nejm_medqa", uid, {"question": r["question"]},
                           {"answer": choice(MC_INSTRUCTION, mc_criteria(options))}, {"answer": gold},
                           {"option_source": source}))
    out.sort(key=lambda e: stable_rank("nejm_medqa", e.uid))
    return out[:limit]


def nejm_reference_accuracy() -> dict:
    """Accuracy of the published GPT-3.5 / GPT-4 free-text answers (graded Y/N by the dataset authors)."""
    df = nejm_table()
    out = {}
    for name, cols in (("GPT-3.5", NEJM_GPT35), ("GPT-4", NEJM_GPT4)):
        for ans, grade in cols:
            g = df[grade]
            out[f"{name} {ans.strip()}"] = {"accuracy": float((g == "Y").mean()), "n": int(g.isin(["Y", "N"]).sum())}
    return out


def build_usmle4(split: str | None = None, limit: int | None = None) -> list[Example]:
    """MedQA-USMLE-4-options (GBaker): the test split (1,273 questions, four options each); `limit` keeps the first
    `limit` questions."""
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")["test"]
    out = []
    for i, r in enumerate(ds):
        keys = sorted(r["options"])
        state = {"subject": "medicine (USMLE)", "question": r["question"]}
        out.append(Example("usmle4", str(i), state,
                           {"answer": choice(MC_INSTRUCTION, {k: r["options"][k] for k in keys})},
                           {"answer": r["answer_idx"]}, {"step": r["meta_info"]}))
    return out[:limit] if limit else out


def build_usmle4_train(limit: int | None = None) -> list[Example]:
    """MedQA-USMLE-4-options train split, hash-ordered, for the supervised option-scoring baselines."""
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")["train"]
    idx = sorted(range(len(ds)), key=lambda i: stable_rank("usmle4-train", i))[:limit or None]
    out = []
    for i in idx:
        r = ds[i]
        keys = sorted(r["options"])
        out.append(Example("usmle4", f"train/{i}", {"subject": "medicine (USMLE)", "question": r["question"]},
                           {"answer": choice(MC_INSTRUCTION, {k: r["options"][k] for k in keys})}, {"answer": r["answer_idx"]}))
    return out
