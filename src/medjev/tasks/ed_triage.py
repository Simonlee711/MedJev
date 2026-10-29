"""MIMIC-IV-ED 2.2: emergency-department disposition decided at triage, as Jev questions.

The decision is made at triage, when it is most useful (bed requests, fast-tracking) and least informed. The state
holds only what is known then: arrival transport, sex, race, age (from the patient's MEDS birth record), the triage
vital signs, pain, acuity (ESI) and chief complaint, and the patient's ED history from earlier stays (past-year ED
visits and admissions). Everything recorded later in the visit (ED medications, repeated vitals, home-medication
reconciliation, which is charted a median 69 min after arrival, and the ED diagnoses) is excluded.

  ed_triage_admission    Noul: will the ED disposition be ADMITTED?
  ed_triage_disposition  Choice: home / admitted / transfer / left before the visit was complete / expired / other

Patients follow the MEDS splits (train / tuning / held_out by subject_id); evaluation uses held_out patients, one
stay per patient, hash-ordered. Patients without a MEDS record (no age) are excluded.
"""

import json
import math
from datetime import timedelta
from functools import cache

import pandas as pd

from medjev.config import DATA_DIR, EVAL_DATASETS, MEDS_DIR
from medjev.jev import Example, choice, noul, stable_rank

ED_DIR = EVAL_DATASETS / "mimic-iv-ed-2.2" / "ed"
CATEGORY = {"HOME": "home", "ADMITTED": "admitted", "TRANSFER": "transfer", "LEFT WITHOUT BEING SEEN": "left",
            "ELOPED": "left", "LEFT AGAINST MEDICAL ADVICE": "left", "EXPIRED": "expired", "OTHER": "other"}
DISPOSITIONS = {
    "home": "Discharged home from the emergency department.",
    "admitted": "Admitted to the hospital.",
    "transfer": "Transferred to another facility.",
    "left": "Left before the visit was complete (without being seen, eloped, or against medical advice).",
    "expired": "Died in the emergency department.",
    "other": "Another disposition.",
}
ADMISSION_Q = noul(
    "Based on the information available at emergency department triage, will this visit end with the patient "
    "admitted to the hospital?",
    true="The emergency department disposition is admission to the hospital.",
    false="The patient is not admitted (discharged home, transferred, leaves, or another disposition).",
)
DISPOSITION_Q = choice("Based on the information available at emergency department triage, what will the "
                       "disposition of this emergency department visit be?", DISPOSITIONS)
TRIAGE_FIELDS = {"temperature": ("temperature_F", 1), "heartrate": ("heart_rate_bpm", 0), "resprate": ("respiratory_rate", 0),
                 "o2sat": ("o2_saturation_pct", 0), "sbp": ("systolic_bp_mmHg", 0), "dbp": ("diastolic_bp_mmHg", 0)}


@cache
def meds_splits() -> dict[int, str]:
    raw = json.loads((MEDS_DIR / "splits.json").read_text())
    return {pid: name.split("/")[0] for name, pids in raw.items() for pid in pids}


def birth_dates() -> pd.Series:
    path = DATA_DIR / "cohorts" / "meds_birth.parquet"
    if not path.is_file():
        import polars as pl
        files = [str(f) for f in sorted((MEDS_DIR / "data").glob("*/*.parquet"))]
        b = (pl.scan_parquet(files).select("patient_id", "time", "code").filter(pl.col("code") == "MEDS_BIRTH")
             .group_by("patient_id").agg(pl.col("time").min().alias("birth")).collect())
        path.parent.mkdir(parents=True, exist_ok=True)
        b.write_parquet(path)
    b = pd.read_parquet(path)
    return b.set_index("patient_id")["birth"]


@cache
def table() -> pd.DataFrame:
    path = DATA_DIR / "cohorts" / "ed_triage.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    e = pd.read_csv(ED_DIR / "edstays.csv.gz", parse_dates=["intime", "outtime"])
    t = pd.read_csv(ED_DIR / "triage.csv.gz")
    df = e.merge(t.drop(columns="subject_id"), on="stay_id", how="inner").sort_values(["subject_id", "intime"])
    # past-year ED history from earlier stays of the same patient
    prior_visits, prior_admits = [], []
    for _, g in df.groupby("subject_id", sort=False):
        times, adm = g["intime"].tolist(), (g["disposition"] == "ADMITTED").tolist()
        for i, ti in enumerate(times):
            lo = ti - timedelta(days=365)
            idx = [j for j in range(i) if times[j] >= lo and g["outtime"].iloc[j] <= ti]
            prior_visits.append(len(idx))
            prior_admits.append(sum(adm[j] for j in idx))
    df["prior_ed_visits"], df["prior_ed_admissions"] = prior_visits, prior_admits
    birth = birth_dates()
    df["age"] = (df["intime"] - df["subject_id"].map(birth)).dt.days / 365.25
    df["split"] = df["subject_id"].map(meds_splits())
    df = df[df["age"].notna() & df["split"].notna() & (df["age"] >= 18)]
    df["category"] = df["disposition"].map(CATEGORY).fillna("other")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def _num(v, digits):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), digits) if digits else int(round(float(v)))


def state(r: dict) -> dict:
    tri = {key: _num(r[col], d) for col, (key, d) in TRIAGE_FIELDS.items()}
    tri["pain"] = None if pd.isna(r["pain"]) else str(r["pain"])
    tri["acuity_esi"] = _num(r["acuity"], 0)
    tri["chief_complaint"] = None if pd.isna(r["chiefcomplaint"]) else str(r["chiefcomplaint"])
    s = {
        "decision_time": "at emergency department triage",
        "patient": {"age": int(r["age"]), "sex": {"F": "female", "M": "male"}.get(r["gender"], r["gender"]),
                    "race": r["race"]},
        "arrival_transport": r["arrival_transport"],
        "triage": {k: v for k, v in tri.items() if v is not None},
        "past_year_emergency_department_history": {"visits": int(r["prior_ed_visits"]),
                                                    "visits_ending_in_admission": int(r["prior_ed_admissions"])},
    }
    return s


def _rows(split: str, limit: int | None, one_per_patient: bool) -> list[dict]:
    df = table()
    df = df[df["split"] == split].copy()
    df["rank"] = df["stay_id"].map(lambda s: stable_rank("ed_triage", s))
    df = df.sort_values("rank")
    if one_per_patient:
        df = df.drop_duplicates("subject_id", keep="first")
    return df.head(limit).to_dict("records") if limit else df.to_dict("records")


def _split(split: str | None) -> str:
    return {"test": "held_out", None: "held_out"}.get(split, split)


def build_admission(split: str | None = None, limit: int | None = None, one_per_patient: bool = True) -> list[Example]:
    sp = _split(split)
    return [Example("ed_triage_admission", f"{sp}/{r['stay_id']}", state(r), {"label": ADMISSION_Q},
                    {"label": r["disposition"] == "ADMITTED"},
                    {"subject_id": int(r["subject_id"]), "stay_id": int(r["stay_id"]), "disposition": r["disposition"]})
            for r in _rows(sp, limit, one_per_patient)]


def build_disposition(split: str | None = None, limit: int | None = None, one_per_patient: bool = True) -> list[Example]:
    sp = _split(split)
    return [Example("ed_triage_disposition", f"{sp}/{r['stay_id']}", state(r), {"answer": DISPOSITION_Q},
                    {"answer": r["category"]},
                    {"subject_id": int(r["subject_id"]), "stay_id": int(r["stay_id"]), "disposition": r["disposition"]})
            for r in _rows(sp, limit, one_per_patient)]
