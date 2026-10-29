"""MIMIC-IV in MEDS format -> binary prediction tasks posed as Jev Noul questions.

MEDS stores one row per clinical event (patient_id, time, code, numeric_value, ...) and has no labels, so
each task defines a cohort of prediction points and a label from the event stream:

- meds_mortality        adult admissions still in hospital 24 h after admission; predict at 24 h whether the
                        patient dies before discharge (discharge location DIED)
- meds_long_los         same cohort and prediction time; label = total stay > 7 days
- meds_readmission_30d  adult admissions discharged alive; predict at discharge whether a new hospital
                        admission starts within 30 days
- meds_ed_disposition   adult emergency-department visits; predict 2 h after ED arrival whether the visit ends in a
                        hospital admission (including observation stays)

Only events with time <= prediction time (and within a 1-year lookback) are used, for the Jev state and the
XGBoost features alike. MEDS stamps diagnoses and DRGs at discharge, so they never leak into the 24 h tasks.
For ED disposition the admission itself must stay invisible: race, insurance, language and marital status live
on admission rows (so their presence would reveal the label) and are not used; ED/admission bookkeeping codes
from 24 h before arrival onward are dropped; and after arrival only laboratory events are kept.
"""

import math
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from functools import cache
from typing import Callable

import polars as pl

from medjev.config import DATA_DIR, MEDS_DIR
from medjev.jev import Example, noul, stable_rank

LOOKBACK = timedelta(days=365)
EVENT_COLS = ["patient_id", "time", "code", "numeric_value", "text_value"]
DEFAULT_SPLIT = "held_out"


def shard_files(split: str, shards: int | None = None) -> list[str]:
    files = sorted((MEDS_DIR / "data" / split).glob("*.parquet"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"no MEDS shards in {MEDS_DIR / 'data' / split}")
    return [str(f) for f in files[:shards]]


# ----------------------------------------------------------------------------------------------------
# Admissions table (one row per hospital admission), cached per split


def admissions(split: str, shards: int | None = None) -> pl.DataFrame:
    path = DATA_DIR / "cohorts" / f"admissions_{split}_{shards or 'all'}.parquet"
    if path.is_file():
        return pl.read_parquet(path)
    lf = pl.scan_parquet(shard_files(split, shards))
    code, parts = pl.col("code"), pl.col("code").str.split("//")
    hadm = pl.col("hadm_id").cast(pl.Int64)
    adm = (
        lf.filter(code.str.starts_with("HOSPITAL_ADMISSION//") & pl.col("hadm_id").is_not_null())
        .select("patient_id", hadm.alias("hadm_id"), pl.col("time").alias("adm_time"),
                parts.list.get(1, null_on_oob=True).alias("admission_type"),
                parts.list.get(2, null_on_oob=True).alias("admission_location"),
                "insurance", "language", "marital_status", "race")
        .unique("hadm_id", keep="first")
    )
    dis = (
        lf.filter(code.str.starts_with("HOSPITAL_DISCHARGE//") & pl.col("hadm_id").is_not_null())
        .select(hadm.alias("hadm_id"), pl.col("time").alias("disch_time"),
                parts.list.get(1, null_on_oob=True).alias("discharge_location"))
        .unique("hadm_id", keep="last")
    )
    birth = lf.filter(code == "MEDS_BIRTH").group_by("patient_id").agg(pl.col("time").min().alias("birth_time"))
    sex = lf.filter(code.str.starts_with("GENDER//")).group_by("patient_id").agg(
        parts.list.get(1, null_on_oob=True).first().alias("sex"))
    death = lf.filter(code == "MEDS_DEATH").group_by("patient_id").agg(pl.col("time").min().alias("death_time"))
    df = (
        adm.join(dis, on="hadm_id", how="inner")
        .join(birth, on="patient_id", how="left")
        .join(sex, on="patient_id", how="left")
        .join(death, on="patient_id", how="left")
        .filter(pl.col("disch_time") > pl.col("adm_time"))
        .with_columns(
            age=(pl.col("adm_time") - pl.col("birth_time")).dt.total_days() / 365.25,
            los_days=(pl.col("disch_time") - pl.col("adm_time")).dt.total_seconds() / 86400,
        )
        .sort("patient_id", "adm_time")
        .with_columns(next_adm_time=pl.col("adm_time").shift(-1).over("patient_id"))
        .collect()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def ed_visits(split: str, shards: int | None = None) -> pl.DataFrame:
    """One row per ED arrival (TRANSFER_TO//ED). An ED transfer that carries a hadm_id led to an admission."""
    path = DATA_DIR / "cohorts" / f"ed_visits_{split}_{shards or 'all'}.parquet"
    if path.is_file():
        return pl.read_parquet(path)
    lf = pl.scan_parquet(shard_files(split, shards))
    code, parts = pl.col("code"), pl.col("code").str.split("//")
    ed = (lf.filter(code.str.starts_with("TRANSFER_TO//ED"))
          .select("patient_id", pl.col("time").alias("adm_time"), pl.col("hadm_id").is_not_null().alias("label"))
          .unique(["patient_id", "adm_time"], keep="first"))
    birth = lf.filter(code == "MEDS_BIRTH").group_by("patient_id").agg(pl.col("time").min().alias("birth_time"))
    sex = lf.filter(code.str.starts_with("GENDER//")).group_by("patient_id").agg(
        parts.list.get(1, null_on_oob=True).first().alias("sex"))
    df = (
        ed.join(birth, on="patient_id", how="left").join(sex, on="patient_id", how="left")
        .with_columns(
            age=(pl.col("adm_time") - pl.col("birth_time")).dt.total_days() / 365.25,
            pred_time=pl.col("adm_time") + ED_HORIZON,
            admission_type=pl.lit(None, dtype=pl.String), insurance=pl.lit(None, dtype=pl.String),
        )
        .sort("patient_id", "adm_time")
        .collect()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


ED_HORIZON = timedelta(hours=2)
ED_TASK = "meds_ed_disposition"
LEAK_PREFIXES = ("HOSPITAL_ADMISSION", "HOSPITAL_DISCHARGE", "ED_REGISTRATION", "ED_OUT", "TRANSFER_TO", "ICU_",
                 "DRG//", "HCPCS//")


# ----------------------------------------------------------------------------------------------------
# Tasks


@dataclass(frozen=True)
class MedsTask:
    name: str
    description: str
    question: dict
    prediction_point: str  # shown to the model
    cohort: Callable[[pl.DataFrame], pl.DataFrame]  # admissions -> rows with pred_time and label


def _alive_at_24h(adm: pl.DataFrame) -> pl.DataFrame:
    return adm.filter((pl.col("age") >= 18) & (pl.col("los_days") > 1)).with_columns(
        pred_time=pl.col("adm_time") + timedelta(hours=24))


def _readmission(adm: pl.DataFrame) -> pl.DataFrame:
    gap = pl.col("next_adm_time") - pl.col("disch_time")
    return adm.filter((pl.col("age") >= 18) & (pl.col("discharge_location").fill_null("") != "DIED")).with_columns(
        pred_time=pl.col("disch_time"),
        label=((gap > timedelta(0)) & (gap <= timedelta(days=30))).fill_null(False),
    )


TASKS: dict[str, MedsTask] = {t.name: t for t in [
    MedsTask(
        "meds_mortality",
        "In-hospital mortality, predicted 24 h after admission (MIMIC-IV MEDS).",
        noul("Based on the patient's record up to now, will this patient die before being discharged from the "
             "current hospital admission?",
             true="The patient dies during the current hospital admission.",
             false="The patient survives to hospital discharge."),
        "24 hours after hospital admission",
        lambda adm: _alive_at_24h(adm).with_columns(label=pl.col("discharge_location").fill_null("") == "DIED"),
    ),
    MedsTask(
        "meds_long_los",
        "Length of stay > 7 days, predicted 24 h after admission (MIMIC-IV MEDS).",
        noul("Based on the patient's record up to now, will the current hospital stay last longer than 7 days in "
             "total, counted from admission to discharge?",
             true="The hospital stay lasts more than 7 days.",
             false="The patient is discharged (or dies) within 7 days of admission."),
        "24 hours after hospital admission",
        lambda adm: _alive_at_24h(adm).with_columns(label=pl.col("los_days") > 7),
    ),
    MedsTask(
        "meds_readmission_30d",
        "30-day hospital readmission, predicted at discharge (MIMIC-IV MEDS).",
        noul("Based on the patient's record up to this discharge, will the patient be admitted to the hospital "
             "again within 30 days after this discharge?",
             true="A new hospital admission starts within 30 days of this discharge.",
             false="No new hospital admission within 30 days of this discharge."),
        "at discharge from the current hospital admission",
        _readmission,
    ),
    MedsTask(
        ED_TASK,
        "Hospital admission from the emergency department, predicted 2 h after ED arrival (MIMIC-IV MEDS).",
        noul("Based on the patient's record up to now, 2 hours after arriving in the emergency department, will this "
             "emergency department visit end in admission to the hospital?",
             true="The patient is admitted to the hospital (including observation) from this visit.",
             false="The patient leaves the emergency department without being admitted."),
        "2 hours after arrival in the emergency department",
        lambda visits: visits.filter(pl.col("age") >= 18),
    ),
]}


def cohort(task: str, split: str, shards: int | None = None) -> pl.DataFrame:
    if task == ED_TASK:
        return TASKS[task].cohort(ed_visits(split, shards)).with_columns(
            sample_id=pl.format("{}/ed/{}/{}", pl.lit(split), pl.col("patient_id"), pl.col("adm_time").dt.epoch("s")))
    return TASKS[task].cohort(admissions(split, shards)).with_columns(
        sample_id=pl.format("{}/{}", pl.lit(split), pl.col("hadm_id")))


def sample(df: pl.DataFrame, task: str, limit: int | None, one_per_patient: bool = True) -> pl.DataFrame:
    """Deterministic hash-ordered sample (a smaller limit is a subset of a larger one), at natural prevalence.
    With one_per_patient, each patient contributes one randomly chosen admission."""
    key = "sample_id" if task == ED_TASK else "hadm_id"  # admissions keep their original ordering key
    df = df.with_columns(rank=pl.col(key).map_elements(lambda h: stable_rank(task, h), return_dtype=pl.String))
    df = df.sort("rank")
    if one_per_patient:
        df = df.unique("patient_id", keep="first", maintain_order=True)
    return df.head(limit) if limit else df


def window_events(split: str, samples: pl.DataFrame, shards: int | None = None, task: str | None = None) -> pl.LazyFrame:
    """Events of each sample's patient within (pred_time - LOOKBACK, pred_time]. Static rows (no time) are dropped;
    age and sex come from the admissions table. For ED disposition the leakage rules above are applied."""
    s = samples.select("sample_id", "patient_id", "adm_time", "pred_time").lazy()
    ev = (
        pl.scan_parquet(shard_files(split, shards)).select(EVENT_COLS)
        .filter(pl.col("time").is_not_null())
        .join(s, on="patient_id", how="inner")
        .filter((pl.col("time") <= pl.col("pred_time")) & (pl.col("time") > pl.col("pred_time") - LOOKBACK))
    )
    if task == ED_TASK:
        t, arrival = pl.col("time"), pl.col("adm_time")
        bookkeeping = pl.any_horizontal([pl.col("code").str.starts_with(p) for p in LEAK_PREFIXES])
        ev = ev.filter(
            ((t < arrival) & ~(bookkeeping & (t >= arrival - timedelta(hours=24))))
            | ((t >= arrival) & pl.col("code").str.starts_with("LAB//"))
        )
    return ev


# ----------------------------------------------------------------------------------------------------
# Code descriptions and lab reference statistics


@cache
def codebook() -> tuple[dict[str, str], dict[str, str], dict[str, tuple[float, float]]]:
    """(code -> description, lab itemid -> name, code -> (mean, sd) of numeric values)."""
    m = pl.read_parquet(MEDS_DIR / "metadata" / "codes.parquet").filter(pl.col("code").is_not_null())
    desc = {c: d for c, d in zip(m["code"], m["description"]) if d}
    lab_names = {}
    loinc = MEDS_DIR / "extract_code_metadata" / "d_labitems_to_loinc.parquet"
    if loinc.is_file():
        for c, d in pl.read_parquet(loinc).select("code", "description").iter_rows():
            if d and c:
                lab_names.setdefault(c.split("//")[1], d)
    for c, d in desc.items():
        if c.startswith("LAB//"):
            lab_names.setdefault(c.split("//")[1], d)
    stats = {}
    for c, n, s, ss in m.select("code", "values/n_occurrences", "values/sum", "values/sum_sqd").iter_rows():
        if n and n > 30:
            mean = s / n
            var = ss / n - mean * mean
            if var > 0:
                stats[c] = (mean, math.sqrt(var))
    return desc, lab_names, stats


# This extract stores ICU chartevents as LAB//<itemid> but ships no d_items table; these are the standard
# MIMIC-IV itemids of the common bedside measurements. Unnamed chartevents are left out of the state.
CHART_ITEMS = {
    "220045": "Heart rate", "220210": "Respiratory rate", "220277": "SpO2 (pulse oximetry)",
    "220179": "Non-invasive BP systolic", "220180": "Non-invasive BP diastolic", "220181": "Non-invasive BP mean",
    "220050": "Arterial BP systolic", "220051": "Arterial BP diastolic", "220052": "Arterial BP mean",
    "223761": "Temperature (F)", "223762": "Temperature (C)", "223835": "Inspired O2 fraction (FiO2)",
    "220739": "GCS eye opening", "223900": "GCS verbal response", "223901": "GCS motor response",
    "220621": "Glucose (serum)", "225664": "Glucose (finger stick)", "220074": "Central venous pressure",
    "226512": "Admission weight (kg)", "220048": "Heart rhythm",
}
FIRST_CHART_ITEMID = 220000
# Serum indices whose LOINC-map description is a bare letter.
LAB_OVERRIDES = {"50934": "Hemolysis index", "50947": "Icterus index", "51678": "Lipemia index"}


def is_chart_item(code: str) -> bool:
    parts = code.split("//")
    return parts[0] == "LAB" and len(parts) > 1 and parts[1].isdigit() and int(parts[1]) >= FIRST_CHART_ITEMID


def medication_name(code: str) -> str | None:
    """MEDICATION//START//<drug> | MEDICATION//STOP//<drug> (pharmacy) or MEDICATION//<drug>//<status> (eMAR).
    Returns the drug for starts and given doses, None for stops, holds, refusals and unknown drugs."""
    parts = code.split("//")
    if len(parts) < 3:
        return None
    if parts[1] == "START":
        name = parts[2]
    elif parts[1] == "STOP" or any(x in parts[2] for x in SKIP_MED_STATUS):
        return None
    else:
        name = parts[1]
    name = name.strip()
    return None if name in ("", "UNK") else name


def describe(code: str) -> str:
    desc, lab_names, _ = codebook()
    parts = code.split("//")
    kind = parts[0]
    if kind == "LAB" and len(parts) > 1:
        item = parts[1]
        return LAB_OVERRIDES.get(item) or CHART_ITEMS.get(item) or lab_names.get(item) or f"lab item {item}"
    if code in desc:
        return desc[code]
    if kind == "MEDICATION":
        return medication_name(code) or " ".join(parts[1:])
    if kind in ("INFUSION_START", "INFUSION_END") and len(parts) > 1:
        return desc.get(f"INFUSION_START//{parts[1]}") or f"infusion item {parts[1]}"
    if kind in ("DIAGNOSIS", "PROCEDURE") and len(parts) >= 4:
        return f"ICD-{parts[2]} {parts[3]}"
    if kind in ("HCPCS", "DRG", "ICU_ADMISSION", "TRANSFER_TO"):
        return parts[-1]
    return " ".join(parts)


def _num(x: float) -> str:
    return f"{x:.4g}"


# ----------------------------------------------------------------------------------------------------
# Rendering a patient timeline as a Jev state


SKIP_MED_STATUS = ("Not Given", "Hold", "Refused", "Stopped")
CAPS = {"labs": 40, "medications": 40, "infusions": 15, "procedures": 15, "diagnoses": 30, "prior_diagnoses": 25}


def _top(counter: Counter, n: int) -> list[str]:
    return [k for k, _ in counter.most_common(n)]


def render_state(task: MedsTask, s: dict, events: list[dict]) -> dict:
    """s: one cohort row; events: that sample's window events sorted by time."""
    _, _, stats = codebook()
    current = [e for e in events if e["time"] >= s["adm_time"]]
    prior = [e for e in events if e["time"] < s["adm_time"]]

    prior_dx = Counter(describe(e["code"]) for e in prior if e["code"].startswith("DIAGNOSIS//"))
    prior_adm = len({e["time"] for e in prior if e["code"].startswith("HOSPITAL_ADMISSION//")})

    labs: dict[str, tuple] = {}  # code -> latest (value text, z)
    meds, infusions, procs, dx, drg, units, vitals = Counter(), Counter(), Counter(), Counter(), [], [], {}
    for e in current:
        code = e["code"]
        kind = code.split("//")[0]
        if kind == "LAB":
            if is_chart_item(code) and code.split("//")[1] not in CHART_ITEMS:
                continue
            val = e["numeric_value"] if e["numeric_value"] is not None and math.isfinite(e["numeric_value"]) else None
            text = _num(val) if val is not None else (e["text_value"] or "").strip()
            if not text or text.startswith("___"):
                continue
            z = None
            if val is not None and code in stats:
                mean, sd = stats[code]
                z = (val - mean) / sd
            labs[code] = (text, z)
        elif kind == "MEDICATION":
            name = medication_name(code)
            if name:
                meds[name] += 1
        elif kind == "INFUSION_START":
            infusions[describe(code)] += 1
        elif kind in ("PROCEDURE", "HCPCS"):
            procs[describe(code)] += 1
        elif kind == "DIAGNOSIS":
            dx[describe(code)] += 1
        elif kind == "DRG":
            drg.append(describe(code))
        elif kind in ("ICU_ADMISSION", "TRANSFER_TO"):
            unit = describe(code)
            if unit not in ("UNK", "discharge") and unit not in units:
                units.append(unit)
        elif "//" not in code and e["text_value"]:  # OMR vitals: Blood Pressure, Weight (Lbs), BMI (kg/m2), ...
            vitals[code] = e["text_value"]

    def lab_line(code: str) -> str:
        text, z = labs[code]
        unit = code.split("//")[2] if code.count("//") >= 2 else ""
        flag = "" if z is None or abs(z) < 2 else (" (high)" if z > 0 else " (low)")
        return f"{describe(code)}: {text}{' ' + unit if unit and unit != 'UNK' else ''}{flag}"

    by_abnormality = sorted(labs, key=lambda c: -(abs(labs[c][1]) if labs[c][1] is not None else 0))
    lab_order = [c for c in by_abnormality if not is_chart_item(c)]
    chart_order = [c for c in by_abnormality if is_chart_item(c)]

    if task.name == ED_TASK:
        prior_ed = len({e["time"] for e in prior if e["code"].startswith("TRANSFER_TO//ED")})
        return _prune({
            "prediction_time": task.prediction_point,
            "patient": {"age": int(s["age"]) if s["age"] is not None else None,
                        "sex": {"F": "female", "M": "male"}.get(s["sex"], s["sex"])},
            "past_year_before_this_visit": {
                "hospital_admissions": prior_adm, "emergency_department_visits": prior_ed,
                "diagnoses": _top(prior_dx, CAPS["prior_diagnoses"]),
            },
            "this_ed_visit_so_far": {"labs_latest": [lab_line(c) for c in lab_order[: CAPS["labs"]]]},
        })

    admission = {
        "admission_type": s["admission_type"],
        "admitted_from": s["admission_location"],
    }
    if task.name == "meds_readmission_30d":
        admission["length_of_stay_days"] = round(s["los_days"], 1)
        admission["discharge_location"] = s["discharge_location"]
    else:
        admission["hours_since_admission"] = 24

    state = {
        "prediction_time": task.prediction_point,
        "patient": {
            "age": int(s["age"]) if s["age"] is not None else None,
            "sex": {"F": "female", "M": "male"}.get(s["sex"], s["sex"]),
            "race": s["race"], "insurance": s["insurance"], "language": s["language"],
            "marital_status": s["marital_status"],
        },
        "current_admission": admission,
        "past_year_before_this_admission": {
            "hospital_admissions": prior_adm,
            "diagnoses": _top(prior_dx, CAPS["prior_diagnoses"]),
        },
        "this_admission_so_far": {
            "care_units": units,
            "vitals": vitals,
            "bedside_measurements_latest": [lab_line(c) for c in chart_order],
            "labs_latest": [lab_line(c) for c in lab_order[: CAPS["labs"]]],
            "medications": _top(meds, CAPS["medications"]),
            "infusions": _top(infusions, CAPS["infusions"]),
            "procedures": _top(procs, CAPS["procedures"]),
            "diagnoses": _top(dx, CAPS["diagnoses"]),
            "drg": list(dict.fromkeys(drg)),
        },
    }
    return _prune(state)


def _prune(x):
    """Drop empty values so the state stays short."""
    if isinstance(x, dict):
        out = {k: _prune(v) for k, v in x.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(x, list):
        return [_prune(v) for v in x]
    return x


def build(task: str, split: str | None = None, limit: int | None = None, shards: int | None = None) -> list[Example]:
    split = split or DEFAULT_SPLIT
    t = TASKS[task]
    samples = sample(cohort(task, split, shards), task, limit)
    events = window_events(split, samples, shards, task=task).sort("sample_id", "time").collect(engine="streaming")
    by_sample = {k[0]: g.to_dicts() for k, g in events.group_by(["sample_id"])}
    out = []
    for s in samples.iter_rows(named=True):
        state = render_state(t, s, by_sample.get(s["sample_id"], []))
        meta = {"patient_id": s["patient_id"], "hadm_id": s.get("hadm_id"), "pred_time": str(s["pred_time"]), "split": split}
        out.append(Example(task, s["sample_id"], state, {"label": t.question}, {"label": bool(s["label"])}, meta))
    return out
