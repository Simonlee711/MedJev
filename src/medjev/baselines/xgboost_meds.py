"""XGBoost baseline for the MEDS binary tasks.

Features come from the same leakage-safe windows as the Jev states (events in the year up to the prediction
time): per-code event counts (whole window and current admission), the last numeric value of frequent
measurement codes, and a few static fields. Trained on the same training examples as every other supervised method
(data/prepared_train) and evaluated on exactly the held-out examples that MedJev answers.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
import xgboost as xgb

from medjev.jev import stable_rank
from medjev.tasks import meds

N_CODES = 1500  # most common codes (by number of training samples that have them)
N_VALUES = 300  # most common codes with numeric values


def aggregate(split: str, samples: pl.DataFrame, shards: int | None = None, task: str | None = None) -> pl.DataFrame:
    """One row per (sample, code): window count, current-admission count, last numeric value."""
    return (
        meds.window_events(split, samples, shards, task=task)
        .with_columns(current=pl.col("time") >= pl.col("adm_time"))
        .group_by("sample_id", "code")
        .agg(
            n=pl.len(),
            n_current=pl.col("current").sum(),
            last=pl.col("numeric_value").sort_by("time").drop_nulls().last(),
        )
        .collect(engine="streaming")
    )


@dataclass
class Vocab:
    codes: list[str]
    value_codes: list[str]
    admission_types: list[str]
    insurances: list[str]

    @staticmethod
    def fit(samples: pl.DataFrame, agg: pl.DataFrame) -> "Vocab":
        codes = agg.group_by("code").len().sort(["len", "code"], descending=[True, False]).head(N_CODES)["code"]
        values = (agg.filter(pl.col("last").is_not_null()).group_by("code").len()
                  .sort(["len", "code"], descending=[True, False]).head(N_VALUES)["code"])
        cats = lambda col: sorted(v for v in samples[col].unique().to_list() if v is not None)
        return Vocab(codes.to_list(), values.to_list(), cats("admission_type"), cats("insurance"))

    def names(self) -> list[str]:
        return (["age", "male", "days_since_admission", "prior_admissions_past_year"]
                + [f"admission_type={v}" for v in self.admission_types] + [f"insurance={v}" for v in self.insurances]
                + [f"count:{c}" for c in self.codes] + [f"count_this_admission:{c}" for c in self.codes]
                + [f"last:{c}" for c in self.value_codes])


def featurize(samples: pl.DataFrame, agg: pl.DataFrame, vocab: Vocab) -> np.ndarray:
    n = samples.height
    row = pl.DataFrame({"sample_id": samples["sample_id"], "row": np.arange(n)})
    static = [
        samples["age"].to_numpy().astype(np.float32),
        (samples["sex"] == "M").to_numpy().astype(np.float32),
        ((samples["pred_time"] - samples["adm_time"]).dt.total_seconds() / 86400).to_numpy().astype(np.float32),
    ]
    prior = (agg.filter(pl.col("code").str.starts_with("HOSPITAL_ADMISSION//"))
             .group_by("sample_id").agg((pl.col("n") - pl.col("n_current")).sum().alias("k")))
    static.append(row.join(prior, on="sample_id", how="left")["k"].fill_null(0).to_numpy().astype(np.float32))
    for col, values in (("admission_type", vocab.admission_types), ("insurance", vocab.insurances)):
        static += [(samples[col] == v).fill_null(False).to_numpy().astype(np.float32) for v in values]

    k = len(vocab.codes)
    dyn = np.full((n, 2 * k + len(vocab.value_codes)), np.nan, dtype=np.float32)  # absent code = missing
    a = agg.join(row, on="sample_id")
    code_idx = pl.DataFrame({"code": vocab.codes, "ci": np.arange(k)})
    c = a.join(code_idx, on="code")
    r, ci = c["row"].to_numpy(), c["ci"].to_numpy()
    dyn[r, ci] = c["n"].to_numpy()
    dyn[r, k + ci] = c["n_current"].to_numpy()
    val_idx = pl.DataFrame({"code": vocab.value_codes, "vi": np.arange(len(vocab.value_codes))})
    v = a.join(val_idx, on="code").filter(pl.col("last").is_not_null())
    dyn[v["row"].to_numpy(), 2 * k + v["vi"].to_numpy()] = v["last"].to_numpy()
    return np.hstack([np.stack(static, axis=1), dyn])


def train_and_predict(task: str, eval_limit: int | None = None, train_shards: int | None = None,
                      max_train: int | None = None, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray, dict]:
    """Returns (held-out eval samples, predicted probabilities, info). The training set is the prepared training
    sample (one admission per patient, hash order); 20% of it is held out for early stopping."""
    pool = meds.sample(meds.cohort(task, "train", train_shards), task, max_train, one_per_patient=True)
    is_val = pool["sample_id"].map_elements(lambda u: int(stable_rank("xgb-val", u)[:8], 16) % 5 == 0,
                                            return_dtype=pl.Boolean)
    train_s, tune_s = pool.filter(~is_val), pool.filter(is_val)
    eval_s = meds.sample(meds.cohort(task, meds.DEFAULT_SPLIT), task, eval_limit)  # = the MedJev examples

    agg_train = aggregate("train", train_s, train_shards, task)
    vocab = Vocab.fit(train_s, agg_train)
    x_train = featurize(train_s, agg_train, vocab)
    del agg_train
    x_tune = featurize(tune_s, aggregate("train", tune_s, train_shards, task), vocab)
    x_eval = featurize(eval_s, aggregate(meds.DEFAULT_SPLIT, eval_s, task=task), vocab)
    y_train, y_tune = train_s["label"].to_numpy(), tune_s["label"].to_numpy()

    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=6, min_child_weight=5, subsample=0.8,
        colsample_bytree=0.5, tree_method="hist", eval_metric="auc", early_stopping_rounds=100,
        n_jobs=-1, random_state=seed,
    )
    model.fit(x_train, y_train, eval_set=[(x_tune, y_tune)], verbose=False)
    probs = model.predict_proba(x_eval)[:, 1]

    names = vocab.names()
    gain = model.get_booster().get_score(importance_type="gain")
    top = sorted(gain.items(), key=lambda kv: -kv[1])[:20]
    info = {
        "n_train": int(len(y_train)), "train_positive_rate": float(y_train.mean()),
        "n_early_stopping": int(len(y_tune)), "tune_auroc": float(model.best_score), "best_iteration": int(model.best_iteration),
        "n_features": len(names), "train_shards": train_shards,
        "top_features_by_gain": [{"feature": _readable(names[int(f[1:])]), "gain": round(g, 2)} for f, g in top],
    }
    return eval_s, probs, info


def _readable(name: str) -> str:
    kind, _, code = name.partition(":")
    return f"{kind}: {meds.describe(code)} [{code}]" if code else name


def matrices(task: str, eval_limit: int | None = None, train_shards: int | None = None, max_train: int | None = None):
    """Feature matrices for the training examples and the held-out evaluation examples (same as the Jev examples):
    (x_train, y_train, x_test, y_test, test sample ids, feature names)."""
    train_s = meds.sample(meds.cohort(task, "train", train_shards), task, max_train, one_per_patient=True)
    eval_s = meds.sample(meds.cohort(task, meds.DEFAULT_SPLIT), task, eval_limit)
    agg_train = aggregate("train", train_s, train_shards, task)
    vocab = Vocab.fit(train_s, agg_train)
    x_tr = featurize(train_s, agg_train, vocab)
    x_te = featurize(eval_s, aggregate(meds.DEFAULT_SPLIT, eval_s, task=task), vocab)
    return (x_tr, train_s["label"].to_numpy().astype(int), x_te, eval_s["label"].to_numpy().astype(int),
            eval_s["sample_id"].to_list(), vocab.names())
