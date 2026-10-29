"""Feature matrices for the ED-triage baselines: structured triage fields + chief-complaint tokens, built on the prepared
training stays and evaluated on the Jev test stays."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from medjev.config import DATA_DIR, PREPARED_DIR
from medjev.jev import read_examples
from medjev.tasks import ed_triage

NUMERIC = ["age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "acuity", "prior_ed_visits",
           "prior_ed_admissions"]


def _matrix(df, vocab=None):
    x = df[NUMERIC].astype(float).copy()
    x["pain"] = pd.to_numeric(df["pain"], errors="coerce")
    x["male"] = (df["gender"] == "M").astype(float)
    cats = vocab["cats"] if vocab else {c: sorted(df[c].dropna().value_counts().head(12).index) for c in ("race", "arrival_transport")}
    for c, values in cats.items():
        for v in values:
            x[f"{c}={v}"] = (df[c] == v).astype(float)
    cc = df["chiefcomplaint"].fillna("").str.lower()
    vec = vocab["vec"] if vocab else CountVectorizer(min_df=5, max_features=300, binary=True).fit(cc)
    words = pd.DataFrame(vec.transform(cc).toarray(), columns=[f"cc:{w}" for w in vec.get_feature_names_out()], index=df.index)
    return pd.concat([x, words], axis=1).to_numpy(np.float32), {"cats": cats, "vec": vec}


def triage_matrices():
    """(x_train, x_test, train table, admission test examples or None, disposition test examples or None, classes)."""
    df = ed_triage.table().set_index("stay_id")
    train_file = next(p for p in (DATA_DIR / "prepared_train" / f"{t}.jsonl" for t in ("ed_triage_admission", "ed_triage_disposition"))
                      if p.is_file())
    tr_ids = [e.meta["stay_id"] for e in read_examples(train_file)]
    load = lambda t: read_examples(PREPARED_DIR / f"{t}.jsonl") if (PREPARED_DIR / f"{t}.jsonl").is_file() else None
    te_adm, te_dis = load("ed_triage_admission"), load("ed_triage_disposition")
    te_ref = te_adm or te_dis
    tr = df.loc[tr_ids].reset_index()
    te = df.loc[[e.meta["stay_id"] for e in te_ref]].reset_index()
    x_tr, vocab = _matrix(tr)
    x_te, _ = _matrix(te, vocab)
    return x_tr, x_te, tr, te_adm, te_dis, list(ed_triage.DISPOSITIONS)
