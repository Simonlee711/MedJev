"""Supervised reference baselines: XGBoost and TF-IDF + logistic regression for every EHR task, with hyperparameters
chosen by 5-fold cross-validation on the training examples only.

EHR tasks use the natural features of each source: bag-of-codes counts and last values for MEDS (xgboost_meds) and
structured triage fields plus chief-complaint tokens for MIMIC-IV-ED.
"""

import json
from itertools import product

import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from medjev.jev import Example

XGB_GRID = [dict(max_depth=d, min_child_weight=m, colsample_bytree=c)
            for d, m, c in product((2, 3, 5), (1, 5), (0.3, 0.7))]
XGB_FIXED = dict(learning_rate=0.05, subsample=0.8, tree_method="hist", n_jobs=-1, random_state=0)
C_GRID = (0.01, 0.1, 1.0, 10.0)


def _score(y, p, multiclass: bool) -> float:
    if multiclass:
        return -log_loss(y, p, labels=np.arange(p.shape[1]))
    return roc_auc_score(y, p)


def tune_xgb(x: np.ndarray, y: np.ndarray, multiclass: bool = False, n_classes: int = 2) -> tuple[dict, int, float]:
    """Grid search by 5-fold CV with early stopping inside each fold -> (params, n_estimators, cv score)."""
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(x, y))
    best = (None, 0, -np.inf)
    for g in XGB_GRID:
        scores, iters = [], []
        for tr, va in folds:
            kw = dict(n_estimators=1000, early_stopping_rounds=50, **XGB_FIXED, **g)
            if multiclass:
                m = xgb.XGBClassifier(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss", **kw)
            else:
                m = xgb.XGBClassifier(eval_metric="auc", **kw)
            m.fit(x[tr], y[tr], eval_set=[(x[va], y[va])], verbose=False)
            p = m.predict_proba(x[va])
            scores.append(_score(y[va], p if multiclass else p[:, 1], multiclass))
            iters.append(m.best_iteration + 1)
        if np.mean(scores) > best[2]:
            best = (g, int(np.mean(iters)), float(np.mean(scores)))
    return best


def fit_xgb(x, y, params, n_estimators, multiclass=False, n_classes=2):
    if multiclass:
        return xgb.XGBClassifier(objective="multi:softprob", num_class=n_classes, n_estimators=n_estimators,
                                 **XGB_FIXED, **params).fit(x, y)
    return xgb.XGBClassifier(n_estimators=n_estimators, **XGB_FIXED, **params).fit(x, y)


def tune_lr(x, y, multiclass=False) -> tuple[float, float]:
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    best = (C_GRID[0], -np.inf)
    for c in C_GRID:
        s = []
        for tr, va in folds.split(x, y):
            m = LogisticRegression(C=c, max_iter=3000).fit(x[tr], y[tr])
            p = m.predict_proba(x[va])
            if multiclass:
                full = np.full((len(va), int(y.max()) + 1), 1e-9)
                full[:, m.classes_] = p
                s.append(-log_loss(y[va], full, labels=np.arange(full.shape[1])))
            else:
                s.append(roc_auc_score(y[va], p[:, 1]))
        if np.mean(s) > best[1]:
            best = (c, float(np.mean(s)))
    return best


def text_of(e: Example) -> str:
    return json.dumps(e.state, ensure_ascii=False)


def tfidf(ngram=(1, 2)):
    return TfidfVectorizer(ngram_range=ngram, min_df=3, max_features=50_000, sublinear_tf=True)
