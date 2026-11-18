"""Classical baseline: three random forests on the no-reference features.

Shared by scripts/baseline_classical.py (training + IXI evaluation) and
scripts/eval_mrart.py (applies the same models to real artefacts). Training is
deterministic for a given seed, so both scripts get identical forests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .features import FEATURE_NAMES

RF_PARAMS = dict(n_estimators=300, min_samples_leaf=5, n_jobs=-1)


def fit_rf(train: pd.DataFrame, seed: int = 0) -> dict:
    X = train[list(FEATURE_NAMES)].to_numpy()
    p = dict(RF_PARAMS, random_state=seed)
    return {
        "quality": RandomForestClassifier(class_weight="balanced", **p).fit(X, train["degraded"]),
        "type": RandomForestClassifier(class_weight="balanced", **p).fit(X, train["type_idx"]),
        "severity": RandomForestRegressor(**p).fit(X, train["severity_target"]),
    }


def predict_rf(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Adds p_degraded, type_prob_<class>, type_pred and sev_pred columns."""
    X = df[list(FEATURE_NAMES)].to_numpy()
    d = df.copy()
    d["p_degraded"] = models["quality"].predict_proba(X)[:, 1]
    prob = models["type"].predict_proba(X)
    for j, c in enumerate(models["type"].classes_):
        d[f"type_prob_{int(c)}"] = prob[:, j]
    d["type_pred"] = models["type"].classes_[np.argmax(prob, axis=1)]
    d["sev_pred"] = models["severity"].predict(X)
    return d


def merge_features(labels: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Join slice labels with cached features; slices with an empty head mask get feature medians."""
    df = labels.merge(feats, on=["subject", "artefact", "slice"], validate="one_to_one")
    cols = list(FEATURE_NAMES)
    df[cols] = df[cols].fillna(df[cols].median())
    return df
