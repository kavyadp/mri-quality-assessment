"""Metrics shared by every model, so the classical baseline and the CNN are scored identically.

``evaluate`` takes a dataframe with the slice labels (site, artefact, severity,
degraded, type_idx, severity_target) plus prediction columns:
  p_degraded   probability that the slice is degraded
  type_pred    predicted class index (TYPE_CLASSES)
  sev_pred     predicted 1 - SSIM
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             roc_auc_score)

from .labels import TYPE_CLASSES


def _core(df: pd.DataFrame) -> dict:
    y, p = df["degraded"].to_numpy(), df["p_degraded"].to_numpy()
    out = {"n_slices": int(len(df)), "n_subjects": int(df["subject"].nunique())}
    out["quality_auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    pred = (p >= 0.5).astype(int)
    out["quality_bal_acc"] = float(balanced_accuracy_score(y, pred))
    out["quality_sensitivity"] = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    out["quality_specificity"] = float(1 - pred[y == 0].mean()) if (y == 0).any() else float("nan")
    out["type_acc"] = float((df["type_pred"] == df["type_idx"]).mean())
    out["type_macro_f1"] = float(f1_score(df["type_idx"], df["type_pred"], average="macro",
                                          labels=list(range(len(TYPE_CLASSES)))))
    out["severity_mae"] = float(mean_absolute_error(df["severity_target"], df["sev_pred"]))
    out["severity_spearman"] = float(spearmanr(df["severity_target"], df["sev_pred"]).statistic)
    return out


def evaluate(df: pd.DataFrame) -> dict:
    res = {"overall": _core(df)}
    res["by_site"] = {site: _core(g) for site, g in df.groupby("site")}
    # Detection rate per simulated artefact and nominal level: which damage gets missed?
    det = {}
    for (art, sev), g in df.groupby(["artefact", "severity"]):
        det[f"{art}_L{sev}"] = {
            "n": int(len(g)),
            "frac_truly_degraded": float(g["degraded"].mean()),
            "frac_flagged": float((g["p_degraded"] >= 0.5).mean()),
            "type_recall": float((g["type_pred"] == g["type_idx"]).mean()),
        }
    res["by_artefact_level"] = det
    res["type_confusion"] = confusion_matrix(df["type_idx"], df["type_pred"],
                                             labels=list(range(len(TYPE_CLASSES)))).tolist()
    return res


def summary_table(results: dict[str, dict]) -> pd.DataFrame:
    """One row per model/split, overall metrics only."""
    return pd.DataFrame({k: v["overall"] for k, v in results.items()}).T
