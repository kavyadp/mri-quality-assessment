"""Stage 4c: can a few clean scans from a new scanner fix the decision threshold?

On an unseen scanner both models rank slices well (high AUROC) but their
probabilities shift, so the default 0.5 threshold flags many clean slices.
In practice a site can easily acquire a handful of good scans. Here the
threshold is set to the 95th percentile of p(degraded) on the clean slices of
k randomly chosen IOP subjects, and the remaining IOP subjects are evaluated.
Repeated 200 times per k.

Writes outputs/comparison/site_calibration.csv.

Usage:
    python scripts/site_calibration.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

MODELS = {
    "classical RF": Path("outputs/baseline/predictions_unseen_site_IOP.csv"),
    "CNN": Path("outputs/cnn/unseen_site/predictions_unseen_site_IOP.csv"),
}


def run(pred: pd.DataFrame, k: int, reps: int, rng: np.random.Generator) -> dict:
    subjects = np.array(sorted(pred.subject.unique()))
    ba, spec, sens = [], [], []
    for _ in range(reps):
        cal = rng.choice(subjects, k, replace=False)
        thr = np.quantile(pred[pred.subject.isin(cal) & (pred.artefact == "clean")].p_degraded, 0.95)
        ev = pred[~pred.subject.isin(cal)]
        flag = (ev.p_degraded > thr).to_numpy()
        y = ev.degraded.to_numpy()
        ba.append(balanced_accuracy_score(y, flag))
        spec.append(1 - flag[y == 0].mean())
        sens.append(flag[y == 1].mean())
    return {"bal_acc": np.mean(ba), "bal_acc_sd": np.std(ba), "specificity": np.mean(spec),
            "sensitivity": np.mean(sens)}


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for model, path in MODELS.items():
        pred = pd.read_csv(path)
        flag = pred.p_degraded >= 0.5
        rows.append({"model": model, "clean_scans_used": 0,
                     "bal_acc": balanced_accuracy_score(pred.degraded, flag), "bal_acc_sd": 0.0,
                     "specificity": 1 - flag[pred.degraded == 0].mean(),
                     "sensitivity": flag[pred.degraded == 1].mean()})
        for k in (3, 5, 10):
            rows.append({"model": model, "clean_scans_used": k, **run(pred, k, 200, rng)})
    df = pd.DataFrame(rows)
    out = Path("outputs/comparison")
    out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / "site_calibration.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
