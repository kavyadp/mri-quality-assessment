"""Training targets derived from the slice table (data/processed/slices.csv).

Three tasks, shared by the classical baseline and the CNN:

quality   binary: degraded if masked SSIM vs. clean < GOOD_SSIM. This is the
          screening question ("does this slice need a second look?").
type      4 classes: clean / noise / blur / motion (what was simulated).
severity  regression target 1 - SSIM. Measured damage, not the nominal level:
          nominal motion levels overlap strongly (see README, Stage 2-3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GOOD_SSIM = 0.95
TYPE_CLASSES = ("clean", "noise", "blur", "motion")


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["degraded"] = (df["ssim"] < GOOD_SSIM).astype(np.int8)
    df["type_idx"] = df["artefact"].map({c: i for i, c in enumerate(TYPE_CLASSES)}).astype(np.int8)
    df["severity_target"] = (1.0 - df["ssim"]).clip(lower=0.0)
    return df
