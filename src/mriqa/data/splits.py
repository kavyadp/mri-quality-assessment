"""Subject-level train/val/test splits.

Splitting by subject (never by slice) prevents leakage: neighbouring slices of
one brain are nearly identical, so a slice-level split would inflate test
scores. Splits are stratified by acquisition site so each split sees all
scanners in the same proportion.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .ixi import IXISubject


def make_subject_splits(subjects: list[IXISubject], fractions=(0.7, 0.15, 0.15),
                        seed: int = 42) -> dict[str, list[str]]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    by_site: dict[str, list[str]] = defaultdict(list)
    for s in subjects:
        by_site[s.site].append(s.key)

    splits = {"train": [], "val": [], "test": []}
    for site in sorted(by_site):
        keys = sorted(by_site[site])
        rng.shuffle(keys)
        n = len(keys)
        n_val = int(round(fractions[1] * n))
        n_test = int(round(fractions[2] * n))
        splits["test"] += keys[:n_test]
        splits["val"] += keys[n_test:n_test + n_val]
        splits["train"] += keys[n_test + n_val:]
    return {k: sorted(v) for k, v in splits.items()}


def save_splits(splits: dict, path: str | Path, seed: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"seed": seed, **splits}, indent=2))


def load_splits(path: str | Path) -> dict[str, list[str]]:
    d = json.loads(Path(path).read_text())
    return {k: d[k] for k in ("train", "val", "test")}
