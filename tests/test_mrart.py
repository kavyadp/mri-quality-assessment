import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location("eval_mrart", Path(__file__).parents[1] / "scripts" / "eval_mrart.py")
eval_mrart = importlib.util.module_from_spec(_spec)
sys.modules["eval_mrart"] = eval_mrart
_spec.loader.exec_module(eval_mrart)


def test_pairwise_order_counts_only_pairs_with_different_scores():
    scans = pd.DataFrame({
        "subject": ["a"] * 3 + ["b"] * 3,
        "score": [1, 2, 3, 1, 1, 3],
        "v": [0.1, 0.2, 0.3, 0.5, 0.1, 0.2],
    })
    # a: 3 pairs, all ordered correctly. b: pair (1,1) ignored; (0.5 vs 0.2) wrong, (0.1 vs 0.2) right.
    assert eval_mrart.pairwise_order(scans, "v") == pytest.approx(4 / 5)


def test_metrics_for_perfect_score():
    rng = np.random.default_rng(0)
    subjects = [f"s{i}" for i in range(12)]
    scans = pd.DataFrame({
        "subject": np.repeat(subjects, 3),
        "acq": ["standard", "headmotion1", "headmotion2"] * 12,
        "score": [1, 2, 3] * 12,
    })
    scans["p_degraded"] = scans.score / 4.0   # 0.25, 0.5, 0.75: perfect ranking
    m = eval_mrart.metrics_for(scans, "p_degraded", rng)
    assert m["auroc_any_artefact"] == pytest.approx(1.0)
    assert m["auroc_unusable"] == pytest.approx(1.0)
    assert m["within_subject_pair_agreement"] == pytest.approx(1.0)
    assert m["bal_acc_default_threshold"] == pytest.approx(1.0)
    assert m["bal_acc_calibrated_3_still_scans"] == pytest.approx(1.0)
