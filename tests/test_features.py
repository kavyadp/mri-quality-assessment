import numpy as np
import pandas as pd
import pytest

from mriqa.artefacts import add_rician_noise, gaussian_blur, simulate_motion
from mriqa.evaluation import evaluate
from mriqa.features import FEATURE_NAMES, head_mask_2d, slice_features
from mriqa.labels import GOOD_SSIM, add_targets


def head_slice(n=128):
    """Bright scalp ring, dark skull gap, textured brain: enough structure for the features."""
    x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    r = np.sqrt((x / 0.75) ** 2 + (y / 0.85) ** 2)
    img = np.where(r < 0.95, 0.9, 0.0) * (r > 0.88)          # scalp
    brain = r < 0.8
    stripes = 0.55 + 0.25 * (np.sin(18 * x) * np.sin(14 * y) > 0)
    img = img + np.where(brain, stripes, 0.0)
    return img.astype(np.float32)


def f(img):
    return dict(zip(FEATURE_NAMES, slice_features(img)))


def test_head_mask_includes_skull_gap():
    img = head_slice()
    m = head_mask_2d(img)
    x, y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128), indexing="ij")
    r = np.sqrt((x / 0.75) ** 2 + (y / 0.85) ** 2)
    assert m[r < 0.85].mean() > 0.98      # dark ring between scalp and brain is inside the head
    assert m[r > 1.05].mean() < 0.01


def test_feature_vector_shape_and_empty_image():
    assert slice_features(head_slice()).shape == (len(FEATURE_NAMES),)
    assert np.isnan(slice_features(np.zeros((64, 64), np.float32))).all()


def test_features_respond_to_their_artefact():
    img = head_slice()
    clean = f(img)
    blurred = f(gaussian_blur(img, 1.5))
    noisy = f(add_rician_noise(img, 0.05, np.random.default_rng(0)))
    assert blurred["lap_var"] < clean["lap_var"] and blurred["hf_energy"] < clean["hf_energy"]
    assert noisy["noise_mad"] > clean["noise_mad"] and noisy["bg_std"] > clean["bg_std"]
    assert noisy["snr"] < clean["snr"]


def test_motion_ghosts_along_phase_encode_axis():
    vol = np.repeat(head_slice()[:, :, None], 8, axis=2)
    moved, _ = simulate_motion(vol, 4, 5.0, 0.0, np.random.default_rng(3), centre_halfwidth=0.03)
    clean, ghosted = f(vol[:, :, 4]), f(moved[:, :, 4])
    assert ghosted["ghost_pe"] > clean["ghost_pe"]
    assert ghosted["hf_anisotropy"] < clean["hf_anisotropy"]


def test_targets_and_evaluate():
    df = pd.DataFrame({
        "subject": ["a"] * 4 + ["b"] * 4, "site": ["Guys"] * 4 + ["IOP"] * 4,
        "artefact": ["clean", "noise", "blur", "motion"] * 2, "severity": [0, 1, 2, 3] * 2,
        "ssim": [1.0, GOOD_SSIM - 0.01, 0.97, 0.7] * 2,
    })
    df = add_targets(df)
    assert df["degraded"].tolist() == [0, 1, 0, 1] * 2
    df["p_degraded"] = df["degraded"].astype(float)
    df["type_pred"] = df["type_idx"]
    df["sev_pred"] = df["severity_target"]
    res = evaluate(df)
    o = res["overall"]
    assert o["quality_auroc"] == pytest.approx(1.0) and o["type_macro_f1"] == pytest.approx(1.0)
    assert o["severity_mae"] == pytest.approx(0.0)
    assert set(res["by_site"]) == {"Guys", "IOP"}
    assert np.trace(np.array(res["type_confusion"])) == 8
