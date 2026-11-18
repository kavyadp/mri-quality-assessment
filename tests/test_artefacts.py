import numpy as np
import pytest

from mriqa.artefacts import (ARTEFACTS, SEVERITIES, add_rician_noise, apply_artefact, gaussian_blur,
                             rigid_transform, simulate_motion)
from mriqa.data.preprocess import crop_window_2d, extract_slices, resample_isotropic
from mriqa.metrics import masked_psnr, masked_ssim


def phantom(shape=(40, 48, 44)):
    """Ellipsoid 'head' with an inner darker ellipsoid, values in [0, 1]."""
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    r = x**2 / 0.7**2 + y**2 / 0.8**2 + z**2 / 0.75**2
    vol = np.where(r < 1, 0.8, 0.0) + np.where(r < 0.3, -0.4, 0.0)
    return vol.astype(np.float32)


def test_rician_noise_is_positive_and_biased_in_background():
    vol = phantom()
    out = add_rician_noise(vol, 0.05, np.random.default_rng(0))
    assert out.shape == vol.shape and out.dtype == np.float32
    assert out.min() >= 0
    bg = vol == 0
    # Rayleigh mean in background is sigma*sqrt(pi/2) ~ 0.063, not zero.
    assert abs(out[bg].mean() - 0.05 * np.sqrt(np.pi / 2)) < 0.005


def test_blur_preserves_mean_and_reduces_gradients():
    vol = phantom()
    out = gaussian_blur(vol, 1.5)
    assert np.isclose(out.mean(), vol.mean(), rtol=1e-3)
    assert np.abs(np.diff(out, axis=0)).max() < np.abs(np.diff(vol, axis=0)).max()


def test_rigid_transform_identity_and_translation():
    vol = phantom()
    assert np.allclose(rigid_transform(vol, (0, 0, 0), (0, 0, 0)), vol, atol=1e-5)
    shifted = rigid_transform(vol, (0, 0, 0), (3, 0, 0))
    assert np.allclose(shifted[13:30], vol[10:27], atol=1e-5)


def test_motion_without_movements_is_identity():
    vol = phantom()
    out, events = simulate_motion(vol, 0, 3.0, 3.0, np.random.default_rng(0))
    assert events == []
    assert np.allclose(out, vol, atol=1e-5)


def test_motion_keeps_centre_segment_as_reference():
    vol = phantom()
    out, events = simulate_motion(vol, 2, 3.0, 3.0, np.random.default_rng(1), centre_halfwidth=0.1)
    # n movements -> n+1 segments, one of which (the centre) is the reference.
    assert len(events) == 2
    # The head stays where it was at the k-space centre: no global shift.
    com0 = np.array(np.nonzero(vol > 0.4)).mean(axis=1)
    com1 = np.array(np.nonzero(out > 0.4)).mean(axis=1)
    assert np.linalg.norm(com0 - com1) < 1.0


@pytest.mark.parametrize("art", ARTEFACTS)
def test_apply_artefact_severity_ordering(art):
    vol = phantom()
    mask = vol > 0
    errs = []
    for sev in SEVERITIES:
        out, params = apply_artefact(vol, art, sev, np.random.default_rng(0))
        assert out.shape == vol.shape and isinstance(params, dict)
        errs.append(np.mean((out[mask] - vol[mask]) ** 2))
    assert errs[0] < errs[2]


def test_apply_artefact_rejects_unknown():
    with pytest.raises(ValueError):
        apply_artefact(phantom(), "spikes", 1, np.random.default_rng(0))


def test_masked_metrics():
    a = phantom()[:, :, 22]
    mask = a > 0
    assert masked_ssim(a, a, mask) == pytest.approx(1.0)
    assert masked_psnr(a, a, mask) == float("inf")
    assert np.isnan(masked_ssim(a, a, np.zeros_like(mask)))
    b = a.copy()
    b[~mask] = 0.5  # damage only outside the head: masked PSNR is unaffected
    assert masked_psnr(a, b, mask) == float("inf")


def test_resample_isotropic_shape():
    vol = np.zeros((10, 20, 20), np.float32)
    assert resample_isotropic(vol, (1.2, 0.9375, 0.9375)).shape == (12, 19, 19)


def test_crop_and_extract_pad_when_window_exceeds_volume():
    vol = phantom()
    mask = vol > 0
    slices = np.array([10, 20, 30])
    window = crop_window_2d(mask, slices, 64)  # larger than the 40x48 plane
    out = extract_slices(vol, slices, window)
    assert out.shape == (3, 64, 64)
    assert np.isclose(out.sum(), vol[:, :, slices].sum(), rtol=1e-6)  # nothing cropped, only padded
