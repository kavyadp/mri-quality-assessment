"""Classical no-reference image-quality features for one 2D slice.

These only look at the (possibly degraded) slice itself: no clean reference
and no mask from the clean scan, exactly as at deployment. The head mask is
re-estimated from the slice with Otsu thresholding.

Each feature targets a mechanism:
  sharpness  Laplacian variance, gradient magnitude, high-frequency energy -> blur
  noise      background statistics, residual noise estimate, SNR, entropy -> noise
  ghosting   signal outside the head along the phase-encode (L-R, image axis 0)
             direction vs. the readout direction, spectral anisotropy -> motion
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

FEATURE_NAMES = (
    "lap_var", "grad_mean", "grad_p90_p50", "hf_energy", "hf_anisotropy",
    "bg_mean", "bg_std", "bg_frac_bright", "noise_mad", "snr", "fg_std", "fg_entropy",
    "ghost_pe", "ghost_ratio", "tv",
)


def head_mask_2d(img: np.ndarray) -> np.ndarray:
    if img.max() <= img.min():
        return np.zeros(img.shape, bool)
    # Low threshold + hole filling so the scalp ring and the dark skull/CSF inside it
    # count as head; otherwise they would inflate every background statistic.
    m = img > 0.5 * threshold_otsu(img)
    m = ndimage.binary_closing(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)


def _radial_energy(img: np.ndarray) -> tuple[float, float]:
    """Fraction of spectral energy above 1/4 Nyquist, and its axis-0 / axis-1 ratio."""
    f = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    ky, kx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(img.shape[1])),
                         np.fft.fftshift(np.fft.fftfreq(img.shape[0])))
    r = np.sqrt(kx**2 + ky**2)
    total = f.sum() + 1e-12
    hf = f[r > 0.125].sum() / total
    # High frequency along axis 0 (phase-encode, where motion ghosts spread) vs axis 1.
    hf0 = f[(np.abs(kx) > 0.125) & (np.abs(ky) < 0.05)].sum()
    hf1 = f[(np.abs(ky) > 0.125) & (np.abs(kx) < 0.05)].sum()
    return float(hf), float(np.log((hf0 + 1e-12) / (hf1 + 1e-12)))


def slice_features(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    mask = head_mask_2d(img)
    if mask.sum() < 100:
        return np.full(len(FEATURE_NAMES), np.nan, np.float32)
    inner = ndimage.binary_erosion(mask, iterations=3)
    if inner.sum() < 50:
        inner = mask
    bg = ~ndimage.binary_dilation(mask, iterations=6)

    lap = ndimage.laplace(img)
    gx, gy = ndimage.sobel(img, 0), ndimage.sobel(img, 1)
    grad = np.hypot(gx, gy)[inner]
    hf, hf_aniso = _radial_energy(img)

    fg = img[inner]
    bgv = img[bg] if bg.any() else np.zeros(1, np.float32)
    # Residual after a 3x3 median: dominated by noise in flat regions (robust MAD estimate).
    resid = (img - ndimage.median_filter(img, 3))[inner]
    noise_mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    hist, _ = np.histogram(fg, bins=64, range=(0, 1))
    p = hist[hist > 0] / hist.sum()

    # Ghosting: background beside the head along axis 0 (phase-encode) vs along axis 1.
    xs, ys = np.nonzero(mask)
    band_pe = np.zeros_like(bg)
    band_ro = np.zeros_like(bg)
    band_pe[:, ys.min():ys.max() + 1] = True   # same y-range as head -> left/right of it
    band_ro[xs.min():xs.max() + 1, :] = True   # same x-range as head -> front/back of it
    pe = img[bg & band_pe]
    ro = img[bg & band_ro]
    ghost_pe = float(pe.mean()) if pe.size else 0.0
    ghost_ro = float(ro.mean()) if ro.size else 0.0

    feats = [
        float(lap[inner].var()),
        float(grad.mean()),
        float(np.percentile(grad, 90) / (np.percentile(grad, 50) + 1e-6)),
        hf,
        hf_aniso,
        float(bgv.mean()),
        float(bgv.std()),
        float((bgv > 0.05).mean()),
        float(noise_mad),
        float(fg.mean() / (bgv.std() + 1e-3)),
        float(fg.std()),
        float(-(p * np.log2(p)).sum()),
        ghost_pe,
        float(np.log((ghost_pe + 1e-4) / (ghost_ro + 1e-4))),
        float((np.abs(gx) + np.abs(gy))[inner].mean()),
    ]
    return np.asarray(feats, np.float32)
