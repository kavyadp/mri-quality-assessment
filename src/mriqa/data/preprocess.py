"""Volume preprocessing shared by dataset generation and inference."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def resample_isotropic(vol: np.ndarray, spacing, target_mm: float = 1.0, order: int = 1) -> np.ndarray:
    """Resample a volume to isotropic voxels (default 1 mm)."""
    factors = np.asarray(spacing, dtype=float) / target_mm
    return ndimage.zoom(vol, factors, order=order).astype(np.float32)


def crop_window_2d(mask: np.ndarray, slices: np.ndarray, size: int) -> tuple[slice, slice]:
    """Fixed-size in-plane window centred on the head over the selected axial slices.

    One window per volume (not per slice), so all slices of a subject share
    the same field of view, as on a scanner.
    """
    proj = mask[:, :, slices].any(axis=2)
    xs, ys = np.nonzero(proj)
    cx = (xs.min() + xs.max()) // 2 if len(xs) else mask.shape[0] // 2
    cy = (ys.min() + ys.max()) // 2 if len(ys) else mask.shape[1] // 2
    return slice(cx - size // 2, cx - size // 2 + size), slice(cy - size // 2, cy - size // 2 + size)


def extract_slices(vol: np.ndarray, slices: np.ndarray, window: tuple[slice, slice]) -> np.ndarray:
    """Axial slices [S, size, size] cut with a window that may extend past the volume (zero padded)."""
    sx, sy = window
    size_x, size_y = sx.stop - sx.start, sy.stop - sy.start
    pad = max(0, -sx.start, -sy.start, sx.stop - vol.shape[0], sy.stop - vol.shape[1])
    if pad:
        vol = np.pad(vol, ((pad, pad), (pad, pad), (0, 0)))
        sx, sy = slice(sx.start + pad, sx.stop + pad), slice(sy.start + pad, sy.stop + pad)
    out = vol[sx, sy][:, :, slices]
    assert out.shape[:2] == (size_x, size_y)
    return np.moveaxis(out, 2, 0)


def prepare_scan(path, stride: int = 2, size: int = 224, top_offset_mm: float = 20.0,
                 depth_mm: float = 65.0) -> dict | None:
    """Full pipeline for one scan, as used to build the IXI dataset.

    RAS reorientation, 1 mm isotropic resampling, head mask, percentile
    normalisation, axial slices 20-85 mm below the vertex (every ``stride``-th),
    one fixed in-plane window. Returns None if the vertex is outside the field
    of view. Applying exactly this to new data (e.g. MR-ART) keeps evaluation
    consistent with training.
    """
    from .ixi import foreground_mask, load_canonical, normalize_intensity, select_axial_slices

    vol, img = load_canonical(path)
    vol = resample_isotropic(vol, img.header.get_zooms()[:3])
    mask = foreground_mask(vol)
    keep = select_axial_slices(mask, spacing_z=1.0, top_offset_mm=top_offset_mm, depth_mm=depth_mm)[::stride]
    if len(keep) == 0:
        return None
    norm = normalize_intensity(vol, mask)
    window = crop_window_2d(mask, keep, size)
    return {
        "volume": norm,
        "mask3d": mask,
        "slices": extract_slices(norm, keep, window),
        "mask": extract_slices(mask.astype(np.float32), keep, window) > 0.5,
        "slice_z": keep,
        "window": window,
    }
