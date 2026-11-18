"""Full-reference image-quality metrics, restricted to the head.

Whole-slice SSIM/PSNR is dominated by the background (most of the slice is
air), so noise in empty space would count as much as damage to the brain.
Both metrics here are computed only over head-mask pixels.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def masked_ssim(ref: np.ndarray, img: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    if not mask.any():
        return float("nan")
    _, ssim_map = structural_similarity(ref, img, data_range=data_range, full=True)
    return float(ssim_map[mask].mean())


def masked_psnr(ref: np.ndarray, img: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    if not mask.any():
        return float("nan")
    mse = float(np.mean((ref[mask] - img[mask]) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(data_range**2 / mse)
