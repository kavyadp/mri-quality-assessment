"""Physically motivated MRI artefact simulation on 3D volumes.

All functions take a clean volume normalised to roughly [0, 1] (1 mm isotropic,
RAS orientation) and return a degraded float32 volume of the same shape.

Why 3D: IXI T1 scans are acquired as a 3D sagittal volume, so k-space is 3D and
motion during the scan corrupts the whole volume. Simulating per 2D axial
slice would produce artefacts that cannot occur in this acquisition.

Artefacts
---------
noise   Rician noise. MR magnitude images are |complex signal + complex
        Gaussian noise|, so noise is Rician: signal-dependent, with a positive
        bias in dark regions (background is not zero-mean).
blur    Gaussian blur (loss of effective resolution, e.g. from heavy
        k-space filtering or interpolation).
motion  Rigid head motion during acquisition. K-space is filled over time;
        if the head moves, k-space lines acquired after each movement come
        from a shifted/rotated object. Mixing them produces the ghosting and
        ringing typical of real motion, not just blur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

ARTEFACTS = ("noise", "blur", "motion")
SEVERITIES = (1, 2, 3)

# Parameters per severity level. Level 1 should be subtle (borderline usable),
# level 3 clearly non-diagnostic. Tuned by eye with scripts/make_artefact_gallery.py.
NOISE_SIGMA = {1: 0.02, 2: 0.04, 3: 0.08}   # std of each Gaussian channel, image range ~[0, 1]
BLUR_SIGMA_MM = {1: 0.6, 2: 1.0, 3: 1.6}
MOTION = {  # (number of movements, max translation mm, max rotation deg, protected centre half-width)
    # L1 uses two small moves: a single move often lands far out in k-space
    # and leaves the image unchanged (SSIM ~0.999), which mislabels the slice.
    1: (2, 1.5, 1.5, 0.15),
    2: (2, 3.0, 3.0, 0.10),
    3: (4, 5.0, 5.0, 0.03),
}


def add_rician_noise(vol: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    real = vol + rng.normal(0.0, sigma, vol.shape)
    imag = rng.normal(0.0, sigma, vol.shape)
    return np.sqrt(real**2 + imag**2).astype(np.float32)


def gaussian_blur(vol: np.ndarray, sigma_mm: float, spacing_mm: float = 1.0) -> np.ndarray:
    return ndimage.gaussian_filter(vol, sigma=sigma_mm / spacing_mm).astype(np.float32)


def rigid_transform(vol: np.ndarray, rotation_deg, translation_vox) -> np.ndarray:
    """Rotate about the volume centre (Euler XYZ, degrees) and translate (voxels)."""
    ax, ay, az = np.deg2rad(rotation_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    rot = rz @ ry @ rx
    centre = (np.array(vol.shape) - 1) / 2.0
    # affine_transform maps output coords -> input coords, so use the inverse transform.
    inv = rot.T
    offset = centre - inv @ (centre + np.asarray(translation_vox, dtype=float))
    return ndimage.affine_transform(vol, inv, offset=offset, order=1, mode="constant", cval=0.0)


@dataclass
class MotionEvent:
    kspace_fraction: float      # start of the k-space segment acquired in this pose (0-1)
    rotation_deg: tuple
    translation_mm: tuple


def simulate_motion(vol: np.ndarray, n_movements: int, max_translation_mm: float,
                    max_rotation_deg: float, rng: np.random.Generator,
                    centre_halfwidth: float = 0.05, slow_axis: int = 0, spacing_mm: float = 1.0
                    ) -> tuple[np.ndarray, list[MotionEvent]]:
    """Rigid motion in 3D k-space.

    K-space is assumed to be filled sequentially along ``slow_axis`` (the
    outermost phase-encoding loop). That axis is split at ``n_movements``
    random times into segments; the head holds one pose per segment. The
    segment containing the k-space centre is the reference pose, and every
    other segment gets a random rigid pose. Each k-space segment is taken from
    the Fourier transform of the volume in the pose held while that segment
    was acquired, and the image is reconstructed as the magnitude of the
    inverse FFT.

    For IXI (sagittal 3D, stored PSR), slow_axis=0 (left-right in RAS) is the
    partition direction under this sequential-ordering assumption. It is a
    parameter because the true ordering of the scanner protocol is not
    published with the dataset.

    Damage depends strongly on how close to the k-space centre the motion
    happens, so movement times are drawn outside a protected window
    ``0.5 +/- centre_halfwidth``; a narrower window means more severe
    artefacts. Movements are also kept away from the outer edge (15-85 % of
    the acquisition), where they would have almost no visible effect.
    """
    n = vol.shape[slow_axis]
    if not 0 <= centre_halfwidth < 0.35:
        raise ValueError("centre_halfwidth must be in [0, 0.35)")
    side = rng.choice([-1, 1], n_movements)
    offset = rng.uniform(centre_halfwidth, 0.35, n_movements)
    times = np.sort(0.5 + side * offset)
    bounds = np.concatenate([[0], (times * n).astype(int), [n]])

    # The pose held while the k-space centre is acquired dominates image contrast
    # and position, so it is the reference (identity) pose. Without this, a move
    # before the centre just shifts the whole image, and SSIM vs. the clean scan
    # would measure misalignment rather than artefact.
    centre_seg = int(np.searchsorted(bounds, n // 2, side="right") - 1)

    kspace = np.fft.fftshift(np.fft.fftn(vol))
    events = []
    for i in range(len(bounds) - 1):
        if i == centre_seg:
            continue
        # Magnitudes in [0.5, 1] x max with random sign, so severity levels separate cleanly.
        rot = tuple(rng.uniform(0.5, 1.0, 3) * rng.choice([-1, 1], 3) * max_rotation_deg)
        trans = tuple(rng.uniform(0.5, 1.0, 3) * rng.choice([-1, 1], 3) * max_translation_mm)
        moved = rigid_transform(vol, rot, np.asarray(trans) / spacing_mm)
        k_moved = np.fft.fftshift(np.fft.fftn(moved))
        seg = [slice(None)] * vol.ndim
        seg[slow_axis] = slice(bounds[i], bounds[i + 1])
        kspace[tuple(seg)] = k_moved[tuple(seg)]
        events.append(MotionEvent(float(bounds[i] / n), rot, trans))
    out = np.abs(np.fft.ifftn(np.fft.ifftshift(kspace)))
    return out.astype(np.float32), events


def apply_artefact(vol: np.ndarray, artefact: str, severity: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Apply one artefact at one severity level. Returns (volume, parameters used)."""
    if artefact == "noise":
        s = NOISE_SIGMA[severity]
        return add_rician_noise(vol, s, rng), {"sigma": s}
    if artefact == "blur":
        s = BLUR_SIGMA_MM[severity]
        return gaussian_blur(vol, s), {"sigma_mm": s}
    if artefact == "motion":
        n, t, r, c = MOTION[severity]
        out, events = simulate_motion(vol, n, t, r, rng, centre_halfwidth=c)
        return out, {"n_movements": n, "events": [e.__dict__ for e in events]}
    raise ValueError(f"unknown artefact {artefact!r}; expected one of {ARTEFACTS}")
