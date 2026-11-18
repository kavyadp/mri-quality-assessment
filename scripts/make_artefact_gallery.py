"""Render every artefact at every severity for one subject (used to tune levels).

Writes outputs/artefacts/fig_gallery.png (clean + 3 artefacts x 3 severities,
with head-masked SSIM/PSNR vs. clean) and fig_motion_kspace.png (what motion does to k-space).

Usage:
    python scripts/make_artefact_gallery.py --subject IXI002
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mriqa.artefacts import ARTEFACTS, SEVERITIES, apply_artefact, simulate_motion
from mriqa.data.ixi import foreground_mask, list_subjects, load_canonical, normalize_intensity, select_axial_slices
from mriqa.data.preprocess import resample_isotropic
from mriqa.metrics import masked_psnr, masked_ssim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw/IXI-T1"))
    ap.add_argument("--subject", default="IXI002")
    ap.add_argument("--out", type=Path, default=Path("outputs/artefacts"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    s = next(x for x in list_subjects(args.data) if x.key == args.subject)
    vol, img = load_canonical(s.path)
    t0 = time.perf_counter()
    vol = resample_isotropic(vol, img.header.get_zooms()[:3])
    mask = foreground_mask(vol)
    clean = normalize_intensity(vol, mask)
    keep = select_axial_slices(mask, spacing_z=1.0)
    z = int(keep[len(keep) // 3])  # lower third: ventricles and cortex, plenty of structure
    print(f"preprocess {time.perf_counter() - t0:.1f}s  shape={clean.shape}  slice z={z}")

    def view(v):
        return np.rot90(v[:, :, z])

    fig, axes = plt.subplots(len(ARTEFACTS), len(SEVERITIES) + 1, figsize=(13, 13))
    for r, art in enumerate(ARTEFACTS):
        axes[r, 0].imshow(view(clean), cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title("clean", fontsize=10)
        axes[r, 0].set_ylabel(art, fontsize=12)
        for c, sev in enumerate(SEVERITIES, start=1):
            t0 = time.perf_counter()
            deg, _ = apply_artefact(clean, art, sev, np.random.default_rng(args.seed))
            deg = normalize_intensity(deg, mask)  # same per-image normalisation a real scan would get
            dt = time.perf_counter() - t0
            a, b = clean[:, :, z], deg[:, :, z]
            ssim = masked_ssim(a, b, mask[:, :, z])  # inside the head only, as in the labels
            psnr = masked_psnr(a, b, mask[:, :, z])
            axes[r, c].imshow(view(deg), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(f"{art} L{sev}  SSIM {ssim:.3f}  PSNR {psnr:.1f}", fontsize=10)
            print(f"{art:6s} L{sev}: SSIM {ssim:.3f} PSNR {psnr:5.1f}  ({dt:.1f}s)")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{s.key} ({s.site}): simulated artefacts by severity (axial z={z}, 1 mm)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out / "fig_gallery.png", dpi=100)
    plt.close(fig)

    # K-space view of motion: which lines came from a moved head, and the result.
    rng = np.random.default_rng(args.seed)
    moved, events = simulate_motion(clean, 2, 3.0, 3.0, rng)
    k_clean = np.log1p(np.abs(np.fft.fftshift(np.fft.fftn(clean))))
    k_moved = np.log1p(np.abs(np.fft.fftshift(np.fft.fftn(moved))))
    kz = clean.shape[2] // 2
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(np.rot90(k_clean[:, :, kz]), cmap="magma")
    axes[0].set_title("k-space, clean (log magnitude)")
    axes[1].imshow(np.rot90(np.abs(k_moved - k_clean)[:, :, kz]), cmap="magma")
    for e in events:
        x = e.kspace_fraction * clean.shape[0]
        axes[1].axvline(x, color="cyan", lw=1, ls="--")
    axes[1].set_title("|difference|: k-space acquired in moved poses")
    axes[2].imshow(view(clean), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("image, clean")
    axes[3].imshow(view(normalize_intensity(moved, mask)), cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("image, 2 movements (ghosting along L-R)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out / "fig_motion_kspace.png", dpi=100)
    plt.close(fig)
    print(f"figures in {args.out}")


if __name__ == "__main__":
    main()
