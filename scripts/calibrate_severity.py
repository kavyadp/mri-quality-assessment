"""Check that artefact severity levels are well separated across subjects.

For several subjects and random seeds, applies every artefact at every level
and reports the masked SSIM/PSNR (mean over the kept axial slices). Good levels
increase monotonically, overlap little between neighbours, and give
comparable SSIM ranges across artefact types.

Usage:
    python scripts/calibrate_severity.py --n-subjects 6 --n-seeds 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from mriqa.artefacts import ARTEFACTS, SEVERITIES, apply_artefact
from mriqa.data.ixi import foreground_mask, list_subjects, load_canonical, normalize_intensity, select_axial_slices
from mriqa.data.preprocess import resample_isotropic
from mriqa.metrics import masked_psnr, masked_ssim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw/IXI-T1"))
    ap.add_argument("--n-subjects", type=int, default=6)
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--slice-stride", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("outputs/artefacts/severity_calibration.csv"))
    args = ap.parse_args()

    subjects = list_subjects(args.data)
    # Spread picks across sites when possible.
    by_site = {}
    for s in subjects:
        by_site.setdefault(s.site, []).append(s)
    picks = []
    while len(picks) < args.n_subjects and any(by_site.values()):
        for site in list(by_site):
            if by_site[site] and len(picks) < args.n_subjects:
                picks.append(by_site[site].pop(0))

    rows = []
    jobs = [(s, seed, a, v) for s in picks for seed in range(args.n_seeds) for a in ARTEFACTS for v in SEVERITIES]
    cache = {}
    for s, seed, art, sev in tqdm(jobs):
        if s.key not in cache:
            vol, img = load_canonical(s.path)
            vol = resample_isotropic(vol, img.header.get_zooms()[:3])
            mask = foreground_mask(vol)
            keep = select_axial_slices(mask, spacing_z=1.0)[:: args.slice_stride]
            cache = {s.key: (normalize_intensity(vol, mask), mask, keep)}  # one subject in memory at a time
        clean, mask, keep = cache[s.key]
        deg, _ = apply_artefact(clean, art, sev, np.random.default_rng(seed))
        deg = normalize_intensity(deg, mask)
        ssim = np.mean([masked_ssim(clean[:, :, k], deg[:, :, k], mask[:, :, k]) for k in keep])
        psnr = np.mean([masked_psnr(clean[:, :, k], deg[:, :, k], mask[:, :, k]) for k in keep])
        rows.append({"subject": s.key, "site": s.site, "seed": seed, "artefact": art,
                     "severity": sev, "ssim": ssim, "psnr": psnr})

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["artefact", "severity"])[["ssim", "psnr"]].agg(["mean", "min", "max"]).round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
