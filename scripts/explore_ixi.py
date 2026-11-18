"""Stage 1: explore the IXI T1 data and make the subject-level splits.

Writes to outputs/exploration/:
  volume_summary.csv     one row per volume: site, shape, spacing, orientation,
                         intensity stats, usable axial slices, age/sex
  fig_orthoviews.png     one subject per site in axial/coronal/sagittal views
  fig_axial_selection.png head coverage per axial slice and the kept range
  fig_intensity_hist.png normalised foreground intensity histograms by site
and the split file data/splits.json.

Usage:
    python scripts/explore_ixi.py
    python scripts/explore_ixi.py --max-volumes 30    # faster first look
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from mriqa.data.ixi import (
    foreground_fraction_per_slice,
    foreground_mask,
    head_top_index,
    list_subjects,
    load_canonical,
    load_demographics,
    normalize_intensity,
    select_axial_slices,
)
from mriqa.data.splits import make_subject_splits, save_splits

SITE_COLORS = {"Guys": "#1f77b4", "HH": "#d62728", "IOP": "#2ca02c"}


def summarise(subjects, demographics) -> tuple[pd.DataFrame, dict]:
    rows, hist_by_site, cache, qc = [], {}, {}, []
    bins = np.linspace(0, 1, 101)
    for s in tqdm(subjects, desc="volumes"):
        try:
            raw = nib.load(str(s.path))
            vol, img = load_canonical(s.path)
        except Exception as e:  # corrupt / truncated download
            print(f"[warn] could not read {s.path.name}: {e}")
            continue
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        mask = foreground_mask(vol)
        norm = normalize_intensity(vol, mask)
        keep = select_axial_slices(mask, spacing_z=zooms[2])
        h, _ = np.histogram(norm[mask], bins=bins, density=True)
        hist_by_site.setdefault(s.site, []).append(h)
        if s.site not in cache:  # keep one example volume per site for figures
            cache[s.site] = (s, norm, mask, keep, zooms)
        if len(keep):  # lowest / middle / highest kept slice, for the QC montage
            picks = [keep[0], keep[len(keep) // 2], keep[-1]]
            qc.append((s, [norm[:, :, k] for k in picks], picks, zooms))

        row = {
            "subject": s.key, "site": s.site, "file": s.path.name,
            "orig_orientation": "".join(nib.aff2axcodes(raw.affine)),
            "shape_x": vol.shape[0], "shape_y": vol.shape[1], "shape_z": vol.shape[2],
            "spacing_x": round(float(zooms[0]), 3), "spacing_y": round(float(zooms[1]), 3),
            "spacing_z": round(float(zooms[2]), 3),
            "int_min": float(vol.min()), "int_p99": float(np.percentile(vol[mask], 99)),
            "int_max": float(vol.max()), "head_fraction": float(mask.mean()),
            "head_top": -1 if head_top_index(mask) is None else head_top_index(mask),
            "n_axial_kept": len(keep),
            "axial_first": int(keep[0]) if len(keep) else -1,
            "axial_last": int(keep[-1]) if len(keep) else -1,
        }
        if demographics is not None and s.subject_id in demographics.index:
            row.update(demographics.loc[s.subject_id].to_dict())
        rows.append(row)
    hists = {k: (bins, np.mean(v, axis=0)) for k, v in hist_by_site.items()}
    return pd.DataFrame(rows), {"hists": hists, "examples": cache, "qc": qc}


def show(ax, sl, spacing_a: float, spacing_b: float, title: str = "") -> None:
    """Show a 2D slice [a, b] with anterior/superior up and true physical proportions.

    rot90 puts axis b on the rows, so the pixel aspect (height/width) is spacing_b/spacing_a.
    """
    ax.imshow(np.rot90(sl), cmap="gray", vmin=0, vmax=1, aspect=spacing_b / spacing_a)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def fig_orthoviews(examples, out: Path) -> None:
    sites = [s for s in SITE_COLORS if s in examples]
    fig, axes = plt.subplots(len(sites), 3, figsize=(9, 3.1 * len(sites)), squeeze=False)
    for r, site in enumerate(sites):
        s, norm, _, _, (sx, sy, sz) = examples[site]
        x, y, z = (d // 2 for d in norm.shape)
        show(axes[r, 0], norm[:, :, z], sx, sy, f"{s.key} ({site}) - axial")
        show(axes[r, 1], norm[:, y, :], sx, sz, f"{s.key} ({site}) - coronal")
        show(axes[r, 2], norm[x, :, :], sy, sz, f"{s.key} ({site}) - sagittal")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_slice_range_qc(qc, out: Path, max_subjects: int = 16) -> None:
    """Lowest / middle / highest kept axial slice for many subjects: checks the
    selection rule never includes neck or eyes and never misses the cerebrum."""
    qc = qc[:max_subjects]
    cols = 4
    rows = int(np.ceil(len(qc) / cols))
    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 3 * 1.3, rows * 1.9), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (s, slices, picks, (sx, sy, _)) in enumerate(qc):
        r, c0 = divmod(i, cols)
        for j, (sl, k) in enumerate(zip(slices, picks)):
            label = ["low", "mid", "high"][j]
            show(axes[r, c0 * 3 + j], sl, sx, sy, f"{s.key} {label} z={k}" if j == 0 else f"{label} z={k}")
    fig.suptitle("Kept axial range per subject (lowest / middle / highest slice)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_axial_selection(examples, out: Path) -> None:
    site = next(iter(s for s in SITE_COLORS if s in examples))
    s, norm, mask, keep, (sx, sy, _) = examples[site]
    frac = foreground_fraction_per_slice(mask)
    picks = np.linspace(keep[0], keep[-1], 6).astype(int) if len(keep) else []

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(frac, color="k", lw=1)
    if len(keep):
        ax.axvspan(keep[0], keep[-1], color="tab:green", alpha=0.2, label=f"kept ({len(keep)} slices)")
    top = head_top_index(mask)
    if top is not None:
        ax.axvline(top, ls="--", color="gray", lw=0.8, label="top of head (vertex)")
    ax.set_xlabel("axial slice index (inferior -> superior)")
    ax.set_ylabel("head coverage")
    ax.set_title(f"{s.key} ({site}): which axial slices are usable")
    ax.legend(fontsize=8)
    for i, k in enumerate(picks):
        show(fig.add_subplot(2, len(picks), len(picks) + i + 1), norm[:, :, k], sx, sy, f"z={k}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_intensity_hist(hists, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for site, (bins, h) in sorted(hists.items()):
        centers = 0.5 * (bins[1:] + bins[:-1])
        ax.plot(centers, h, label=site, color=SITE_COLORS.get(site))
    ax.set_xlabel("normalised intensity (foreground)")
    ax.set_ylabel("density (mean over volumes)")
    ax.set_title("Intensity distributions differ by site/scanner")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/raw/IXI-T1"))
    ap.add_argument("--demographics", type=Path, default=Path("data/raw/IXI.xls"))
    ap.add_argument("--out", type=Path, default=Path("outputs/exploration"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--max-volumes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    subjects = list_subjects(args.data)
    if not subjects:
        raise SystemExit(f"No IXI T1 volumes in {args.data}. Run scripts/download_ixi.py first.")
    demo = load_demographics(args.demographics) if args.demographics.exists() else None

    # Splits use every subject on disk, independent of --max-volumes.
    splits = make_subject_splits(subjects, seed=args.seed)
    save_splits(splits, args.splits, args.seed)

    if args.max_volumes:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(subjects), size=min(args.max_volumes, len(subjects)), replace=False)
        subjects = [subjects[i] for i in sorted(idx)]

    args.out.mkdir(parents=True, exist_ok=True)
    df, extra = summarise(subjects, demo)
    df.to_csv(args.out / "volume_summary.csv", index=False)
    fig_orthoviews(extra["examples"], args.out / "fig_orthoviews.png")
    fig_axial_selection(extra["examples"], args.out / "fig_axial_selection.png")
    fig_intensity_hist(extra["hists"], args.out / "fig_intensity_hist.png")
    fig_slice_range_qc(extra["qc"], args.out / "fig_slice_range_qc.png")
    if (df["head_top"] < 0).any():
        print("[warn] vertex cut off (no slices kept) for: " + ", ".join(df.loc[df["head_top"] < 0, "subject"]))

    print(f"\n{len(df)} volumes analysed")
    print("\nVolumes per site:\n" + df["site"].value_counts().to_string())
    print("\nShape (x, y, z) after RAS reorientation:")
    print(df.groupby(["shape_x", "shape_y", "shape_z"]).size().to_string())
    print("\nVoxel spacing (mm):")
    print(df.groupby(["spacing_x", "spacing_y", "spacing_z"]).size().to_string())
    print("\nOriginal orientation codes:\n" + df["orig_orientation"].value_counts().to_string())
    print(f"\nUsable axial slices per volume: median {df['n_axial_kept'].median():.0f}, "
          f"range {df['n_axial_kept'].min()}-{df['n_axial_kept'].max()}")
    print(f"\nSplits ({args.splits}): " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print(f"Figures and CSV in {args.out}")


if __name__ == "__main__":
    main()
