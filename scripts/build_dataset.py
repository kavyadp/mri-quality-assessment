"""Stage 2/3: build the slice dataset (clean + simulated artefacts) with quality labels.

For every subject:
  1. load, reorient to RAS, resample to 1 mm isotropic, head mask, normalise;
  2. keep axial slices 20-85 mm below the vertex (every ``--stride``-th);
  3. apply each artefact (noise, blur, motion) to the whole 3D volume at one
     severity, cycled across subjects so every level is equally common and each
     subject contributes all three artefact types;
  4. re-normalise each degraded volume with the clean head mask (as a real
     scan would be normalised), so intensity range is not a shortcut;
  5. cut one fixed in-plane window per subject and save the slices.

Outputs (in --out):
  slices/IXIxxx.npz   clean, noise, blur, motion: float16 [S, H, W]; mask: bool [S, H, W];
                      slice_z: int [S] (axial index in the 1 mm RAS volume)
  slices.csv          one row per (subject, version, slice): split, site, artefact,
                      severity (0 = clean), masked SSIM/PSNR vs. clean
  volumes.csv         one row per degraded volume with the simulation parameters
  excluded.csv        subjects with no usable slice range (vertex cut off)

Usage:
    python scripts/build_dataset.py --workers 6
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from mriqa.artefacts import ARTEFACTS, apply_artefact
from mriqa.data.ixi import (IXISubject, foreground_mask, head_top_index, list_subjects, load_canonical,
                            normalize_intensity, select_axial_slices)
from mriqa.data.preprocess import crop_window_2d, extract_slices, resample_isotropic
from mriqa.data.splits import load_splits
from mriqa.metrics import masked_psnr, masked_ssim


def severity_for(index: int, artefact_idx: int) -> int:
    """Cycle levels 1-3 over subjects, offset per artefact, so each subject gets a mix."""
    return (index + artefact_idx) % 3 + 1


def process_subject(subject: IXISubject, index: int, split: str, out_dir: Path, stride: int,
                    size: int, seed: int) -> dict:
    vol, img = load_canonical(subject.path)
    vol = resample_isotropic(vol, img.header.get_zooms()[:3])
    mask = foreground_mask(vol)
    keep = select_axial_slices(mask, spacing_z=1.0)[::stride]
    if len(keep) == 0:
        return {"excluded": {"subject": subject.key, "site": subject.site, "split": split,
                             "reason": "vertex cut off" if head_top_index(mask) is None else "no slices"}}
    clean = normalize_intensity(vol, mask)
    window = crop_window_2d(mask, keep, size)
    clean_sl = extract_slices(clean, keep, window)
    mask_sl = extract_slices(mask.astype(np.float32), keep, window) > 0.5
    arrays = {"clean": clean_sl.astype(np.float16), "mask": mask_sl, "slice_z": keep.astype(np.int16)}

    base = {"subject": subject.key, "site": subject.site, "split": split}
    slice_rows = [{**base, "artefact": "clean", "severity": 0, "slice": i, "slice_z": int(z),
                   "ssim": 1.0, "psnr": float("inf")} for i, z in enumerate(keep)]
    vol_rows = []
    for a_idx, art in enumerate(ARTEFACTS):
        sev = severity_for(index, a_idx)
        # Seed from (global seed, subject, artefact) so any subject can be regenerated alone.
        rng = np.random.default_rng([seed, subject.subject_id, a_idx])
        deg, params = apply_artefact(clean, art, sev, rng)
        deg = normalize_intensity(deg, mask)
        deg_sl = extract_slices(deg, keep, window)
        arrays[art] = deg_sl.astype(np.float16)
        for i, z in enumerate(keep):
            slice_rows.append({**base, "artefact": art, "severity": sev, "slice": i, "slice_z": int(z),
                               "ssim": masked_ssim(clean_sl[i], deg_sl[i], mask_sl[i]),
                               "psnr": masked_psnr(clean_sl[i], deg_sl[i], mask_sl[i])})
        vol_rows.append({**base, "artefact": art, "severity": sev, "params": json.dumps(params)})

    np.savez_compressed(out_dir / "slices" / f"{subject.key}.npz", **arrays)
    return {"slices": slice_rows, "volumes": vol_rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw/IXI-T1"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--stride", type=int, default=2, help="keep every n-th 1 mm axial slice")
    ap.add_argument("--size", type=int, default=224, help="in-plane crop (pixels = mm)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-subjects", type=int, default=None)
    args = ap.parse_args()
    (args.out / "slices").mkdir(parents=True, exist_ok=True)

    split_of = {k: name for name, keys in load_splits(args.splits).items() for k in keys}
    subjects = [s for s in list_subjects(args.data) if s.key in split_of]
    missing = len(list_subjects(args.data)) - len(subjects)
    if missing:
        print(f"warning: {missing} subjects are not in {args.splits}; re-run explore_ixi.py")
    subjects = subjects[: args.max_subjects]

    t0 = time.perf_counter()
    slices, volumes, excluded = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_subject, s, i, split_of[s.key], args.out, args.stride,
                               args.size, args.seed): s for i, s in enumerate(subjects)}
        for f in tqdm(as_completed(futures), total=len(futures), desc="subjects"):
            r = f.result()
            if "excluded" in r:
                excluded.append(r["excluded"])
            else:
                slices += r["slices"]
                volumes += r["volumes"]

    df = pd.DataFrame(slices).sort_values(["subject", "artefact", "slice"])
    df.to_csv(args.out / "slices.csv", index=False)
    pd.DataFrame(volumes).sort_values(["subject", "artefact"]).to_csv(args.out / "volumes.csv", index=False)
    pd.DataFrame(excluded, columns=["subject", "site", "split", "reason"]).to_csv(args.out / "excluded.csv",
                                                                                   index=False)

    print(f"\n{df.subject.nunique()} subjects, {len(df)} slice rows, {len(excluded)} excluded, "
          f"{(time.perf_counter() - t0) / 60:.1f} min")
    print(df.groupby("split").subject.nunique().to_string())
    deg = df[df.artefact != "clean"]
    print(deg.groupby(["artefact", "severity"]).ssim.describe()[["count", "mean", "min", "max"]].round(3)
          .to_string())


if __name__ == "__main__":
    main()
