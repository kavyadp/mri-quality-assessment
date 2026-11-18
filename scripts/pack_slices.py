"""Pack every slice into one memory-mapped array for fast random access during training.

Reading 578 compressed .npz files per batch would starve the GPU; one
uncompressed float16 array (~7.6 GB) on disk is read lazily by the data loader
and mostly served from the OS page cache.

Writes data/processed/packed.npy [N, H, W] float16, row i = row i of
data/processed/packed_index.csv (slices.csv in packing order).

Usage:
    python scripts/pack_slices.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    df = pd.read_csv(args.data / "slices.csv").sort_values(["subject", "artefact", "slice"]).reset_index(drop=True)
    first = np.load(next((args.data / "slices").glob("*.npz")))
    h, w = first["clean"].shape[1:]
    arr = np.lib.format.open_memmap(args.data / "packed.npy", mode="w+", dtype=np.float16, shape=(len(df), h, w))

    row = 0
    for subject, g in tqdm(df.groupby("subject", sort=True), desc="packing"):
        z = np.load(args.data / "slices" / f"{subject}.npz")
        for artefact, ga in g.groupby("artefact", sort=True):
            idx = ga["slice"].to_numpy()
            assert (idx == np.arange(len(idx))).all(), f"{subject} {artefact}: slices not contiguous"
            arr[row:row + len(idx)] = z[artefact][idx]
            row += len(idx)
    assert row == len(df)
    arr.flush()
    df.to_csv(args.data / "packed_index.csv", index=False)
    print(f"packed {len(df)} slices of {h}x{w} into {args.data / 'packed.npy'}")


if __name__ == "__main__":
    main()
