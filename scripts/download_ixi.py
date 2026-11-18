"""Download IXI T1-weighted brain MRI volumes.

The full IXI-T1.tar archive is ~4.8 GB (~580 subjects). This script streams the
archive and extracts members as they arrive, so ``--max-subjects`` lets you stop
early and download only part of it (useful for a first look).

Only files named like ``IXI002-Guys-0828-T1.nii.gz`` are written, flattened into
the output folder; paths inside the archive are never used, so a malformed
archive cannot write outside ``--out``.

Usage:
    python scripts/download_ixi.py --max-subjects 50     # quick start
    python scripts/download_ixi.py                       # full dataset

Dataset: https://brain-development.org/ixi-dataset/  (CC BY-SA 3.0)
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

BASE_URL = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI"
T1_ARCHIVE = "IXI-T1.tar"
DEMOGRAPHICS = "IXI.xls"
USER_AGENT = "mriqa-downloader/0.1"


def _open(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=60)


def download_demographics(out_dir: Path) -> None:
    dest = out_dir / DEMOGRAPHICS
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    with _open(f"{BASE_URL}/{DEMOGRAPHICS}") as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"[ok] {dest}")


class _ProgressReader:
    """File-like wrapper that updates a tqdm bar as bytes are read."""

    def __init__(self, raw, bar: tqdm):
        self.raw, self.bar = raw, bar

    def read(self, n: int = -1) -> bytes:
        chunk = self.raw.read(n)
        self.bar.update(len(chunk))
        return chunk


def download_t1(out_dir: Path, max_subjects: int | None) -> int:
    from mriqa.data.ixi import parse_ixi_filename  # local import keeps --help fast

    existing = {p.name for p in out_dir.glob("IXI*-T1.nii.gz")}
    written = 0
    with _open(f"{BASE_URL}/{T1_ARCHIVE}") as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=T1_ARCHIVE) as bar:
            # "r|" = sequential stream mode: no seeking, so we can stop early.
            with tarfile.open(fileobj=_ProgressReader(resp, bar), mode="r|") as tar:
                for member in tar:
                    if max_subjects is not None and len(existing) >= max_subjects:
                        break
                    name = Path(member.name).name
                    if not member.isfile() or parse_ixi_filename(name) is None:
                        continue
                    if name in existing:
                        continue
                    src = tar.extractfile(member)
                    tmp = out_dir / (name + ".part")
                    with open(tmp, "wb") as f:
                        shutil.copyfileobj(src, f)
                    tmp.replace(out_dir / name)
                    existing.add(name)
                    written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/raw/IXI-T1"))
    ap.add_argument("--max-subjects", type=int, default=None,
                    help="stop once this many T1 volumes are on disk (default: all)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    download_demographics(args.out.parent)
    n = download_t1(args.out, args.max_subjects)
    total = len(list(args.out.glob("IXI*-T1.nii.gz")))
    print(f"[ok] wrote {n} new volumes; {total} T1 volumes now in {args.out}")


if __name__ == "__main__":
    main()
