"""Download the MR-ART dataset (OpenNeuro ds004173, CC0) from the public OpenNeuro S3 bucket.

MR-ART: 148 healthy adults, each scanned three times with the same T1 MPRAGE
(Siemens Prisma 3T): standing still, slight head motion, more head motion.
Every scan has a neuroradiologist score (1 good, 2 medium, 3 unusable).

Only what this project needs is fetched (~4.9 GB): T1w NIfTI + JSON sidecars,
per-subject scans.tsv (the scores), participants.tsv, dataset metadata, and
the MRIQC group table (classical image-quality metrics per scan). The large
MRIQC HTML reports are skipped. Files keep their BIDS paths under --out.

Citation: Narai A. et al. Movement-related artefacts (MR-ART) dataset of
matched motion-corrupted and clean structural MRI brain scans. Sci Data 9, 630 (2022).

Usage:
    python scripts/download_mrart.py [--max-subjects 5]
"""

from __future__ import annotations

import argparse
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

BUCKET = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds004173"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
WANTED = re.compile(
    rf"^{DATASET}/("
    r"dataset_description\.json|README|participants\.(tsv|json)"
    r"|sub-\d+/sub-\d+_scans\.(tsv|json)"
    r"|sub-\d+/anat/sub-\d+_acq-(standard|headmotion1|headmotion2)_T1w\.(nii\.gz|json)"
    r"|derivatives/mriqc-[\d.]+/group_T1w\.tsv"
    r")$"
)


def list_objects(prefix: str) -> list[tuple[str, int]]:
    objs, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(q)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", NS):
            objs.append((c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text)))
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true":
            return objs
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)


def fetch(key: str, size: int, out: Path) -> int:
    dest = out / key.split("/", 1)[1]
    if dest.exists() and dest.stat().st_size == size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(f"{BUCKET}/{urllib.parse.quote(key)}", timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if tmp.stat().st_size != size:
        tmp.unlink()
        raise IOError(f"size mismatch for {key}")
    tmp.replace(dest)
    return size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/MR-ART"))
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    objs = [(k, s) for k, s in list_objects(f"{DATASET}/") if WANTED.match(k)]
    subjects = sorted({m.group(1) for k, _ in objs if (m := re.search(r"/(sub-\d+)/", k))})
    if args.max_subjects:
        keep = set(subjects[: args.max_subjects])
        objs = [(k, s) for k, s in objs if not (m := re.search(r"/(sub-\d+)/", k)) or m.group(1) in keep]
    total = sum(s for _, s in objs)
    print(f"{len(objs)} files, {total / 1e9:.2f} GB, {len(subjects)} subjects available")

    done = 0
    with ThreadPoolExecutor(args.threads) as pool, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for (k, s), n in zip(objs, pool.map(lambda o: fetch(o[0], o[1], args.out), objs)):
            bar.update(s)
            done += n
    n_t1 = len(list(args.out.glob("sub-*/anat/*_T1w.nii.gz")))
    print(f"[ok] downloaded {done / 1e9:.2f} GB new; {n_t1} T1w volumes in {args.out}")


if __name__ == "__main__":
    main()
