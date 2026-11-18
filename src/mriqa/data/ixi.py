"""Loading and basic preprocessing for IXI T1-weighted brain MRI.

Conventions used throughout the project:
  * Volumes are reoriented to RAS+ canonical orientation, so array axis 2 is
    always inferior -> superior and ``vol[:, :, k]`` is an axial slice.
  * Intensities are normalised per volume to [0, 1] using foreground
    percentiles (MRI intensities are arbitrary units and vary by scanner).
  * The foreground mask covers the whole head (IXI is not skull-stripped).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

# e.g. IXI002-Guys-0828-T1.nii.gz -> id=2, site=Guys
_IXI_RE = re.compile(r"^IXI(?P<id>\d{3})-(?P<site>[A-Za-z]+)-(?P<scan>\d{4})-T1\.nii\.gz$")

SITES = ("Guys", "HH", "IOP")  # Guy's Hospital (1.5T), Hammersmith (3T), Institute of Psychiatry (1.5T)


@dataclass(frozen=True)
class IXISubject:
    subject_id: int
    site: str
    path: Path

    @property
    def key(self) -> str:
        return f"IXI{self.subject_id:03d}"


def parse_ixi_filename(name: str) -> dict | None:
    m = _IXI_RE.match(name)
    if m is None:
        return None
    return {"subject_id": int(m["id"]), "site": m["site"], "scan": m["scan"]}


def list_subjects(root: str | Path) -> list[IXISubject]:
    """All IXI T1 volumes under ``root``, sorted by subject id."""
    subjects = []
    for p in sorted(Path(root).glob("IXI*-T1.nii.gz")):
        info = parse_ixi_filename(p.name)
        if info is not None:
            subjects.append(IXISubject(info["subject_id"], info["site"], p))
    return sorted(subjects, key=lambda s: s.subject_id)


def load_canonical(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load a NIfTI volume reoriented to RAS+. Returns (float32 array, image)."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asarray(img.dataobj, dtype=np.float32), img


def foreground_mask(vol: np.ndarray) -> np.ndarray:
    """Head mask: Otsu threshold, keep largest component, fill holes slice-wise."""
    mask = vol > threshold_otsu(vol)
    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    for k in range(mask.shape[2]):
        mask[:, :, k] = ndimage.binary_fill_holes(mask[:, :, k])
    return mask


def normalize_intensity(vol: np.ndarray, mask: np.ndarray | None = None,
                        lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
    """Clip to foreground percentiles and rescale to [0, 1]."""
    ref = vol[mask] if mask is not None and mask.any() else vol
    lo, hi = np.percentile(ref, [lower, upper])
    return np.clip((vol - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)


def foreground_fraction_per_slice(mask: np.ndarray) -> np.ndarray:
    """Fraction of each axial slice covered by the head mask."""
    return mask.mean(axis=(0, 1))


def head_top_index(mask: np.ndarray, min_rel_coverage: float = 0.02) -> int | None:
    """Index of the most superior axial slice that still contains head (the vertex).

    Returns None if the head touches the top of the field of view (vertex cut off).
    """
    frac = foreground_fraction_per_slice(mask)
    if frac.max() == 0:
        return None
    present = np.flatnonzero(frac >= min_rel_coverage * frac.max())
    top = int(present[-1])
    return None if top >= mask.shape[2] - 1 else top


def select_axial_slices(mask: np.ndarray, spacing_z: float,
                        top_offset_mm: float = 20.0, depth_mm: float = 65.0) -> np.ndarray:
    """Axial slices through the cerebrum, chosen by distance below the vertex.

    IXI's field of view extends to the neck, so head coverage alone keeps neck
    and face slices. Instead we anchor on the top of the head and keep slices
    from ``top_offset_mm`` to ``top_offset_mm + depth_mm`` below it.
    Defaults were tuned by eye on IXI adults (outputs/exploration/fig_slice_range_qc.png):
    less than ~20 mm below the vertex is mostly scalp and skull; more than ~85 mm
    reaches the orbits and posterior fossa.
    """
    top = head_top_index(mask)
    if top is None:
        return np.array([], dtype=int)
    hi = top - int(round(top_offset_mm / spacing_z))
    lo = top - int(round((top_offset_mm + depth_mm) / spacing_z))
    return np.arange(max(lo, 0), max(hi, 0) + 1)


def load_demographics(xls_path: str | Path):
    """IXI.xls -> DataFrame indexed by subject id with sex and age (if present)."""
    import pandas as pd

    df = pd.read_excel(xls_path)
    df = df.rename(columns={"IXI_ID": "subject_id", "SEX_ID (1=m, 2=f)": "sex_id", "AGE": "age"})
    if "sex_id" not in df.columns:  # column header varies between releases
        sex_col = next((c for c in df.columns if str(c).upper().startswith("SEX")), None)
        if sex_col is not None:
            df = df.rename(columns={sex_col: "sex_id"})
    keep = [c for c in ("subject_id", "sex_id", "age") if c in df.columns]
    df = df[keep].dropna(subset=["subject_id"]).drop_duplicates("subject_id")
    df["subject_id"] = df["subject_id"].astype(int)
    return df.set_index("subject_id")
