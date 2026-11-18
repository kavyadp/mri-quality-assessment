"""Tests on a synthetic head volume, so they run without downloading IXI."""

import nibabel as nib
import numpy as np
import pytest

from mriqa.data.ixi import (
    foreground_mask,
    head_top_index,
    list_subjects,
    load_canonical,
    normalize_intensity,
    parse_ixi_filename,
    select_axial_slices,
)
from mriqa.data.splits import make_subject_splits


def _synthetic_head(shape=(64, 64, 48), seed=0):
    """Ellipsoid 'head' with a brighter inner 'brain' and background noise."""
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*(np.linspace(-1, 1, n) for n in shape), indexing="ij")
    r = x**2 + y**2 + (z / 0.8) ** 2
    vol = np.where(r < 0.8, 300.0, 0.0) + np.where(r < 0.5, 400.0, 0.0)
    return (vol + rng.normal(0, 10, shape).clip(0)).astype(np.float32)


@pytest.fixture
def ixi_dir(tmp_path):
    """Three fake subjects stored in a non-RAS (LPS) orientation."""
    lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    for i, site in [(2, "Guys"), (12, "HH"), (35, "IOP")]:
        img = nib.Nifti1Image(_synthetic_head(seed=i), lps)
        nib.save(img, tmp_path / f"IXI{i:03d}-{site}-0{i:03d}-T1.nii.gz")
    (tmp_path / "IXI999-Guys-0001-T2.nii.gz").write_bytes(b"")  # not T1: must be ignored
    return tmp_path


def test_parse_filename():
    assert parse_ixi_filename("IXI002-Guys-0828-T1.nii.gz") == {"subject_id": 2, "site": "Guys", "scan": "0828"}
    assert parse_ixi_filename("IXI002-Guys-0828-T2.nii.gz") is None
    assert parse_ixi_filename("../IXI002-Guys-0828-T1.nii.gz") is None


def test_list_subjects_ignores_non_t1(ixi_dir):
    subs = list_subjects(ixi_dir)
    assert [s.key for s in subs] == ["IXI002", "IXI012", "IXI035"]
    assert [s.site for s in subs] == ["Guys", "HH", "IOP"]


def test_load_canonical_reorients_to_ras(ixi_dir):
    s = list_subjects(ixi_dir)[0]
    vol, img = load_canonical(s.path)
    assert nib.aff2axcodes(img.affine) == ("R", "A", "S")
    raw = np.asarray(nib.load(str(s.path)).dataobj)
    np.testing.assert_allclose(vol, raw[::-1, ::-1, :])  # LPS -> RAS flips x and y


def test_mask_normalise_and_slice_selection():
    vol = _synthetic_head()
    mask = foreground_mask(vol)
    assert 0.1 < mask.mean() < 0.6
    norm = normalize_intensity(vol, mask)
    assert norm.min() >= 0 and norm.max() <= 1
    top = head_top_index(mask)
    assert top is not None and top < vol.shape[2] - 1
    keep = select_axial_slices(mask, spacing_z=1.0, top_offset_mm=3, depth_mm=10)
    assert len(keep) == 11
    assert keep[-1] == top - 3 and keep[0] == top - 13
    assert np.all(np.diff(keep) == 1), "kept slices should be one contiguous block"


def test_vertex_cut_off_returns_no_slices():
    vol = _synthetic_head()[:, :, :30]  # crop above the middle: head touches the top of the FOV
    mask = foreground_mask(vol)
    assert head_top_index(mask) is None
    assert len(select_axial_slices(mask, spacing_z=1.0)) == 0


def test_splits_are_disjoint_and_complete(ixi_dir):
    from mriqa.data.ixi import IXISubject
    from pathlib import Path

    subs = [IXISubject(i, ["Guys", "HH", "IOP"][i % 3], Path(f"x{i}")) for i in range(100)]
    sp = make_subject_splits(subs, seed=0)
    all_keys = sp["train"] + sp["val"] + sp["test"]
    assert len(all_keys) == len(set(all_keys)) == 100
    assert make_subject_splits(subs, seed=0) == sp, "same seed -> same split"
    assert 12 <= len(sp["test"]) <= 18
