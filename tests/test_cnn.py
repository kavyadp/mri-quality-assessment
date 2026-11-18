import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from mriqa.cnn import QualityNet, SliceDataset, augment, multitask_loss  # noqa: E402
from mriqa.labels import add_targets  # noqa: E402


def test_dataset_reads_selected_rows(tmp_path):
    arr = np.arange(6 * 8 * 8, dtype=np.float16).reshape(6, 8, 8)
    np.save(tmp_path / "packed.npy", arr)
    table = add_targets(pd.DataFrame({"subject": list("aabbcc"), "artefact": ["clean", "noise"] * 3,
                                      "ssim": [1.0, 0.9] * 3}))
    ds = SliceDataset(tmp_path / "packed.npy", table.iloc[[1, 4]])  # keeps original row index
    x, deg, typ, sev = ds[0]
    assert len(ds) == 2 and x.shape == (1, 8, 8)
    assert torch.equal(x[0], torch.from_numpy(arr[1]))
    assert deg == 1.0 and typ == 1 and sev == pytest.approx(0.1)
    assert torch.equal(ds[1][0][0], torch.from_numpy(arr[4]))


def test_augment_preserves_shape_and_no_blur():
    x = torch.zeros(4, 1, 32, 32)
    x[:, :, 10:20, 10:20] = 1.0
    y = augment(x, max_shift=2)
    assert y.shape == x.shape
    # Only flips, integer shifts and monotonic intensity maps: still two intensity levels per image.
    for img in y:
        assert len(torch.unique(img)) == 2


def test_model_outputs_and_loss():
    model = QualityNet()
    out = model(torch.rand(2, 1, 64, 64))
    assert out["quality"].shape == (2,) and out["type"].shape == (2, 4) and out["severity"].shape == (2,)
    loss, parts = multitask_loss(out, torch.tensor([0.0, 1.0]), torch.tensor([0, 3]), torch.tensor([0.0, 0.3]))
    assert torch.isfinite(loss) and set(parts) == {"quality", "type", "severity"}
    loss.backward()
