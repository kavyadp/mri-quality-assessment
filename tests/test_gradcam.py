import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mriqa.cnn import QualityNet  # noqa: E402
from mriqa.gradcam import GradCAM, fraction_in_mask  # noqa: E402


@pytest.mark.parametrize("head", ["quality", "severity", "type"])
def test_gradcam_shape_and_range(head):
    torch.manual_seed(0)
    cam = GradCAM(QualityNet())
    maps, out = cam(torch.rand(2, 1, 64, 64), head=head)
    cam.remove()
    assert maps.shape == (2, 64, 64)
    assert maps.min() >= 0 and maps.max() <= 1 + 1e-6
    assert out["type"].shape == (2, 4)


def test_gradcam_follows_the_evidence():
    """With a local encoder (small receptive field, no biases), evidence placed only on the
    left must produce a CAM only on the left. (ResNet-18's receptive field covers a whole
    small test image, so it cannot be used for this check.)"""
    torch.manual_seed(0)
    model = QualityNet()
    conv = torch.nn.Conv2d(1, 512, 3, padding=1, bias=False)
    model.encoder = torch.nn.Sequential(conv, torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    with torch.no_grad():
        conv.weight.abs_()
        model.quality_head.weight.abs_()
    cam = GradCAM(model, layer=conv)
    x = torch.zeros(1, 1, 64, 64)
    x[:, :, :, :24] = torch.rand(1, 1, 64, 24) + 0.1   # content only on the left
    maps, _ = cam(x, head="quality")
    cam.remove()
    left = np.zeros((64, 64), bool)
    left[:, :26] = True                               # content + 1-pixel receptive-field margin
    assert fraction_in_mask(maps[0], left) > 0.99


def test_fraction_in_mask():
    cam = np.zeros((4, 4))
    cam[0, 0], cam[3, 3] = 3.0, 1.0
    m = np.zeros((4, 4), bool)
    m[0, 0] = True
    assert fraction_in_mask(cam, m) == pytest.approx(0.75)
    assert np.isnan(fraction_in_mask(np.zeros((4, 4)), m))
