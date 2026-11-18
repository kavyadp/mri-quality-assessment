"""Multi-task CNN for slice quality: one shared encoder, three heads.

  quality   1 logit   degraded (SSIM < 0.95) vs acceptable
  type      4 logits  clean / noise / blur / motion
  severity  1 value   1 - SSIM (measured damage)

The encoder is a ResNet-18 adapted to single-channel input and trained from
scratch (ImageNet colour statistics do not transfer cleanly to MR magnitude
images, and 50k slices is enough to train it).

Augmentation is restricted to operations that do not create or remove
artefacts, because the labels describe the image as stored: flips, integer
shifts (no interpolation, which would blur), and monotonic intensity changes
(gamma, contrast) that mimic differences between scanners.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import resnet18

from .labels import TYPE_CLASSES


class SliceDataset(Dataset):
    """Rows of the packed array selected by ``rows`` (indices into packed_index.csv)."""

    def __init__(self, packed: Path, table: pd.DataFrame):
        self.packed = Path(packed)
        self.rows = table.index.to_numpy()
        self.degraded = table["degraded"].to_numpy(np.float32)
        self.type_idx = table["type_idx"].to_numpy(np.int64)
        self.severity = table["severity_target"].to_numpy(np.float32)
        self._arr = None  # opened lazily so each DataLoader worker gets its own handle

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        if self._arr is None:
            self._arr = np.load(self.packed, mmap_mode="r")
        x = torch.from_numpy(np.array(self._arr[self.rows[i]], dtype=np.float16))[None]
        return x, self.degraded[i], self.type_idx[i], self.severity[i]


def augment(x: torch.Tensor, max_shift: int = 8) -> torch.Tensor:
    """Label-preserving augmentation on a GPU batch [B, 1, H, W] in [0, ~1]."""
    b = x.shape[0]
    dev = x.device
    flip_lr = torch.rand(b, device=dev) < 0.5
    x = torch.where(flip_lr[:, None, None, None], x.flip(2), x)
    flip_ap = torch.rand(b, device=dev) < 0.5
    x = torch.where(flip_ap[:, None, None, None], x.flip(3), x)
    sx, sy = np.random.randint(-max_shift, max_shift + 1, 2)
    x = torch.roll(x, shifts=(int(sx), int(sy)), dims=(2, 3))
    gamma = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.7, 1.4)
    scale = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.85, 1.15)
    return (x.clamp(min=0) ** gamma) * scale


class QualityNet(nn.Module):
    def __init__(self, width_dropout: float = 0.2):
        super().__init__()
        enc = resnet18(weights=None)
        enc.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        enc.fc = nn.Identity()
        self.encoder = enc
        self.drop = nn.Dropout(width_dropout)
        # Named *_head: nn.Module already has a .type() method that must not be shadowed.
        self.quality_head = nn.Linear(512, 1)
        self.type_head = nn.Linear(512, len(TYPE_CLASSES))
        self.severity_head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.drop(self.encoder(x))
        return {"quality": self.quality_head(h)[:, 0], "type": self.type_head(h),
                "severity": self.severity_head(h)[:, 0]}


def multitask_loss(out: dict, degraded, type_idx, severity, sev_weight: float = 10.0) -> tuple:
    lq = F.binary_cross_entropy_with_logits(out["quality"].float(), degraded)
    lt = F.cross_entropy(out["type"].float(), type_idx)
    ls = F.smooth_l1_loss(out["severity"].float(), severity, beta=0.02)
    return lq + lt + sev_weight * ls, {"quality": lq.item(), "type": lt.item(), "severity": ls.item()}


@torch.no_grad()
def predict_array(model: QualityNet, slices: np.ndarray, device: str = "cuda", batch: int = 64) -> dict:
    """Predict a stack of slices [N, H, W] with left-right flip averaging."""
    model.eval()
    pq, pt, ps = [], [], []
    for i in range(0, len(slices), batch):
        x = torch.from_numpy(np.asarray(slices[i:i + batch], dtype=np.float32))[:, None].to(device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.float16,
                            enabled=device.startswith("cuda")):
            o1, o2 = model(x), model(x.flip(2))
        pq.append(((torch.sigmoid(o1["quality"].float()) + torch.sigmoid(o2["quality"].float())) / 2).cpu())
        pt.append(((o1["type"].float().softmax(1) + o2["type"].float().softmax(1)) / 2).cpu())
        ps.append(((o1["severity"].float() + o2["severity"].float()) / 2).clamp(0, 1).cpu())
    return {"p_degraded": torch.cat(pq).numpy(), "type_prob": torch.cat(pt).numpy(),
            "sev_pred": torch.cat(ps).numpy()}
