"""Grad-CAM for QualityNet: which image regions drive a given head's output.

Grad-CAM (Selvaraju et al., ICCV 2017) weights the last convolutional feature
maps (ResNet layer4, 7x7 for a 224x224 input) by the spatially averaged gradient
of the chosen output, sums them, keeps the positive part, and upsamples to the
input size. The map is coarse (32-pixel cells) and shows evidence *for* the
output, not a segmentation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .cnn import QualityNet


class GradCAM:
    def __init__(self, model: QualityNet, layer: torch.nn.Module | None = None):
        self.model = model.eval()
        self.layer = layer if layer is not None else model.encoder.layer4
        self._acts = self._grads = None
        # Tensor hook on the layer output (module backward hooks clash with ResNet's in-place ReLU).
        self._h = self.layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output) -> None:
        self._acts = output
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "_grads", g))

    def remove(self) -> None:
        self._h.remove()

    def __call__(self, x: torch.Tensor, head: str = "quality", target: int | torch.Tensor | None = None
                 ) -> tuple[np.ndarray, dict]:
        """x: [B, 1, H, W]. head: 'quality' | 'severity' | 'type'. For 'type', target is the class
        (int, or one per image); default = predicted class. Returns CAMs [B, H, W] in [0, 1] and outputs."""
        x = x.detach().float().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        if head == "type":
            if target is None:
                target = out["type"].argmax(1)
            target = torch.as_tensor(target, device=x.device).expand(len(x))
            score = out["type"].gather(1, target[:, None])[:, 0]
        else:
            score = out[head]
        score.sum().backward()
        w = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.flatten(1).amax(1).clamp(min=1e-8)[:, None, None]
        return cam.detach().cpu().numpy(), {k: v.detach().cpu() for k, v in out.items()}


def fraction_in_mask(cam: np.ndarray, mask: np.ndarray) -> float:
    """Share of the CAM's total mass that lies inside the mask (e.g. the head)."""
    total = cam.sum()
    return float(cam[mask].sum() / total) if total > 0 else float("nan")
