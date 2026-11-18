"""Stage 6: Grad-CAM. Where does the CNN look, and does it look at the head?

Figures (outputs/gradcam/):
  fig_gradcam_ixi.png    one IXI test slice: clean, noise, blur, motion; heatmaps for the
                         quality head and for the predicted artefact type
  fig_gradcam_mrart.png  MR-ART: a still scan, a real-motion scan the CNN flags, and one it misses
Table (outputs/gradcam/cam_in_head.csv):
  share of Grad-CAM mass (quality head) inside the head, per artefact, compared with
  the share of the image the head occupies. A value near the head's area means no
  preference; well below it means the evidence is mostly in the background.

Usage:
    python scripts/make_gradcam.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mriqa.cnn import QualityNet
from mriqa.features import head_mask_2d
from mriqa.gradcam import GradCAM, fraction_in_mask
from mriqa.labels import TYPE_CLASSES, add_targets


def cams(cam: GradCAM, slices: np.ndarray, head: str, batch: int = 32) -> tuple[np.ndarray, dict]:
    maps, outs = [], []
    for i in range(0, len(slices), batch):
        x = torch.from_numpy(np.asarray(slices[i:i + batch], np.float32))[:, None].cuda()
        m, o = cam(x, head=head)
        maps.append(m)
        outs.append(o)
    out = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
    return np.concatenate(maps), out


def overlay(ax, img: np.ndarray, cam: np.ndarray | None, title: str) -> None:
    ax.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1)
    if cam is not None:
        ax.imshow(np.rot90(cam), cmap="jet", alpha=0.4, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--model", type=Path, default=Path("outputs/cnn/standard/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs/gradcam"))
    ap.add_argument("--n-per-group", type=int, default=300)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    model = QualityNet().cuda()
    model.load_state_dict(torch.load(args.model, weights_only=True))
    cam = GradCAM(model)

    packed = np.load(args.data / "packed.npy", mmap_mode="r")
    idx = add_targets(pd.read_csv(args.data / "packed_index.csv"))
    row_of = {(r.subject, r.artefact, r.slice): i for i, r in enumerate(idx.itertuples())}

    # ---- Figure 1: one IXI test slice under each artefact --------------------------------
    test = idx[idx.split == "test"]
    lv = test[test.artefact != "clean"].groupby("subject").severity.agg(lambda s: tuple(sorted(s)))
    subject = next(s for s in sorted(lv.index) if test[(test.subject == s) & (test.artefact == "motion")].severity.iloc[0] == 3)
    sl = 12  # lower-middle slice: ventricles visible
    rows = [row_of[(subject, a, sl)] for a in ("clean", "noise", "blur", "motion")]
    x = np.asarray(packed[rows], np.float32)
    q_maps, out = cams(cam, x, "quality")
    t_maps, _ = cams(cam, x, "type")
    p = torch.sigmoid(out["quality"]).numpy()
    t = out["type"].argmax(1).numpy()
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.6))
    for c, (r, a) in enumerate(zip(rows, ("clean", "noise", "blur", "motion"))):
        meta = idx.iloc[r]
        lvl = "" if a == "clean" else f" L{meta.severity}"
        overlay(axes[0, c], x[c], None, f"{a}{lvl}  (SSIM {meta.ssim:.3f})")
        overlay(axes[1, c], x[c], q_maps[c], f"quality head: p(degraded) {p[c]:.2f}")
        overlay(axes[2, c], x[c], t_maps[c], f"type head: predicted {TYPE_CLASSES[t[c]]}")
    fig.suptitle(f"Grad-CAM, IXI test subject {subject}, axial slice {sl} (red = evidence for the output)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out / "fig_gradcam_ixi.png", dpi=110)
    plt.close(fig)

    # ---- Table: share of evidence inside the head ----------------------------------------
    records = []
    for art in ("clean", "noise", "blur", "motion"):
        g = test[test.artefact == art]
        if art != "clean":
            g = g[g.degraded == 1]   # truly degraded slices only
        pick = rng.choice(g.index.to_numpy(), min(args.n_per_group, len(g)), replace=False)
        maps, _ = cams(cam, packed[pick], "quality")
        for i, m in zip(pick, maps):
            r = idx.loc[i]
            clean = np.asarray(packed[row_of[(r.subject, "clean", r.slice)]], np.float32)
            head = head_mask_2d(clean)
            records.append({"dataset": "IXI test", "group": art, "cam_in_head": fraction_in_mask(m, head),
                            "head_area": head.mean()})

    mr_idx = pd.read_csv(args.data / "mrart" / "index.csv")
    mr = np.load(args.data / "mrart" / "slices.npy", mmap_mode="r")
    for name, sel in (("still, score 1", (mr_idx.acq == "standard") & (mr_idx.score == 1)),
                      ("real motion, score 3", mr_idx.score == 3)):
        pick = rng.choice(mr_idx[sel].index.to_numpy(), args.n_per_group, replace=False)
        maps, _ = cams(cam, mr[pick], "quality")
        for i, m in zip(pick, maps):
            head = head_mask_2d(np.asarray(mr[i], np.float32))
            records.append({"dataset": "MR-ART", "group": name, "cam_in_head": fraction_in_mask(m, head),
                            "head_area": head.mean()})
    df = pd.DataFrame(records)
    summary = df.groupby(["dataset", "group"], sort=False)[["cam_in_head", "head_area"]].agg(["mean", "std"]).round(3)
    summary.to_csv(args.out / "cam_in_head.csv")
    print(summary.to_string())

    # ---- Figure 2: MR-ART still vs real motion (flagged and missed) ----------------------
    scans = pd.read_csv("outputs/mrart/scan_predictions.csv")
    scans = scans[scans.model == "CNN"]
    bad = scans[scans.score == 3].sort_values("p_degraded")
    picks = [("still (score 1)", scans[(scans.acq == "standard") & (scans.score == 1)].iloc[0]),
             ("real motion, flagged", bad.iloc[-1]),
             ("real motion, missed", bad.iloc[len(bad) // 4])]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for c, (label, s) in enumerate(picks):
        rows = mr_idx[(mr_idx.subject == s.subject) & (mr_idx.acq == s.acq)]
        i = rows.index[len(rows) // 3]
        x = np.asarray(mr[[i]], np.float32)
        qm, o = cams(cam, x, "quality")
        tm, _ = cams(cam, x, "type")
        pt = TYPE_CLASSES[int(o["type"].argmax(1)[0])]
        overlay(axes[0, c], x[0], None, f"{label}\n{s.subject}, {s.acq}, radiologist {int(s.score)}")
        overlay(axes[1, c], x[0], qm[0], f"quality head: p(degraded) {float(torch.sigmoid(o['quality'])[0]):.2f}")
        overlay(axes[2, c], x[0], tm[0], f"type head: predicted {pt}")
    fig.suptitle("Grad-CAM on real motion (MR-ART), CNN trained only on simulated artefacts")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out / "fig_gradcam_mrart.png", dpi=110)
    plt.close(fig)
    cam.remove()
    print(f"\nfigures in {args.out}")


if __name__ == "__main__":
    main()
