"""Stage 4b: train the multi-task CNN and score it with the baseline's metrics.

Two experiments (--experiment):
  standard     train on the train split, select the epoch on val, report val + test
  unseen_site  train on Guys + HH subjects of the train split (val: Guys + HH of
               val), report on every IOP subject; the scanner is never seen

Outputs in outputs/cnn/<experiment>/: best.pt, history.csv, metrics.json,
predictions_<split>.csv, fig_curves.png.

Usage:
    python scripts/pack_slices.py                          # once
    python scripts/train_cnn.py --experiment standard
    python scripts/train_cnn.py --experiment unseen_site
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mriqa.cnn import QualityNet, SliceDataset, augment, multitask_loss
from mriqa.evaluation import evaluate, summary_table
from mriqa.labels import add_targets


@torch.no_grad()
def predict(model, loader, device) -> dict[str, np.ndarray]:
    model.eval()
    pq, pt, ps = [], [], []
    for x, *_ in loader:
        x = x.to(device, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.float16):
            # Test-time flip averaging: cheap and removes left/right bias.
            o1, o2 = model(x), model(x.flip(2))
        pq.append(((torch.sigmoid(o1["quality"].float()) + torch.sigmoid(o2["quality"].float())) / 2).cpu())
        pt.append(((o1["type"].float().softmax(1) + o2["type"].float().softmax(1)) / 2).cpu())
        ps.append(((o1["severity"].float() + o2["severity"].float()) / 2).clamp(0, 1).cpu())
    return {"p_degraded": torch.cat(pq).numpy(), "type_prob": torch.cat(pt).numpy(),
            "sev_pred": torch.cat(ps).numpy()}


def with_preds(table: pd.DataFrame, p: dict) -> pd.DataFrame:
    d = table.copy()
    d["p_degraded"] = p["p_degraded"]
    d["type_pred"] = p["type_prob"].argmax(1)
    d["sev_pred"] = p["sev_pred"]
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["standard", "unseen_site"], default="standard")
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("outputs/cnn"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None, help="stop each epoch early (smoke test)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    out = args.out / args.experiment
    out.mkdir(parents=True, exist_ok=True)

    table = add_targets(pd.read_csv(args.data / "packed_index.csv"))
    if args.experiment == "standard":
        train_t, val_t = table[table.split == "train"], table[table.split == "val"]
        evals = {"val": val_t, "test": table[table.split == "test"]}
    else:
        seen = table.site != "IOP"
        train_t, val_t = table[seen & (table.split == "train")], table[seen & (table.split == "val")]
        evals = {"val_seen_sites": val_t, "unseen_site_IOP": table[~seen]}
    print(f"train {len(train_t)} slices / {train_t.subject.nunique()} subjects, val {len(val_t)}")

    packed = args.data / "packed.npy"
    dl = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    train_dl = DataLoader(SliceDataset(packed, train_t), shuffle=True, drop_last=True, **dl)
    val_dl = DataLoader(SliceDataset(packed, val_t), shuffle=False, **dl)

    model = QualityNet().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = min(len(train_dl), args.max_steps or len(train_dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda")

    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n = time.perf_counter(), 0.0, 0
        for step, (x, yq, yt, ys) in enumerate(train_dl):
            if step >= steps_per_epoch:
                break
            x = augment(x.to(device, non_blocking=True).float()).contiguous(memory_format=torch.channels_last)
            yq, yt, ys = yq.to(device), yt.to(device), ys.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                o = model(x)
            loss, _ = multitask_loss(o, yq, yt, ys)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            n += 1

        # Validation loss on the same objective (no augmentation) picks the checkpoint.
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for x, yq, yt, ys in val_dl:
                with torch.autocast("cuda", dtype=torch.float16):
                    o = model(x.to(device).float())
                loss, _ = multitask_loss(o, yq.to(device), yt.to(device), ys.to(device))
                vl += loss.item() * len(x)
                vn += len(x)
        vl /= vn
        history.append({"epoch": epoch, "train_loss": running / n, "val_loss": vl,
                        "minutes": (time.perf_counter() - t0) / 60})
        flag = ""
        if vl < best:
            best = vl
            torch.save(model.state_dict(), out / "best.pt")
            flag = "  *"
        print(f"epoch {epoch:2d}  train {running / n:.4f}  val {vl:.4f}  "
              f"({history[-1]['minutes']:.1f} min){flag}", flush=True)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
    results = {}
    for name, t in evals.items():
        loader = DataLoader(SliceDataset(packed, t), shuffle=False, **dl)
        d = with_preds(t, predict(model, loader, device))
        results[f"cnn/{name}"] = evaluate(d)
        keep = ["subject", "site", "split", "artefact", "severity", "slice", "ssim", "degraded", "type_idx",
                "severity_target", "p_degraded", "type_pred", "sev_pred"]
        d[keep].to_csv(out / f"predictions_{name}.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(summary_table(results).round(3).T.to_string())

    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h.epoch, h.train_loss, label="train (augmented)")
    ax.plot(h.epoch, h.val_loss, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("multi-task loss")
    ax.set_title(f"CNN training: {args.experiment}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_curves.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
