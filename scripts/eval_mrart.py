"""Stage 5: test on REAL motion artefacts (MR-ART), with no retraining.

MR-ART (OpenNeuro ds004173): 148 adults, each scanned three times with the
same Siemens Prisma 3T MPRAGE: still ("standard"), slight head motion
("headmotion1") and more head motion ("headmotion2"). A neuroradiologist
scored every scan: 1 good, 2 medium, 3 unusable. Nothing here was seen during
training: new scanner vendor, new sequence, real rather than simulated motion.

Both models trained on IXI (standard split, all three IXI sites) are applied
to every slice, processed by exactly the IXI pipeline. Slice predictions are
averaged into one score per scan, then compared with the radiologist score:

  detection    AUROC for score >= 2 (any artefact) and score 3 (unusable)
  agreement    Spearman correlation between scan score and radiologist score
  ordering     within each person: are scans ordered like the radiologist ordered them?
  type         does the type head call real motion "motion"?
  calibration  balanced accuracy at the default threshold, and after setting the
               threshold from k still scans of other people (as in Stage 4)
  reference    the same AUROC for standard MRIQC image-quality metrics

Outputs in outputs/mrart/: scan_predictions.csv, metrics.json, figures.
Slices are cached in data/processed/mrart/ (re-used by Grad-CAM).

Usage:
    python scripts/eval_mrart.py --workers 6
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from mriqa.baseline import fit_rf, merge_features, predict_rf
from mriqa.cnn import QualityNet, predict_array
from mriqa.data.preprocess import prepare_scan
from mriqa.features import FEATURE_NAMES, slice_features
from mriqa.labels import TYPE_CLASSES, add_targets

ACQ = {"standard": "still", "headmotion1": "slight motion", "headmotion2": "more motion"}
MOTION = TYPE_CLASSES.index("motion")
MRIQC_METRICS = ["cjv", "cnr", "efc", "fber", "snr_total", "qi_1", "qi_2", "wm2max", "fwhm_avg", "inu_range"]


def scan_table(root: Path) -> pd.DataFrame:
    rows = []
    for tsv in sorted(root.glob("sub-*/sub-*_scans.tsv")):
        sub = tsv.parent.name
        for _, r in pd.read_csv(tsv, sep="\t").iterrows():
            acq = r["filename"].split("_acq-")[1].split("_")[0]
            rows.append({"subject": sub, "acq": acq, "score": int(r["score"]),
                         "path": str(tsv.parent / r["filename"])})
    return pd.DataFrame(rows)


def process(path: str) -> dict | None:
    r = prepare_scan(path)
    if r is None:
        return None
    feats = np.stack([slice_features(s) for s in r["slices"]])
    return {"slices": r["slices"].astype(np.float16), "slice_z": r["slice_z"], "features": feats}


def load_or_build_cache(scans: pd.DataFrame, cache: Path, workers: int) -> tuple[np.ndarray, pd.DataFrame]:
    if (cache / "slices.npy").exists() and (cache / "index.csv").exists():
        return np.load(cache / "slices.npy", mmap_mode="r"), pd.read_csv(cache / "index.csv")
    cache.mkdir(parents=True, exist_ok=True)
    slices, rows = [], []
    with ProcessPoolExecutor(workers) as pool:
        for (_, s), r in tqdm(zip(scans.iterrows(), pool.map(process, scans.path, chunksize=2)),
                              total=len(scans), desc="MR-ART scans"):
            if r is None:
                print(f"skip {s.subject} {s.acq}: vertex outside field of view")
                continue
            slices.append(r["slices"])
            for i, (z, f) in enumerate(zip(r["slice_z"], r["features"])):
                rows.append({"subject": s.subject, "acq": s.acq, "score": s.score, "slice": i,
                             "slice_z": int(z), **dict(zip(FEATURE_NAMES, f))})
    arr = np.concatenate(slices)
    np.save(cache / "slices.npy", arr)
    idx = pd.DataFrame(rows)
    idx.to_csv(cache / "index.csv", index=False)
    return arr, idx


def scan_level(df: pd.DataFrame, model: str) -> pd.DataFrame:
    g = df.groupby(["subject", "acq", "score"])
    out = g.agg(p_degraded=("p_degraded", "mean"), frac_flagged=("flag", "mean"),
                sev_pred=("sev_pred", "mean"), p_motion=("p_motion", "mean")).reset_index()
    out["type_motion"] = g.apply(lambda x: x[[f"tp{j}" for j in range(4)]].mean().to_numpy().argmax() == MOTION
                                 ).to_numpy()
    out["model"] = model
    return out


def pairwise_order(scans: pd.DataFrame, col: str) -> float:
    """Within subject, over scan pairs the radiologist scored differently: fraction ordered the same way."""
    agree = total = 0
    for _, g in scans.groupby("subject"):
        v, s = g[col].to_numpy(), g["score"].to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if s[i] != s[j]:
                    total += 1
                    agree += (v[i] - v[j]) * (s[i] - s[j]) > 0
    return agree / total if total else float("nan")


def metrics_for(scans: pd.DataFrame, col: str, rng: np.random.Generator) -> dict:
    y_any, y_bad = (scans.score >= 2).to_numpy(), (scans.score == 3).to_numpy()
    v = scans[col].to_numpy()
    m = {"auroc_any_artefact": roc_auc_score(y_any, v), "auroc_unusable": roc_auc_score(y_bad, v),
         "spearman_vs_score": spearmanr(v, scans.score).statistic,
         "within_subject_pair_agreement": pairwise_order(scans, col)}
    if col == "p_degraded":
        flag = v >= 0.5
        m["bal_acc_default_threshold"] = balanced_accuracy_score(y_any, flag)
        m["specificity_default_threshold"] = float(1 - flag[~y_any].mean())
        m["sensitivity_default_threshold"] = float(flag[y_any].mean())
        subjects = scans.subject.unique()
        for k in [k for k in (3, 10) if k < len(subjects)]:
            ba = []
            for _ in range(200):
                cal = rng.choice(subjects, k, replace=False)
                ref = scans[scans.subject.isin(cal) & (scans.acq == "standard") & (scans.score == 1)][col]
                if ref.empty:
                    continue
                thr = np.quantile(ref, 0.95)
                ev = scans[~scans.subject.isin(cal)]
                ba.append(balanced_accuracy_score(ev.score >= 2, ev[col] > thr))
            m[f"bal_acc_calibrated_{k}_still_scans"] = float(np.mean(ba))
    if "type_motion" in scans:
        m["motion_typed_as_motion"] = float(scans.loc[y_any, "type_motion"].mean())
        m["still_typed_as_motion"] = float(scans.loc[~y_any, "type_motion"].mean())
    return {k: float(x) for k, x in m.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrart", type=Path, default=Path("data/raw/MR-ART"))
    ap.add_argument("--ixi", type=Path, default=Path("data/processed"))
    ap.add_argument("--cache", type=Path, default=Path("data/processed/mrart"))
    ap.add_argument("--cnn", type=Path, default=Path("outputs/cnn/standard/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs/mrart"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    scans = scan_table(args.mrart)
    print(f"{scans.subject.nunique()} subjects, {len(scans)} scans; scores:\n"
          f"{pd.crosstab(scans.acq.map(ACQ), scans.score).to_string()}")
    arr, idx = load_or_build_cache(scans, args.cache, args.workers)
    idx[list(FEATURE_NAMES)] = idx[list(FEATURE_NAMES)].fillna(idx[list(FEATURE_NAMES)].median())

    # Classical RF: the identical forests from Stage 4a (same data, same seed).
    ixi = merge_features(add_targets(pd.read_csv(args.ixi / "slices.csv")),
                         pd.read_csv("outputs/baseline/features.csv.gz"))
    rf = predict_rf(fit_rf(ixi[ixi.split == "train"], seed=0), idx)
    rf["flag"] = rf.p_degraded >= 0.5
    rf["p_motion"] = rf[f"type_prob_{MOTION}"]
    for j in range(4):
        rf[f"tp{j}"] = rf[f"type_prob_{j}"]

    # CNN trained on the standard IXI split.
    model = QualityNet().cuda()
    model.load_state_dict(torch.load(args.cnn, weights_only=True))
    p = predict_array(model, arr)
    cnn = idx.copy()
    cnn["p_degraded"], cnn["sev_pred"] = p["p_degraded"], p["sev_pred"]
    cnn["flag"] = cnn.p_degraded >= 0.5
    cnn["p_motion"] = p["type_prob"][:, MOTION]
    for j in range(4):
        cnn[f"tp{j}"] = p["type_prob"][:, j]

    per_scan = pd.concat([scan_level(rf, "classical RF"), scan_level(cnn, "CNN")])
    per_scan.to_csv(args.out / "scan_predictions.csv", index=False)

    results = {}
    for model_name, g in per_scan.groupby("model"):
        for col in ("p_degraded", "sev_pred", "p_motion"):
            results[f"{model_name} / {col}"] = metrics_for(g, col, rng)

    # MRIQC reference: widely used classical image-quality metrics, computed on the full 3D scan.
    mq_path = next(args.mrart.glob("derivatives/mriqc-*/group_T1w.tsv"), None)
    if mq_path is not None:
        mq = pd.read_csv(mq_path, sep="\t")
        mq["subject"] = mq.bids_name.str.split("_").str[0]
        mq["acq"] = mq.bids_name.str.split("_acq-").str[1].str.split("_").str[0]
        mq = mq.merge(scans[["subject", "acq", "score"]], on=["subject", "acq"])
        for metric in MRIQC_METRICS:
            r = metrics_for(mq.assign(v=mq[metric]), "v", rng)
            # Direction is not known a priori for every IQM; report the better one (optimistic for MRIQC).
            if r["auroc_any_artefact"] < 0.5:
                r = metrics_for(mq.assign(v=-mq[metric]), "v", rng)
            results[f"MRIQC / {metric}"] = r

    (args.out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = pd.DataFrame(results).T
    table.round(3).to_csv(args.out / "summary.csv")
    print(table.round(3).to_string())

    # Figure 1: scan score by radiologist score, both models.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (model_name, g) in zip(axes, per_scan.groupby("model", sort=False)):
        data = [g[g.score == s].p_degraded for s in (1, 2, 3)]
        ax.boxplot(data, widths=0.5, showfliers=False)
        for s, d in zip((1, 2, 3), data):
            ax.scatter(np.full(len(d), s) + rng.uniform(-0.15, 0.15, len(d)), d, s=8, alpha=0.4)
        auc = results[f"{model_name} / p_degraded"]["auroc_any_artefact"]
        ax.set_title(f"{model_name}: AUROC (score >= 2) {auc:.2f}")
        ax.set_xticks([1, 2, 3], ["1 good", "2 medium", "3 unusable"])
        ax.set_xlabel("neuroradiologist score")
    axes[0].set_ylabel("mean p(degraded) over slices")
    fig.suptitle("MR-ART real motion: model score vs. radiologist (no retraining)")
    fig.tight_layout()
    fig.savefig(args.out / "fig_score_vs_radiologist.png", dpi=120)
    plt.close(fig)

    # Figure 2: one subject's three scans with both models' scores.
    cnn_scans = per_scan[per_scan.model == "CNN"].set_index(["subject", "acq"])
    sub = next(s for s, g in scans.groupby("subject") if sorted(g.score) == [1, 2, 3])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, acq in zip(axes, ACQ):
        rows = idx[(idx.subject == sub) & (idx.acq == acq)]
        k = rows.index[len(rows) // 3]
        ax.imshow(np.rot90(np.asarray(arr[k], np.float32)), cmap="gray", vmin=0, vmax=1)
        r = cnn_scans.loc[(sub, acq)]
        ax.set_title(f"{ACQ[acq]} (radiologist {int(r.score)})\nCNN p(degraded) {r.p_degraded:.2f}, "
                     f"p(motion) {r.p_motion:.2f}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"MR-ART {sub}: same person, three acquisitions")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.out / "fig_example_subject.png", dpi=110)
    plt.close(fig)
    print(f"\noutputs in {args.out}")


if __name__ == "__main__":
    main()
