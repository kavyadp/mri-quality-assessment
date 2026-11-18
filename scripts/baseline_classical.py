"""Stage 4a: classical baseline. Hand-crafted no-reference features + random forests.

Three models on the same 15 features (see src/mriqa/features.py):
  quality  RandomForestClassifier, degraded (SSIM < 0.95) vs acceptable
  type     RandomForestClassifier, clean / noise / blur / motion
  severity RandomForestRegressor, 1 - SSIM

Trained on the train split only; val and test are reported. Also reports how
well each feature alone separates degraded from acceptable slices (AUROC), i.e.
what a single classical metric with a threshold would achieve.

Outputs in outputs/baseline/: features.csv.gz (cache), metrics.json,
summary.csv, test_predictions.csv, fig_confusion.png, fig_importance.png,
fig_severity.png.

Usage:
    python scripts/baseline_classical.py --workers 10
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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from mriqa.baseline import fit_rf, merge_features, predict_rf
from mriqa.evaluation import evaluate, summary_table
from mriqa.features import FEATURE_NAMES, slice_features
from mriqa.labels import TYPE_CLASSES, add_targets

VERSIONS = ("clean", "noise", "blur", "motion")


def subject_features(path: Path) -> pd.DataFrame:
    z = np.load(path)
    rows = []
    for art in VERSIONS:
        for i, sl in enumerate(z[art]):
            rows.append({"subject": path.stem, "artefact": art, "slice": i,
                         **dict(zip(FEATURE_NAMES, slice_features(sl)))})
    return pd.DataFrame(rows)


def load_features(data: Path, cache: Path, workers: int) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache)
    paths = sorted((data / "slices").glob("*.npz"))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(tqdm(pool.map(subject_features, paths, chunksize=4), total=len(paths), desc="features"))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(cache, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("outputs/baseline"))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    labels = add_targets(pd.read_csv(args.data / "slices.csv"))
    feats = load_features(args.data, args.out / "features.csv.gz", args.workers)
    df = merge_features(labels, feats)
    print(f"{len(df)} slices")

    train = df[df.split == "train"]
    models = fit_rf(train, args.seed)

    results, preds = {}, {}
    for split in ("val", "test"):
        d = predict_rf(models, df[df.split == split])
        results[f"classical_rf/{split}"] = evaluate(d)
        preds[split] = d

    # Unseen scanner: train on Guys + HH (train split only), test on every IOP subject.
    # Random splits share scanners between train and test; this does not.
    d = predict_rf(fit_rf(train[train.site != "IOP"], args.seed), df[df.site == "IOP"])
    results["classical_rf/unseen_site_IOP"] = evaluate(d)
    preds["unseen_site_IOP"] = d

    # Single-feature AUROC on test: a classical metric with one threshold.
    t = preds["test"]
    single = {f: float(max(a, 1 - a)) for f in FEATURE_NAMES
              for a in [roc_auc_score(t["degraded"], t[f])]}
    results["single_feature_test_auroc"] = dict(sorted(single.items(), key=lambda kv: -kv[1]))

    (args.out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summ = summary_table({k: v for k, v in results.items() if k.startswith("classical_rf")})
    summ.round(4).to_csv(args.out / "summary.csv")
    keep = ["subject", "site", "split", "artefact", "severity", "slice", "ssim", "degraded", "type_idx",
            "severity_target", "p_degraded", "type_pred", "sev_pred"]
    t[keep].to_csv(args.out / "test_predictions.csv", index=False)
    preds["unseen_site_IOP"][keep].to_csv(args.out / "predictions_unseen_site_IOP.csv", index=False)

    print(summ.round(3).T.to_string())
    print("\nTest by site:")
    print(pd.DataFrame(results["classical_rf/test"]["by_site"]).T[
        ["quality_auroc", "quality_bal_acc", "type_macro_f1", "severity_mae"]].round(3).to_string())
    print("\nTest by artefact and level:")
    print(pd.DataFrame(results["classical_rf/test"]["by_artefact_level"]).T.round(3).to_string())
    print("\nBest single features (test AUROC):")
    for f, a in list(results["single_feature_test_auroc"].items())[:5]:
        print(f"  {f:15s} {a:.3f}")

    # Figures
    cm = np.array(results["classical_rf/test"]["type_confusion"], float)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(TYPE_CLASSES)):
        for j in range(len(TYPE_CLASSES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(4), TYPE_CLASSES)
    ax.set_yticks(range(4), TYPE_CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("simulated")
    ax.set_title("Classical RF: artefact type (test, row-normalised)")
    fig.tight_layout()
    fig.savefig(args.out / "fig_confusion.png", dpi=120)
    plt.close(fig)

    imp = pd.Series(models["quality"].feature_importances_, FEATURE_NAMES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax, color="#4c72b0")
    ax.set_title("Feature importance, quality model (impurity)")
    fig.tight_layout()
    fig.savefig(args.out / "fig_importance.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharex=True, sharey=True)
    for ax, art in zip(axes, VERSIONS[1:]):
        g = t[t.artefact == art]
        ax.hexbin(1 - g["severity_target"], 1 - g["sev_pred"], gridsize=40, extent=(0.3, 1, 0.3, 1),
                  mincnt=1, cmap="viridis", bins="log")
        ax.plot([0.3, 1], [0.3, 1], "w--", lw=1)
        ax.set_title(f"{art}: MAE {np.abs(g.severity_target - g.sev_pred).mean():.3f}")
        ax.set_xlabel("true SSIM")
    axes[0].set_ylabel("predicted SSIM")
    fig.suptitle("Classical RF: severity on test slices")
    fig.tight_layout()
    fig.savefig(args.out / "fig_severity.png", dpi=120)
    plt.close(fig)
    print(f"\noutputs in {args.out}")


if __name__ == "__main__":
    main()
