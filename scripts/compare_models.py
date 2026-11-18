"""Stage 4c: put the classical baseline and the CNN side by side.

Reads outputs/baseline/metrics.json and outputs/cnn/<experiment>/metrics.json
and writes outputs/comparison/{summary.csv, by_artefact_level.csv, fig_comparison.png}.

Usage:
    python scripts/compare_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAIRS = {  # setting -> (baseline key, cnn experiment, cnn key)
    "seen scanners (test)": ("classical_rf/test", "standard", "cnn/test"),
    "unseen scanner (IOP)": ("classical_rf/unseen_site_IOP", "unseen_site", "cnn/unseen_site_IOP"),
}
METRICS = ["quality_auroc", "quality_bal_acc", "quality_specificity", "quality_sensitivity",
           "type_macro_f1", "severity_mae", "severity_spearman"]


def main() -> None:
    out = Path("outputs/comparison")
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path("outputs/baseline/metrics.json").read_text(encoding="utf-8"))
    rows, levels = [], []
    for setting, (bkey, exp, ckey) in PAIRS.items():
        p = Path("outputs/cnn") / exp / "metrics.json"
        if not p.exists():
            print(f"skip {setting}: {p} missing")
            continue
        cnn = json.loads(p.read_text(encoding="utf-8"))
        for model, res in (("classical RF", base[bkey]), ("CNN", cnn[ckey])):
            rows.append({"setting": setting, "model": model, **{m: res["overall"][m] for m in METRICS}})
            for lvl, v in res["by_artefact_level"].items():
                levels.append({"setting": setting, "model": model, "artefact_level": lvl, **v})

    summ = pd.DataFrame(rows).set_index(["setting", "model"])
    summ.round(4).to_csv(out / "summary.csv")
    lv = pd.DataFrame(levels)
    lv.round(4).to_csv(out / "by_artefact_level.csv", index=False)
    print(summ.round(3).to_string())
    print()
    print(lv.pivot_table(index="artefact_level", columns=["setting", "model"], values="type_recall")
          .round(3).to_string())

    show = ["quality_auroc", "quality_specificity", "type_macro_f1", "severity_spearman"]
    settings = list(dict.fromkeys(summ.index.get_level_values(0)))
    fig, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 4.6), squeeze=False)
    for ax, s in zip(axes[0], settings):
        d = summ.loc[s, show]
        x = np.arange(len(show))
        for k, (model, color) in enumerate((("classical RF", "#999999"), ("CNN", "#4c72b0"))):
            if model in d.index:
                bars = ax.bar(x + (k - 0.5) * 0.38, d.loc[model], 0.38, label=model, color=color)
                ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_xticks(x, [m.replace("_", "\n") for m in show], fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_title(s)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_comparison.png", dpi=120)
    plt.close(fig)
    print(f"\noutputs in {out}")


if __name__ == "__main__":
    main()
