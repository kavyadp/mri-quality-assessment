"""Copy the figures and result tables used in the README from outputs/ (git-ignored) to docs/.

Usage:
    python scripts/collect_figures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

FILES = {
    "outputs/artefacts/fig_gallery.png": "docs/figures/artefact_gallery.png",
    "outputs/artefacts/fig_motion_kspace.png": "docs/figures/motion_kspace.png",
    "outputs/exploration/fig_axial_selection.png": "docs/figures/axial_selection.png",
    "outputs/baseline/fig_confusion.png": "docs/figures/baseline_confusion.png",
    "outputs/baseline/fig_importance.png": "docs/figures/baseline_importance.png",
    "outputs/comparison/fig_comparison.png": "docs/figures/baseline_vs_cnn.png",
    "outputs/cnn/standard/fig_curves.png": "docs/figures/cnn_training_curves.png",
    "outputs/mrart/fig_score_vs_radiologist.png": "docs/figures/mrart_score_vs_radiologist.png",
    "outputs/mrart/fig_example_subject.png": "docs/figures/mrart_example_subject.png",
    "outputs/gradcam/fig_gradcam_ixi.png": "docs/figures/gradcam_simulated.png",
    "outputs/gradcam/fig_gradcam_mrart.png": "docs/figures/gradcam_real_motion.png",
    "outputs/comparison/summary.csv": "docs/results/baseline_vs_cnn.csv",
    "outputs/comparison/site_calibration.csv": "docs/results/site_calibration.csv",
    "outputs/mrart/summary.csv": "docs/results/mrart.csv",
    "outputs/gradcam/cam_in_head.csv": "docs/results/gradcam_in_head.csv",
    "outputs/artefacts/severity_calibration.csv": "docs/results/severity_calibration.csv",
}


def main() -> None:
    for src, dst in FILES.items():
        if not Path(src).exists():
            print(f"missing {src}")
            continue
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"copied into docs/ ({sum(Path(s).exists() for s in FILES)} of {len(FILES)} files)")


if __name__ == "__main__":
    main()
