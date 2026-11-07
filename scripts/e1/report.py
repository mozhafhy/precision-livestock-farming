"""Agregasi E1: Tabel 7 (rerata ± SD antar-seed), Gambar 2 (box plot), Gambar 3 (overlay).

Pemakaian: python report_e1.py --config C
"""

from __future__ import annotations

import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.file_managements import load_img

from .build_datasets import resolve_raw_path
from .common import COCO_JSON, E1, MODELS, load_protocol, run_id
from .metrics import METRIC_COLS
from .preprocessing import CocoIndex, load_rgb

HEADERS = {
    "iou_exc": "IoU ekskreta",
    "dsc_exc": "DSC ekskreta",
    "iou_bg": "IoU latar",
    "miou": "mIoU",
    "pa": "PA",
    "precision": "Presisi",
    "recall": "Recall",
}


def load_runs(cfg: str, seeds: list[int]) -> dict:
    return {
        (m, s): pd.read_csv(E1 / run_id(m, cfg, s) / "metrics_per_image.csv")
        for m in MODELS
        for s in seeds
    }


def table7(runs: dict, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        row = {"Model": m}
        for c in METRIC_COLS:
            per_seed = np.array([runs[(m, s)][c].mean() for s in seeds])
            row[HEADERS[c]] = f"{per_seed.mean():.3f} ± {per_seed.std(ddof=1):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def mean_iou_per_image(runs: dict, model: str, seeds: list[int]) -> pd.Series:
    return pd.concat(
        [runs[(model, s)].set_index("image")["iou_exc"] for s in seeds], axis=1
    ).mean(axis=1)


def figure2(runs, seeds, out):
    data = [mean_iou_per_image(runs, m, seeds).values for m in MODELS]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot(
        data,
        tick_labels=MODELS,
        showmeans=True,
        flierprops={"marker": "o", "markersize": 3},  # type: ignore
    )
    ax.set_ylabel("IoU ekskreta per citra (rerata 3 seed)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _contour(ax, mask, color):
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=1.2)


def figure3(runs, seeds, cfg, out):
    coco = CocoIndex(COCO_JSON)
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(10, 3.4 * len(MODELS)))
    for r, m in enumerate(MODELS):
        s = mean_iou_per_image(runs, m, seeds).sort_values()
        picks = {
            "Terburuk": s.index[0],
            "Median": s.index[len(s) // 2],
            "Terbaik": s.index[-1],
        }
        run_dir = E1 / run_id(m, cfg, seeds[0])
        for c, (case, stem) in enumerate(picks.items()):
            ax = axes[r, c]
            file_name = runs[(m, seeds[0])].set_index("image").loc[stem, "file_name"]
            img = coco.image(file_name)
            ax.imshow(load_rgb(resolve_raw_path(img)))
            _contour(ax, coco.union_mask(img), "lime")  # GT
            pred = (
                load_img((run_dir / "pred_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
                > 127
            )
            _contour(ax, pred, "red")  # prediksi (seed pertama)
            ax.set_title(f"{m}\n{case}: IoU rerata seed = {s[stem]:.3f}", fontsize=8)
            ax.axis("off")
    fig.suptitle(f"Hijau = GT, merah = prediksi (seed {seeds[0]})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C")
    a = ap.parse_args()
    seeds = load_protocol()["seeds"]
    runs = load_runs(a.config, seeds)

    t7 = table7(runs, seeds)
    t7.to_csv(E1 / "tabel7.csv", index=False)
    (E1 / "tabel7.md").write_text(t7.to_markdown(index=False))
    print(t7.to_string(index=False))
    figure2(runs, seeds, E1 / "gambar2_boxplot.png")
    figure3(runs, seeds, a.config, E1 / "gambar3_overlay.png")
