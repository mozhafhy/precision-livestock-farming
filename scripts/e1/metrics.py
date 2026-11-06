"""Metrik segmentasi biner per citra, dihitung pada resolusi asli citra."""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_COLS = ["iou_exc", "dsc_exc", "iou_bg", "miou", "pa", "precision", "recall"]


def image_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    n = gt.size
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = n - tp - fp - fn

    # Tanpa prediksi pada citra yang punya ekskreta: tp=0 -> IoU=DSC=0 (aturan 4).
    iou_exc = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    dsc_exc = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou_bg = tn / (tn + fp + fn) if (tn + fp + fn) else 1.0
    return {
        "iou_exc": iou_exc,
        "dsc_exc": dsc_exc,
        "iou_bg": iou_bg,
        "miou": (iou_exc + iou_bg) / 2,
        "pa": (tp + tn) / n,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_px": tp + fp,
        "gt_px": tp + fn,
        "no_pred": int(tp + fp == 0),
    }


def summarize(df: pd.DataFrame) -> dict:
    """Rerata antar-citra untuk satu run."""
    return {c: float(df[c].mean()) for c in METRIC_COLS}
