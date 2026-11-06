"""Evaluasi semantik di ruang citra asli.

Semua instance prediksi digabung menjadi satu masker biner, dipetakan balik ke
ukuran citra asli lewat riwayat sampel, lalu dibandingkan dengan GT penuh.
Protokol ini identik untuk keempat kombinasi (dan bisa dipakai juga untuk U-Net).
"""

from __future__ import annotations

import numpy as np

from .coco import group_by_image, load_coco
from .geometry import rasterize, to_polys
from .history import SampleRecord, load_manifest, restore_mask
from .logger import get_logger, stage


def ground_truth_mask(anns: list[dict], height: int, width: int) -> np.ndarray:
    return rasterize([poly for ann in anns for poly in to_polys(ann)], height, width)


def binary_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    """IoU dan Dice dua masker biner. Keduanya kosong dianggap sempurna (1.0)."""
    p, g = pred > 0, gt > 0
    intersection = float(np.logical_and(p, g).sum())
    union = float(np.logical_or(p, g).sum())
    total = float(p.sum() + g.sum())
    return {
        "iou": intersection / union if union else 1.0,
        "dice": 2 * intersection / total if total else 1.0,
    }


def predict_mask(model, cfg, record: SampleRecord, conf: float = 0.25) -> np.ndarray:
    """Masker biner prediksi (size x size, ruang input model) untuk satu sampel."""
    result = model.predict(
        str(cfg.root / record.image), imgsz=cfg.size, conf=conf, verbose=False
    )[0]
    polygons = [
        xy
        for xy in (result.masks.xy if result.masks is not None else [])
        if len(xy) >= 3
    ]
    return rasterize(polygons, cfg.size, cfg.size)


def evaluate_semantic(
    cfg,
    weights,
    variant: str,
    mode: str,
    split: str = "test",
    conf: float = 0.25,
    log=None,
) -> dict:
    from ultralytics import YOLO

    log = log or get_logger()
    with stage(log, f"Evaluasi semantik {variant}/{mode}/{split}"):
        model = YOLO(str(weights))
        records = [
            r
            for r in load_manifest(cfg.dataset_dir(variant, mode)).values()
            if r.split == split
        ]
        anns_by_image = group_by_image(load_coco(cfg.annotation_path(split)))

        scores = []
        for record in records:
            pred = restore_mask(predict_mask(model, cfg, record, conf), record)
            width, height = record.orig_size
            gt = ground_truth_mask(
                anns_by_image.get(record.image_id, []), height, width
            )
            scores.append(binary_scores(pred, gt))

    return {
        "n_images": len(scores),
        "semantic_iou": float(np.mean([s["iou"] for s in scores])),
        "semantic_dice": float(np.mean([s["dice"] for s in scores])),
    }
