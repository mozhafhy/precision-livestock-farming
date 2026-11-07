"""Skenario E2 & E3: Evaluasi Lengkap Ablasi Normalisasi Lintas Domain (Tabel 8 Laporan)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from ..e0.p4_degrade import degrade
from ..e1.common import (
    COCO_JSON,
    CONFIGS,
    E0,
    E1,
    ROI_MARGIN,
    input_size,
    load_protocol,
)
from ..e1.metrics import image_metrics
from ..e1.preprocessing import (
    CocoIndex,
    invert_to_original,
    load_rgb,
    preprocess_image,
)
from ..e1.run import yolo_instances, yolo_union

ROOT = Path(".")
E1_DIR = ROOT / "results" / "E1_arsitektur"
E2_DIR = ROOT / "results" / "E2_ablasi"
MDY_RAW_DIR = ROOT / "dataset" / "raw" / "mdy"
MDY_COCO_JSON = ROOT / "dataset" / "annotations" / "mdy" / "annotations.coco.json"
MDY_SPLIT_CSV = E0 / "splits" / "external_mendeley.csv"
KGL_SPLIT_TEST = E0 / "splits" / "test.csv"
KGL_RAW_DIR = ROOT / "dataset" / "raw" / "kgl"
DEGRADED_DIR = ROOT / "dataset" / "degraded" / "sintetis100"


def load_dataset_for_config(config_name: str, size: int):
    cfg_flags = CONFIGS[config_name]
    idx_mdy = CocoIndex(MDY_COCO_JSON)
    df_mdy = pd.read_csv(MDY_SPLIT_CSV)
    mdy_items = []
    for _, row in df_mdy.iterrows():
        image_id = row["image_id"]
        img_p = MDY_RAW_DIR / f"{image_id}.jpg"
        if not img_p.exists():
            continue
        rgb = load_rgb(img_p)
        img_dict = idx_mdy.image(f"{image_id}.jpg")
        bboxes = idx_mdy.bboxes(img_dict)
        gt_union = idx_mdy.union_mask(img_dict)
        canvas, tf = preprocess_image(rgb, bboxes, size, margin=ROI_MARGIN, **cfg_flags)
        mdy_items.append(
            {"image_id": image_id, "canvas": canvas, "tf": tf, "gt": gt_union}
        )

    idx_kgl = CocoIndex(COCO_JSON)
    df_test = pd.read_csv(KGL_SPLIT_TEST)
    DEGRADED_DIR.mkdir(parents=True, exist_ok=True)
    deg_items = []
    for _, row in df_test.iterrows():
        image_id = row["image_id"]
        src_p = KGL_RAW_DIR / f"{image_id}.jpg"
        deg_p = DEGRADED_DIR / f"{image_id}.png"
        if not deg_p.exists():
            orig_bgr = cv2.imread(str(src_p))
            cv2.imwrite(str(deg_p), degrade(orig_bgr, target_dim=100))
        rgb = load_rgb(deg_p)
        img_dict = idx_kgl.image(f"{image_id}.jpg")
        bboxes = idx_kgl.bboxes(img_dict)
        gt_union = idx_kgl.union_mask(img_dict)
        canvas, tf = preprocess_image(rgb, bboxes, size, margin=ROI_MARGIN, **cfg_flags)
        deg_items.append(
            {"image_id": image_id, "canvas": canvas, "tf": tf, "gt": gt_union}
        )
    return mdy_items, deg_items


def evaluate_items(
    model, model_name: str, items: list[dict], size: int, conf: float, device: str
) -> float:
    rows = []

    class DummyItem:
        def __init__(self, x, tf):
            self.x, self.tf = x, tf

    eval_items_obj = [DummyItem(it["canvas"], it["tf"]) for it in items]
    insts = yolo_instances(model, eval_items_obj, conf, size, device)
    for it, inst in zip(items, insts):
        pred_mask = yolo_union(inst, it["tf"], conf)
        rows.append(image_metrics(pred_mask, it["gt"])["iou_exc"])
    return float(np.mean(rows))


def main():
    E2_DIR.mkdir(parents=True, exist_ok=True)
    proto = load_protocol()
    size = input_size(proto)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "yolov8s-seg"

    configs = [
        ("A", "A. Tanpa normalisasi"),
        ("B", "B. Normalisasi spasial"),
        ("C", "C. Spasial + spektral"),
    ]
    table_rows = []

    for cfg_code, cfg_label in configs:
        print(f">>> Mengevaluasi {cfg_label}...")
        run_name = f"E1_{model_name}_cfg{cfg_code}_s0"
        run_dir = E1_DIR / run_name
        if not run_dir.exists():
            run_dir = E2_DIR / run_name

        ckpt_path = run_dir / "best_val.pt"
        summary_data = json.loads(
            (run_dir / "metrics_summary.json").read_text(encoding="utf-8")
        )
        conf = summary_data["selection"].get("yolo_conf", 0.3)
        in_domain_iou = summary_data["test"]["iou_exc"]

        from ultralytics import YOLO

        model = YOLO(str(ckpt_path))

        mdy_items, deg_items = load_dataset_for_config(cfg_code, size)
        ext_iou = evaluate_items(model, model_name, mdy_items, size, conf, device)
        deg_iou = evaluate_items(model, model_name, deg_items, size, conf, device)

        table_rows.append(
            {
                "Konfigurasi": cfg_label,
                "Uji in-domain": f"{in_domain_iou * 100:.1f}%",
                "Uji eksternal (dari E3)": f"{ext_iou * 100:.1f}%",
                "Uji degradasi sintetis (dari E3)": f"{deg_iou * 100:.1f}%",
            }
        )

    df_t8 = pd.DataFrame(table_rows)
    df_t8.to_csv(E2_DIR / "tabel8_lengkap.csv", index=False)
    try:
        (E2_DIR / "tabel8_lengkap.md").write_text(
            df_t8.to_markdown(index=False), encoding="utf-8"
        )
    except Exception:
        (E2_DIR / "tabel8_lengkap.md").write_text(
            df_t8.to_string(index=False), encoding="utf-8"
        )

    print("\n================ ISIAN E2: LENGKAP UNTUK TABEL 8 ================")
    print(df_t8.to_string(index=False))


if __name__ == "__main__":
    main()
