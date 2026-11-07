"""Skenario E2 & E3: Evaluasi Lengkap Ablasi Normalisasi Lintas Domain (Tabel 8 Laporan).

Mengisi seluruh sel pada Isian E2:
- Konfigurasi A (Tanpa normalisasi)
- Konfigurasi B (Normalisasi spasial saja)
- Konfigurasi C (Normalisasi spasial + spektral Gray World)

Masing-masing dievaluasi pada:
1. Uji In-Domain (Kaggle Test 225 citra)
2. Uji Eksternal (Mendeley Data 100 citra)
3. Uji Degradasi Sintetis (100 px 225 citra)

Sesuai aturan metodologi:
Setiap model dievaluasi dengan pipeline prapemrosesan konfigurasinya masing-masing (A, B, C).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from scripts.e0.p4_degrade import degrade
from scripts.e1.common import COCO_JSON, CONFIGS, E0, E1, ROI_MARGIN, input_size, load_protocol
from scripts.e1.metrics import image_metrics
from scripts.e1.preprocessing import CocoIndex, invert_to_original, load_rgb, preprocess_image
from scripts.e1.run import yolo_instances, yolo_union

ROOT = Path(".")
E1_DIR = ROOT / "results" / "E1_arsitektur"
E2_DIR = ROOT / "results" / "E2_ablasi"
MDY_RAW_DIR = ROOT / "dataset" / "raw" / "mdy"
MDY_COCO_JSON = ROOT / "dataset" / "annotations" / "mdy" / "annotations.coco.json"
MDY_SPLIT_CSV = E0 / "splits" / "external_mendeley.csv"
KGL_SPLIT_TEST = E0 / "splits" / "test.csv"
KGL_RAW_DIR = ROOT / "dataset" / "raw" / "kgl"
DEGRADED_DIR = ROOT / "dataset" / "degraded" / "sintetis100"


def load_dataset_for_config(config_name: str, size: int) -> tuple[list[dict], list[dict]]:
    """Memuat item Mendeley dan Sintetis sesuai dengan aturan prapemrosesan Konfigurasi A, B, atau C."""
    cfg_flags = CONFIGS[config_name]

    # 1. Mendeley (100 citra)
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
        mdy_items.append({"image_id": image_id, "canvas": canvas, "tf": tf, "gt": gt_union})

    # 2. Degradasi Sintetis (225 citra)
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
            deg_bgr = degrade(orig_bgr, target_dim=100)
            cv2.imwrite(str(deg_p), deg_bgr)

        rgb = load_rgb(deg_p)
        img_dict = idx_kgl.image(f"{image_id}.jpg")
        bboxes = idx_kgl.bboxes(img_dict)
        gt_union = idx_kgl.union_mask(img_dict)
        canvas, tf = preprocess_image(rgb, bboxes, size, margin=ROI_MARGIN, **cfg_flags)
        deg_items.append({"image_id": image_id, "canvas": canvas, "tf": tf, "gt": gt_union})

    return mdy_items, deg_items


def evaluate_items(model, model_name: str, items: list[dict], size: int, conf: float, device: str) -> float:
    """Evaluasi inferensi batch=1 dan mengembalikan rata-rata IoU ekskreta."""
    rows = []
    if model_name.startswith("yolo"):
        class DummyItem:
            def __init__(self, x, tf):
                self.x = x
                self.tf = tf
        eval_items_obj = [DummyItem(it["canvas"], it["tf"]) for it in items]
        insts = yolo_instances(model, eval_items_obj, conf, size, device)
        for it, inst in zip(items, insts):
            pred_mask = yolo_union(inst, it["tf"], conf)
            m = image_metrics(pred_mask, it["gt"])
            rows.append(m["iou_exc"])
    else:
        imnet_mean = np.array([0.485, 0.456, 0.406], np.float32)
        imnet_std = np.array([0.229, 0.224, 0.225], np.float32)
        dev = torch.device(device)
        model.to(dev)
        model.eval()
        with torch.no_grad():
            for it in items:
                rgb_norm = (it["canvas"].astype(np.float32) / 255.0 - imnet_mean) / imnet_std
                t = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(dev)
                prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
                prob_orig = invert_to_original(prob, it["tf"])
                pred_mask = (prob_orig >= conf).astype(np.uint8)
                m = image_metrics(pred_mask, it["gt"])
                rows.append(m["iou_exc"])

    return float(np.mean(rows))


def main():
    parser = argparse.ArgumentParser(description="Evaluasi Lengkap Ablasi Lintas Domain (Tabel 8)")
    parser.add_argument("--model", default="yolov8s-seg", help="Nama arsitektur model (default: yolov8s-seg)")
    args = parser.parse_args()

    E2_DIR.mkdir(parents=True, exist_ok=True)
    proto = load_protocol()
    size = input_size(proto)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = args.model
    print(f"=====================================================================")
    print(f"EVALUASI ABLASI LINTAS DOMAIN UNTUK TABEL 8 (MODEL: {model_name})")
    print(f"=====================================================================\n")

    configs = [
        ("A", "A. Tanpa normalisasi"),
        ("B", "B. Normalisasi spasial"),
        ("C", "C. Spasial + spektral"),
    ]

    table_rows = []

    for cfg_code, cfg_label in configs:
        print(f">>> Mengevaluasi {cfg_label} (Seed 0)...")
        run_name = f"E1_{model_name}_cfg{cfg_code}_s0"
        run_dir = E1_DIR / run_name
        if not run_dir.exists():
            run_dir = E2_DIR / run_name

        if not run_dir.exists():
            print(f"[Error] Folder bobot {run_name} tidak ditemukan di E1 maupun E2.")
            continue

        ckpt_path = run_dir / "best_val.pt"
        summary_path = run_dir / "metrics_summary.json"
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        conf = summary_data["selection"].get("yolo_conf", summary_data["selection"].get("unet_threshold", 0.5))

        # 1. In-domain test IoU dari data training E1/E2
        in_domain_iou = summary_data["test"]["iou_exc"]

        # 2. Muat Model
        if model_name.startswith("yolo"):
            from ultralytics import YOLO
            model = YOLO(str(ckpt_path))
        else:
            import segmentation_models_pytorch as smp
            model = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=3, classes=1)
            weights = torch.load(ckpt_path, map_location="cpu")
            if isinstance(weights, dict) and "state_dict" in weights:
                weights = weights["state_dict"]
            model.load_state_dict(weights)

        # 3. Muat data uji eksternal & sintetis dengan aturan prapemrosesan konfigurasi ini
        mdy_items, deg_items = load_dataset_for_config(cfg_code, size)

        # 4. Inferensi Eksternal Mendeley (100 citra)
        ext_iou = evaluate_items(model, model_name, mdy_items, size, conf, device)

        # 5. Inferensi Degradasi Sintetis (225 citra)
        deg_iou = evaluate_items(model, model_name, deg_items, size, conf, device)

        table_rows.append({
            "Konfigurasi": cfg_label,
            "Uji in-domain": f"{in_domain_iou * 100:.1f}%",
            "Uji eksternal (dari E3)": f"{ext_iou * 100:.1f}%",
            "Uji degradasi sintetis (dari E3)": f"{deg_iou * 100:.1f}%",
        })

    df_t8 = pd.DataFrame(table_rows)

    # Simpan hasil resmi
    out_csv = E2_DIR / "tabel8_lengkap.csv"
    out_md = E2_DIR / "tabel8_lengkap.md"
    df_t8.to_csv(out_csv, index=False)
    try:
        out_md.write_text(df_t8.to_markdown(index=False), encoding="utf-8")
    except Exception:
        out_md.write_text(df_t8.to_string(index=False), encoding="utf-8")

    print("\n================ ISIAN E2: LENGKAP UNTUK TABEL 8 ================")
    print(df_t8.to_string(index=False))
    print(f"\nHasil tersimpan di {out_csv} dan {out_md}\n")


if __name__ == "__main__":
    main()
