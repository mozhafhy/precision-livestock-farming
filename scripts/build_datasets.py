"""Skrip Penyiapan Dataset Masukan untuk Segmentasi (Tahap 4: Normalisasi Spasial & Spektral).

KHUSUS ROI DETEKTOR YOLOv8n:
- Otomatis mendeteksi dan menggunakan GPU (CUDA) bila tersedia.
- Bebas dari dependensi pycocotools.
- Memotong citra dan masker biner ground-truth ke ukuran 512x512 piksel.
- Menghasilkan dua varian: raw dan gray_world.
- Output tersimpan di: dataset/cropped/
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

# Mencegah pembuatan folder __pycache__
sys.dont_write_bytecode = True

# Penentuan Root Proyek yang dinamis
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.file_managements import load_json


# -----------------------------------------------------------------------------
# 1. Normalisasi Warna Gray World (Modul Hal. 13-14)
# -----------------------------------------------------------------------------
def apply_gray_world(img_bgr: np.ndarray) -> np.ndarray:
    """Menerapkan koreksi warna Gray World (Buchsbaum, 1980)."""
    f = img_bgr.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)  # Rerata intensitas B, G, R
    mean_gray = means.mean()
    gain = mean_gray / (means + 1e-6)
    corrected = np.clip(f * gain, 0.0, 255.0)
    return corrected.astype(np.uint8)


# -----------------------------------------------------------------------------
# 2. Dekoder Masker Anotasi COCO Murni (Tanpa Butuh pycocotools)
# -----------------------------------------------------------------------------
def _decode_coco_rle_string(rle_str: str, h: int, w: int) -> np.ndarray:
    """Dekoder murni Python untuk COCO compressed RLE LEB128."""
    counts = []
    p = 0
    while p < len(rle_str):
        x = 0
        k = 0
        more = 1
        while more:
            c = ord(rle_str[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)

    flat_mask = np.zeros(h * w, dtype=np.uint8)
    cur = 0
    val = 0
    for cnt in counts:
        flat_mask[cur : cur + cnt] = val
        cur += cnt
        val = 255 - val
    return flat_mask.reshape((w, h)).T


def decode_coco_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    """Mengubah anotasi poligon atau RLE COCO menjadi masker biner H x W."""
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    if segmentation is None:
        return binary_mask

    if isinstance(segmentation, list):
        for poly in segmentation:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(binary_mask, [pts], 255)
        return binary_mask

    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        size = segmentation.get("size", [height, width])
        h, w = size[0], size[1]

        if isinstance(counts, str):
            return _decode_coco_rle_string(counts, h, w)
        elif isinstance(counts, list):
            flat_mask = np.zeros(h * w, dtype=np.uint8)
            cur = 0
            val = 0
            for cnt in counts:
                flat_mask[cur : cur + cnt] = val
                cur += cnt
                val = 255 - val
            return flat_mask.reshape((w, h)).T

    return binary_mask


# -----------------------------------------------------------------------------
# 3. Pengambilan Bounding Box dari YOLOv8n Menggunakan GPU
# -----------------------------------------------------------------------------
def get_yolo_roi_box(
    model: Any,
    img_bgr: np.ndarray,
    device: str | int = 0,
    conf: float = 0.47,
    iou: float = 0.6,
    padding_px: int = 0,
) -> tuple[int, int, int, int]:
    """Mendeteksi ROI menggunakan model YOLOv8n dan mengambil kotak gabungan."""
    h, w = img_bgr.shape[:2]
    results = model.predict(img_bgr, conf=conf, iou=iou, imgsz=640, device=device, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return 0, 0, w, h

    boxes = results.boxes.xyxy.cpu().numpy()
    x1 = int(np.floor(np.min(boxes[:, 0]))) - padding_px
    y1 = int(np.floor(np.min(boxes[:, 1]))) - padding_px
    x2 = int(np.ceil(np.max(boxes[:, 2]))) + padding_px
    y2 = int(np.ceil(np.max(boxes[:, 3]))) + padding_px

    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


# -----------------------------------------------------------------------------
# 4. Pipeline Pembuatan Dataset
# -----------------------------------------------------------------------------
def build_dataset_yolo(
    target_size: int = 512,
    raw_dir: Path | None = None,
    coco_path: Path | None = None,
    split_csv_path: Path | None = None,
    yolo_weights_path: Path | None = None,
    output_base_dir: Path | None = None,
) -> None:
    raw_dir = raw_dir or ROOT / "dataset" / "raw"
    coco_path = coco_path or ROOT / "dataset" / "annotations" / "master_cleaned.json"
    split_csv_path = split_csv_path or ROOT / "dataset" / "split.csv"
    yolo_weights_path = yolo_weights_path or ROOT / "dataset" / "detector" / "runs" / "train-2" / "weights" / "best.pt"
    output_base_dir = output_base_dir or ROOT / "dataset" / "cropped"

    # Deteksi otomatis GPU / CPU
    if torch.cuda.is_available():
        device: str | int = 0
        device_name = f"{torch.cuda.get_device_name(0)} (CUDA:0)"
    else:
        device = "cpu"
        device_name = "CPU (CUDA tidak terdeteksi)"

    print("=" * 70)
    print("PENYIAPAN DATASET MASUKAN SEGMENTASI (ROI KHUSUS YOLOV8n)")
    print(f"Perangkat Komputasi   : {device_name}")
    print(f"Resolusi target       : {target_size}x{target_size} piksel")
    print(f"Bobot Detektor YOLOv8 : {yolo_weights_path}")
    print(f"Folder Output Tujuan  : {output_base_dir}")
    print("=" * 70)

    if not split_csv_path.exists():
        raise FileNotFoundError(f"File split.csv tidak ditemukan di: {split_csv_path}")
    if not coco_path.exists():
        raise FileNotFoundError(f"File master_cleaned.json tidak ditemukan di: {coco_path}")
    if not yolo_weights_path.exists():
        raise FileNotFoundError(f"Bobot YOLOv8n best.pt tidak ditemukan di: {yolo_weights_path}")

    df_split = pd.read_csv(split_csv_path, dtype=str)
    split_map = dict(zip(df_split["image_id"], df_split["split"]))

    coco_data = load_json(coco_path)
    images_info = {img["file_name"]: img for img in coco_data.get("images", [])}
    annotations_by_image = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        annotations_by_image[ann["image_id"]].append(ann)

    from ultralytics import YOLO

    yolo_model = YOLO(str(yolo_weights_path))
    print("-> Detektor YOLOv8n berhasil dimuat.")

    for variant in ["raw", "gray_world"]:
        for split in ["train", "val", "test"]:
            (output_base_dir / variant / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_base_dir / variant / "masks" / split).mkdir(parents=True, exist_ok=True)

    raw_images = sorted(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")))
    total_images = len(raw_images)
    print(f"\nMemproses {total_images} citra mentah pada {device_name}...")

    processed_count = 0

    for idx, img_path in enumerate(raw_images, 1):
        filename = img_path.name
        img_info = images_info.get(filename)
        if not img_info:
            continue

        split = split_map.get(img_path.stem)
        if not split:
            continue
        image_id = img_info["id"]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]

        full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in annotations_by_image.get(image_id, []):
            mask_part = decode_coco_mask(ann.get("segmentation"), orig_h, orig_w)
            full_mask = np.bitwise_or(full_mask, mask_part)

        x1, y1, x2, y2 = get_yolo_roi_box(yolo_model, img_bgr, device=device, conf=0.47, iou=0.6)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, orig_w, orig_h

        crop_img_raw = img_bgr[y1:y2, x1:x2]
        crop_mask = full_mask[y1:y2, x1:x2]

        crop_img_gw = apply_gray_world(crop_img_raw)

        resized_raw = cv2.resize(crop_img_raw, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        resized_gw = cv2.resize(crop_img_gw, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        resized_mask = cv2.resize(crop_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        resized_mask = np.where(resized_mask > 127, 255, 0).astype(np.uint8)

        out_img_raw = output_base_dir / "raw" / "images" / split / f"{img_path.stem}.jpg"
        out_mask_raw = output_base_dir / "raw" / "masks" / split / f"{img_path.stem}.png"
        cv2.imwrite(str(out_img_raw), resized_raw)
        cv2.imwrite(str(out_mask_raw), resized_mask)

        out_img_gw = output_base_dir / "gray_world" / "images" / split / f"{img_path.stem}.jpg"
        out_mask_gw = output_base_dir / "gray_world" / "masks" / split / f"{img_path.stem}.png"
        cv2.imwrite(str(out_img_gw), resized_gw)
        cv2.imwrite(str(out_mask_gw), resized_mask)

        processed_count += 1
        if idx % 50 == 0 or idx == total_images:
            print(f"[{idx}/{total_images}] Citra selesai diproses...")

    print("\n" + "=" * 70)
    print(f"SELESAI! {processed_count} citra berhasil di-crop menggunakan YOLOv8n ({device_name}).")
    print(f"Hasil tersimpan di: {output_base_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YOLO-Cropped Excreta Dataset (512x512)")
    parser.add_argument("--imgsz", type=int, default=512, help="Resolusi target (default: 512)")
    args = parser.parse_args()

    build_dataset_yolo(target_size=args.imgsz)
