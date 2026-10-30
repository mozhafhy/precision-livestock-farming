"""Skrip Pembuat Dataset Mentah Utuh: rawGrayworld dan rawMask.

Tugas Skrip:
1. Membaca 305 citra asli utuh dari dataset/raw/.
2. Membaca masker anotasi poligon/RLE utuh dari dataset/annotations/master_cleaned.json.
3. Membaca pembagian resmi (train: 213, val: 46, test: 46) dari dataset/split.csv.
4. Menghasilkan folder dataset/rawGrayworld/:
   - Mentransformasi setiap citra raw menjadi Gray World (Buchsbaum, 1980) tanpa pemotongan.
   - Menyimpan ke dataset/rawGrayworld/{train, val, test}/{image_name}.jpg
5. Menghasilkan folder dataset/rawMask/:
   - Merender masker biner hitam-putih utuh (0 dan 255) tanpa pemotongan.
   - Menyimpan ke dataset/rawMask/{train, val, test}/{image_name}.png
6. Menjamin 100% konsistensi nama file dan distribusi subset antara rawGrayworld dan rawMask.
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

# Mencegah pembuatan file __pycache__
sys.dont_write_bytecode = True

# Penentuan Root Proyek yang dinamis
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.file_managements import load_json


# -----------------------------------------------------------------------------
# 1. Algoritma Koreksi Warna Gray World (Modul Hal. 13-14)
# -----------------------------------------------------------------------------
def apply_gray_world(img_bgr: np.ndarray) -> np.ndarray:
    """Menerapkan koreksi warna Gray World (Buchsbaum, 1980) pada citra BGR penuh."""
    f = img_bgr.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)  # Rerata intensitas kanal B, G, R
    mean_gray = means.mean()
    gain = mean_gray / (means + 1e-6)
    corrected = np.clip(f * gain, 0.0, 255.0)
    return corrected.astype(np.uint8)


# -----------------------------------------------------------------------------
# 2. Dekoder Masker Anotasi COCO Murni (Tanpa Dependensi Eksternal)
# -----------------------------------------------------------------------------
def _decode_coco_rle_string(rle_str: str, h: int, w: int) -> np.ndarray:
    """Dekoder murni Python untuk format COCO compressed RLE LEB128."""
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
    """Mengubah poligon atau RLE anotasi COCO menjadi masker biner H x W [0, 255]."""
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    if segmentation is None:
        return binary_mask

    # Format Poligon (List of coordinate pairs)
    if isinstance(segmentation, list):
        for poly in segmentation:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(binary_mask, [pts], 255)
        return binary_mask

    # Format RLE (Dictionary counts & size)
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
# 3. Pipeline Pembuat rawGrayworld dan rawMask
# -----------------------------------------------------------------------------
def build_raw_grayworld_and_masks(
    raw_dir: Path | None = None,
    coco_path: Path | None = None,
    split_csv_path: Path | None = None,
    output_gw_dir: Path | None = None,
    output_mask_dir: Path | None = None,
) -> None:
    raw_dir = raw_dir or ROOT / "dataset" / "raw"
    coco_path = coco_path or ROOT / "dataset" / "annotations" / "master_cleaned.json"
    split_csv_path = split_csv_path or ROOT / "dataset" / "split.csv"
    output_gw_dir = output_gw_dir or ROOT / "dataset" / "rawGrayworld"
    output_mask_dir = output_mask_dir or ROOT / "dataset" / "rawMask"

    print("=" * 75)
    print("MEMPROSES PEMBUATAN DATASET: rawGrayworld & rawMask (CITRA UTUH TANPA CROP)")
    print(f"Sumber Citra Raw      : {raw_dir}")
    print(f"Sumber Anotasi Master : {coco_path}")
    print(f"Daftar Split CSV      : {split_csv_path}")
    print(f"Tujuan rawGrayworld   : {output_gw_dir}")
    print(f"Tujuan rawMask        : {output_mask_dir}")
    print("=" * 75)

    # 1. Validasi keberadaan file
    if not split_csv_path.exists():
        raise FileNotFoundError(f"File split.csv tidak ditemukan di: {split_csv_path}")
    if not coco_path.exists():
        raise FileNotFoundError(f"File master_cleaned.json tidak ditemukan di: {coco_path}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Folder raw tidak ditemukan di: {raw_dir}")

    # 2. Muat pembagian train, val, test
    df_split = pd.read_csv(split_csv_path, dtype=str)
    split_map = dict(zip(df_split["image_id"], df_split["split"]))

    # 3. Muat metadata anotasi COCO
    coco_data = load_json(coco_path)
    images_info = {img["file_name"]: img for img in coco_data.get("images", [])}
    annotations_by_image = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        annotations_by_image[ann["image_id"]].append(ann)

    # 4. Siapkan subfolder train, val, test pada kedua tujuan
    for split in ["train", "val", "test"]:
        (output_gw_dir / split).mkdir(parents=True, exist_ok=True)
        (output_mask_dir / split).mkdir(parents=True, exist_ok=True)

    # 5. Iterasi dan proses seluruh citra di dataset/raw/
    raw_images = sorted(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")))
    total_images = len(raw_images)
    print(f"\nMemproses {total_images} citra mentah...")

    processed_counts = {"train": 0, "val": 0, "test": 0}

    for idx, img_path in enumerate(raw_images, 1):
        filename = img_path.name
        img_info = images_info.get(filename)
        if not img_info:
            continue

        stem = img_path.stem
        split = split_map.get(stem)
        if not split or split not in processed_counts:
            continue

        image_id = img_info["id"]

        # Baca citra asli BGR
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]

        # A. Buat Citra Gray World (Utuh tanpa crop)
        img_gw = apply_gray_world(img_bgr)
        out_gw_path = output_gw_dir / split / f"{stem}.jpg"
        cv2.imwrite(str(out_gw_path), img_gw)

        # B. Render Masker Biner (Utuh tanpa crop, biner tegas [0, 255])
        full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in annotations_by_image.get(image_id, []):
            mask_part = decode_coco_mask(ann.get("segmentation"), orig_h, orig_w)
            full_mask = np.bitwise_or(full_mask, mask_part)

        # Pastikan biner murni [0, 255] dalam format PNG (lossless)
        full_mask = np.where(full_mask > 127, 255, 0).astype(np.uint8)
        out_mask_path = output_mask_dir / split / f"{stem}.png"
        cv2.imwrite(str(out_mask_path), full_mask)

        processed_counts[split] += 1

        if idx % 50 == 0 or idx == total_images:
            print(f"[{idx}/{total_images}] Citra selesai diproses...")

    print("\n" + "=" * 75)
    print("SELESAI! Seluruh citra rawGrayworld dan rawMask berhasil dibuat dengan sempurna:")
    print(f"-> Train : {processed_counts['train']} pasang citra & masker")
    print(f"-> Val   : {processed_counts['val']} pasang citra & masker")
    print(f"-> Test  : {processed_counts['test']} pasang citra & masker")
    print(f"-> Total : {sum(processed_counts.values())} pasang")
    print(f"Folder rawGrayworld : {output_gw_dir}")
    print(f"Folder rawMask      : {output_mask_dir}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rawGrayworld & rawMask datasets with identical splits")
    args = parser.parse_args()

    build_raw_grayworld_and_masks()
