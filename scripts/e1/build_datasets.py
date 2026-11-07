"""Bangkitkan dataset YOLO dan U-Net dari citra mentah + COCO JSON + split CSV.

Satu pohon dipakai bersama agar kedua keluarga model melihat piksel yang identik:

results/E1_arsitektur/data_cfgC/
    images/<split>/<stem>.png     citra ternormalisasi S x S (dipakai YOLO dan U-Net)
    labels/<split>/<stem>.txt     poligon YOLO-seg (kelas 0 = ekskreta), koordinat ternormalisasi
    masks/<split>/<stem>.png      masker biner U-Net (0/255)
    meta/<split>.json             stem -> {file_name, transform} untuk membalik letterbox
    data.yaml                     konfigurasi dataset Ultralytics

Pemakaian:  python build_datasets.py --config C [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from .common import (
    COCO_JSON,
    CONFIGS,
    E1,
    RAW_DIR,
    ROI_MARGIN,
    SPLITS,
    input_size,
    load_protocol,
    read_split,
)
from .preprocessing import (
    CocoIndex,
    apply_to_mask,
    load_rgb,
    preprocess_image,
    save_rgb,
)


def resolve_raw_path(img: dict) -> Path:
    p = RAW_DIR / img["file_name"]
    return p if p.exists() else RAW_DIR / Path(img["file_name"]).name


def data_dir(config: str) -> Path:
    return E1 / f"data_cfg{config}"


def mask_to_polygons(mask: np.ndarray, min_area: float = 4.0) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [
        c.reshape(-1, 2)
        for c in contours
        if len(c) >= 3 and cv2.contourArea(c) >= min_area
    ]


def write_yolo_label(path: Path, instance_masks: list[np.ndarray]) -> None:
    """Satu baris per komponen kontur: `0 x1 y1 x2 y2 ...` (ternormalisasi 0-1).
    Citra tanpa instans -> berkas kosong (citra latar)."""
    lines = []
    for m in instance_masks:
        s = m.shape[0]
        for poly in mask_to_polygons(m):
            coords = (poly.astype(np.float32) / s).clip(0, 1).reshape(-1)
            lines.append("0 " + " ".join(f"{v:.6f}" for v in coords))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_unet_mask(path: Path, instance_masks: list[np.ndarray], size: int) -> None:
    union = (
        np.any(instance_masks, axis=0)
        if instance_masks
        else np.zeros((size, size), bool)
    )
    cv2.imwrite(str(path), union.astype(np.uint8) * 255)


def build_all(config: str = "C", force: bool = False) -> Path:
    proto = load_protocol()
    size = input_size(proto)
    flags = CONFIGS[config]
    out = data_dir(config)
    if out.exists():
        if not force:
            raise FileExistsError(
                f"{out} sudah ada (pakai --force untuk membangun ulang)"
            )
        shutil.rmtree(out)

    coco = CocoIndex(COCO_JSON)
    seen: set[str] = set()
    (out / "meta").mkdir(parents=True)

    for split in SPLITS:
        for sub in ("images", "labels", "masks"):
            (out / sub / split).mkdir(parents=True)
        meta = {}
        for name in read_split(split):
            img = coco.image(name)
            stem = Path(img["file_name"]).stem
            if stem in seen:
                raise ValueError(f"Stem ganda: {stem}")
            seen.add(stem)

            rgb = load_rgb(resolve_raw_path(img))
            if rgb.shape[:2] != (img["height"], img["width"]):
                raise ValueError(f"{name}: ukuran citra tidak sama dengan COCO")

            x, tf = preprocess_image(
                rgb, coco.bboxes(img), size, margin=ROI_MARGIN, **flags
            )
            inst = [apply_to_mask(m, tf) for m in coco.instance_masks(img)]

            save_rgb(out / "images" / split / f"{stem}.png", x)
            write_yolo_label(out / "labels" / split / f"{stem}.txt", inst)
            write_unet_mask(out / "masks" / split / f"{stem}.png", inst, size)
            meta[stem] = {"file_name": img["file_name"], "transform": tf.to_dict()}

        (out / "meta" / f"{split}.json").write_text(json.dumps(meta, indent=1))
        print(f"{split}: {len(meta)} citra")

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "excreta"},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C", choices=list(CONFIGS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print("Selesai:", build_all(a.config, a.force))
