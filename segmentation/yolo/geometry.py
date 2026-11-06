"""Operasi geometri murni (tanpa I/O): polygon, kotak crop, rasterisasi."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

Box = tuple[int, int, int, int]  # x, y, lebar, tinggi (piksel citra asli)

MIN_POLYGON_AREA = 1.0  # px^2 di ruang input model; di bawah ini dianggap degenerat


def to_polys(ann: dict) -> list[np.ndarray]:
    """Polygon (N, 2) dalam piksel citra asli. RLE ditolak secara eksplisit."""
    segmentation = ann["segmentation"]
    if isinstance(segmentation, list):
        return [
            np.asarray(s, dtype=np.float64).reshape(-1, 2)
            for s in segmentation
            if len(s) >= 6
        ]

    # RLE: decode to binary mask
    mask = mask_utils.decode(segmentation)

    # Handle kemungkinan mask dengan dimensi tambahan (H, W, N)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return [
        contour.reshape(-1, 2).astype(np.float64)
        for contour in contours
        if len(contour) >= 3
    ]


def full_box(width: int, height: int) -> Box:
    return (0, 0, width, height)


def union_box(anns: list[dict], width: int, height: int) -> Box | None:
    """Kotak terluar yang memuat semua bbox anotasi. None bila tidak ada/tidak valid."""
    if not anns:
        return None
    x1 = max(0, math.floor(min(a["bbox"][0] for a in anns)))
    y1 = max(0, math.floor(min(a["bbox"][1] for a in anns)))
    x2 = min(width, math.ceil(max(a["bbox"][0] + a["bbox"][2] for a in anns)))
    y2 = min(height, math.ceil(max(a["bbox"][1] + a["bbox"][3] for a in anns)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def to_crop_space(poly: np.ndarray, box: Box, out_size: int) -> np.ndarray | None:
    """Geser ke koordinat crop, clip ke kotak, skalakan ke out_size.

    Pada mode union-bbox polygon sudah berada di dalam kotak; clip hanya
    pengaman pembulatan. Polygon degenerat menghasilkan None.
    """
    x, y, w, h = box
    moved = np.clip(poly - [x, y], 0, [w, h])
    scaled = moved * [out_size / w, out_size / h]
    if len(scaled) < 3 or polygon_area(scaled) < MIN_POLYGON_AREA:
        return None
    return scaled


def crop_resize(image: np.ndarray, box: Box, out_size: int) -> np.ndarray:
    x, y, w, h = box
    return cv2.resize(
        image[y : y + h, x : x + w],
        (out_size, out_size),
        interpolation=cv2.INTER_LINEAR,
    )


def to_yolo_line(class_id: int, poly: np.ndarray, size: int) -> str:
    coords = (poly / size).clip(0, 1).ravel()
    return f"{class_id} " + " ".join(f"{c:.6f}" for c in coords)


def read_label_polygons(label_path: Path, size: int) -> list[np.ndarray]:
    """Baca file label YOLO-seg kembali menjadi polygon piksel (size x size)."""
    polygons = []
    for line in Path(label_path).read_text().splitlines():
        values = np.asarray(line.split()[1:], dtype=np.float64)
        if values.size >= 6:
            polygons.append(values.reshape(-1, 2) * size)
    return polygons


def rasterize(polygons: list[np.ndarray], height: int, width: int) -> np.ndarray:
    """Masker biner uint8 (0/255) dari kumpulan polygon."""
    mask = np.zeros((height, width), np.uint8)
    if polygons:
        cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in polygons], 255)
    return mask
