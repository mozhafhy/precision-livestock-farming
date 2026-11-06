"""Normalisasi spasial (ROI auto-crop + letterbox) dan spektral (Gray World).

Urutan: crop ROI -> Gray World (statistik hanya dari isi crop, tanpa bingkai) -> letterbox.
Setiap citra menghasilkan objek `Transform` yang cukup untuk membalik prediksi
ke resolusi asli (buang bingkai, resize ke ukuran crop, tempel ke kanvas ukuran asli).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .common import PAD_VALUE, ROI_MARGIN


# --------------------------------------------------------------------------- I/O
def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- COCO
class CocoIndex:
    """Akses anotasi COCO per nama file: bbox, masker instans, masker gabungan."""

    def __init__(self, json_path: str | Path, category_ids: list[int] | None = None):
        with open(json_path, encoding="utf-8") as f:
            coco = json.load(f)
        self.by_name = {Path(im["file_name"]).name: im for im in coco["images"]}
        self.anns: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            if category_ids is None or a["category_id"] in category_ids:
                self.anns.setdefault(a["image_id"], []).append(a)

    def image(self, file_name: str) -> dict:
        return self.by_name[Path(file_name).name]

    def bboxes(self, img: dict) -> list[list[float]]:
        return [a["bbox"] for a in self.anns.get(img["id"], [])]  # xywh

    def instance_masks(self, img: dict) -> list[np.ndarray]:
        h, w = img["height"], img["width"]
        return [_ann_mask(a, h, w) for a in self.anns.get(img["id"], [])]

    def union_mask(self, img: dict) -> np.ndarray:
        masks = self.instance_masks(img)
        if not masks:
            return np.zeros((img["height"], img["width"]), bool)
        return np.any(masks, axis=0)


def _ann_mask(ann: dict, h: int, w: int) -> np.ndarray:
    from pycocotools import mask as mu

    seg = ann.get("segmentation")
    if not seg:
        raise ValueError(f"Anotasi {ann['id']} tidak punya segmentation")
    if isinstance(seg, list):
        rle = mu.merge(mu.frPyObjects(seg, h, w))
    elif isinstance(seg["counts"], list):
        rle = mu.frPyObjects(seg, h, w)
    else:
        rle = seg
    return mu.decode(rle).astype(bool)


# --------------------------------------------------------------------------- Transform
@dataclass
class Transform:
    orig_w: int
    orig_h: int
    crop: tuple  # x0, y0, x1, y1 pada citra asli
    size: int  # sisi kanvas letterbox
    new_w: int  # ukuran isi setelah resize (tanpa bingkai)
    new_h: int
    pad_left: int
    pad_top: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["crop"] = list(self.crop)
        return d

    @staticmethod
    def from_dict(d: dict) -> Transform:
        d = dict(d)
        d["crop"] = tuple(d["crop"])
        return Transform(**d)


# --------------------------------------------------------------------------- Spasial
def roi_box(bboxes: list, w: int, h: int, margin: float = ROI_MARGIN) -> tuple:
    """Union semua bbox COCO (xywh) + margin, dipotong ke batas citra.
    Tanpa bbox -> seluruh citra."""
    full = (0, 0, w, h)
    if not bboxes:
        return full
    b = np.asarray(bboxes, dtype=float)
    x0, y0 = b[:, 0].min(), b[:, 1].min()
    x1, y1 = (b[:, 0] + b[:, 2]).max(), (b[:, 1] + b[:, 3]).max()
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = int(np.floor(max(0, x0 - mx))), int(np.floor(max(0, y0 - my)))
    x1, y1 = int(np.ceil(min(w, x1 + mx))), int(np.ceil(min(h, y1 + my)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return full
    return (x0, y0, x1, y1)


def letterbox(img: np.ndarray, size: int, pad_value: int = PAD_VALUE):
    """Resize dengan rasio tetap ke size x size, bingkai polos di tengah. Mengembalikan (kanvas, tf-parts)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((size, size) + img.shape[2:], pad_value, dtype=img.dtype)
    pl, pt = (size - nw) // 2, (size - nh) // 2
    canvas[pt : pt + nh, pl : pl + nw] = resized
    return canvas, nw, nh, pl, pt


# --------------------------------------------------------------------------- Spektral
def gray_world(rgb: np.ndarray) -> np.ndarray:
    """Gray World: tiap kanal dikali (rerata_abu / rerata_kanal) sehingga rerata R=G=B."""
    x = rgb.astype(np.float32)
    means = x.reshape(-1, 3).mean(0)
    gains = means.mean() / np.maximum(means, 1e-6)
    return np.clip(x * gains, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- Pipeline
def preprocess_image(rgb, bboxes, size, spatial=True, spectral=True, margin=ROI_MARGIN):
    """RGB uint8 asli -> (RGB uint8 size x size, Transform)."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = roi_box(bboxes, w, h, margin) if spatial else (0, 0, w, h)
    crop = rgb[y0:y1, x0:x1]
    if spectral:
        crop = gray_world(crop)
    canvas, nw, nh, pl, pt = letterbox(crop, size)
    tf = Transform(w, h, (x0, y0, x1, y1), size, nw, nh, pl, pt)
    return canvas, tf


def apply_to_mask(mask: np.ndarray, tf: Transform) -> np.ndarray:
    """Terapkan crop + letterbox yang sama pada masker biner asli (nearest). Keluaran uint8 0/1."""
    x0, y0, x1, y1 = tf.crop
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    m = cv2.resize(m, (tf.new_w, tf.new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((tf.size, tf.size), np.uint8)
    canvas[tf.pad_top : tf.pad_top + tf.new_h, tf.pad_left : tf.pad_left + tf.new_w] = m
    return canvas


def invert_to_original(m: np.ndarray, tf: Transform) -> np.ndarray:
    """Peta/masker size x size -> peta float32 pada resolusi asli citra.
    Bingkai dibuang, isi di-resize ke ukuran crop (bilinear), area di luar crop = 0.
    Pemanggil menerapkan ambang pada hasilnya (U-Net: peta probabilitas, YOLO: masker gabungan)."""
    m = m.astype(np.float32)
    inner = m[tf.pad_top : tf.pad_top + tf.new_h, tf.pad_left : tf.pad_left + tf.new_w]
    x0, y0, x1, y1 = tf.crop
    inner = cv2.resize(inner, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((tf.orig_h, tf.orig_w), np.float32)
    out[y0:y1, x0:x1] = inner
    return out
