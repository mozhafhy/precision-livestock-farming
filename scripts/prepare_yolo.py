"""Tahap 3 (data detektor ROI): konversi anotasi COCO per subset ke format YOLO.

    python -m scripts.prepare_yolo --root dataset --out dataset/detector
    python -m scripts.prepare_yolo --root dataset --out dataset/detector --task seg   # opsional, bukan bagian Tahap 3

Keluaran:
    <out>/images/{train,val,test}/   salinan citra (raw/ tidak pernah diubah)
    <out>/labels/{train,val,test}/   label YOLO; citra tanpa instans -> file label kosong (hard negative)
    <out>/data.yaml
Hasil training di <out>/runs/ TIDAK ikut terhapus saat konversi diulang.

Dependensi: pip install ultralytics pycocotools opencv-python pyyaml numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray
from pycocotools import mask as mask_utils

# ---------------------------------------------------------------- tipe
Task = Literal["detect", "seg"]
Box = tuple[float, float, float, float]  # xyxy, piksel
Mask = NDArray[np.uint8]


class Rle(TypedDict):
    counts: str | bytes | list[int]
    size: list[int]


Segmentation = Rle | list[list[int]]


class CocoCategory(TypedDict):
    id: int
    name: str
    supercategory: NotRequired[str]


class CocoImage(TypedDict):
    id: int
    file_name: str
    width: int
    height: int


class CocoAnnotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    bbox: list[float]  # xywh, piksel
    segmentation: NotRequired[Segmentation | None]


class CocoDataset(TypedDict):
    categories: list[CocoCategory]
    images: list[CocoImage]
    annotations: list[CocoAnnotation]


SPLITS: list[str] = ["train", "val", "test"]
SPLIT_ALIASES: dict[str, str] = {
    "train": "train",
    "latih": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "validasi": "val",
    "test": "test",
    "uji": "test",
}


# ---------------------------------------------------------------- geometri
def to_binary_mask(seg: Segmentation, h: int, w: int) -> Mask:
    """Segmentasi COCO (RLE terkompresi/tak terkompresi atau list polygon) -> mask biner."""
    if isinstance(seg, dict):  # RLE
        rle: Any = seg
        if isinstance(rle["counts"], list):  # uncompressed RLE
            rle = mask_utils.frPyObjects(rle, *rle["size"])
        if list(rle["size"]) != [h, w]:
            raise ValueError(f"ukuran RLE {rle['size']} != ukuran gambar {[h, w]}")
        return np.asarray(mask_utils.decode(rle), dtype=np.uint8)
    rles: Any = mask_utils.frPyObjects(seg, h, w)  # list polygon
    return np.asarray(mask_utils.decode(mask_utils.merge(rles)), dtype=np.uint8)


def mask_bbox(mask: Mask) -> Box | None:
    """Batas terluar mask -> (xmin, ymin, xmax, ymax) piksel, atau None bila mask kosong."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def iou_xyxy(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_to_polygon(mask: Mask) -> tuple[NDArray[np.float64] | None, int]:
    """Kontur eksternal terbesar. Return (polygon Nx2 piksel, jumlah komponen)."""
    found, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in found if len(c) >= 3]
    if not contours:
        return None, 0
    best = max(contours, key=cv2.contourArea)
    return best.reshape(-1, 2).astype(np.float64), len(contours)


# ---------------------------------------------------------------- validasi split
def check_leakage(coco: dict[str, CocoDataset]) -> dict[str, set[str]]:
    """Satu image_id tidak boleh muncul di lebih dari satu subset (daftar periksa panduan, 10.2)."""
    names = {s: {Path(i["file_name"]).stem for i in coco[s]["images"]} for s in SPLITS}
    bad = False
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = names[a] & names[b]
        if inter:
            print(
                f"[KEBOCORAN] {a} ∩ {b}: {len(inter)} citra, contoh: {sorted(inter)[:5]}"
            )
            bad = True
    if bad:
        sys.exit("Berhenti: ada citra yang muncul di lebih dari satu subset.")
    return names


def check_split_csv(root: Path, names: dict[str, set[str]]) -> None:
    """Bandingkan penugasan subset di JSON dengan split.csv (kolom image_id, split)."""
    p = root / "split.csv"
    if not p.exists():
        print(
            "[peringatan] split.csv tidak ditemukan; verifikasi penugasan subset dilewati"
        )
        return
    with p.open(newline="", encoding="utf-8") as f:
        rows: list[dict[str, str]] = list(csv.DictReader(f))
    if not rows or not {"image_id", "split"} <= set(rows[0]):
        print(
            "[peringatan] split.csv tidak punya kolom image_id/split; verifikasi dilewati"
        )
        return
    ref: dict[str, str | None] = {
        Path(r["image_id"].strip()).stem: SPLIT_ALIASES.get(r["split"].strip().lower())
        for r in rows
    }
    if not set(SPLITS) <= {v for v in ref.values() if v}:
        print(
            "[peringatan] nilai kolom split di split.csv tidak dikenali; verifikasi dilewati"
        )
        return
    mism = [
        (n, s, ref.get(n)) for s in SPLITS for n in sorted(names[s]) if ref.get(n) != s
    ]
    if mism:
        print(
            f"[KETIDAKSESUAIAN] {len(mism)} citra: subset di JSON != split.csv. Contoh (id, JSON, csv):"
        )
        for m in mism[:5]:
            print("   ", m)
        sys.exit("Berhenti: JSON per subset tidak konsisten dengan split.csv.")
    print("split.csv: konsisten dengan JSON train/val/test")


# ---------------------------------------------------------------- konversi
def build_class_map(train_json: CocoDataset) -> tuple[dict[int, int], dict[int, str]]:
    """Buang kategori placeholder Roboflow (supercategory 'none'); id kontigu dari 0."""
    cats = sorted(
        (c for c in train_json["categories"] if c.get("supercategory") != "none"),
        key=lambda c: c["id"],
    )
    id_map = {c["id"]: i for i, c in enumerate(cats)}
    return id_map, {i: c["name"] for i, c in enumerate(cats)}


def convert_split(
    split: str,
    coco: dict[str, CocoDataset],
    root: Path,
    out: Path,
    task: Task,
    id_map: dict[int, int],
    stats: Counter[str],
    bbox_ious: list[float],
    sizes: Counter[tuple[str, tuple[int, int]]],
) -> None:
    data = coco[split]
    img_out = out / "images" / split
    lbl_out = out / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    anns_by_img: dict[int, list[CocoAnnotation]] = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    for im in data["images"]:
        name, w, h = im["file_name"], im["width"], im["height"]
        src = root / "raw" / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, img_out / name)
        sizes[(name.split("_")[0], (w, h))] += 1

        lines: list[str] = []
        for a in anns_by_img.get(im["id"], []):
            if a["category_id"] not in id_map:
                stats["skip_cat"] += 1
                continue
            cls = id_map[a["category_id"]]
            seg = a.get("segmentation")

            if task == "detect":
                x, y, bw, bh = a["bbox"]
                coco_box: Box = (x, y, x + bw, y + bh)
                box: Box = coco_box
                mb: Box | None = None
                if seg:
                    try:
                        mb = mask_bbox(to_binary_mask(seg, h, w))
                    except ValueError as e:
                        stats["size_mismatch"] += 1
                        print(f"[lewati mask] {name}: {e}")
                if mb is None:
                    stats["bbox_fallback"] += 1  # tak ada mask valid -> pakai bbox COCO
                else:
                    box = mb  # panduan 4.3: kotak diturunkan dari batas terluar mask
                    bbox_ious.append(iou_xyxy(mb, coco_box))
                x1, y1, x2, y2 = (
                    max(box[0], 0),
                    max(box[1], 0),
                    min(box[2], w),
                    min(box[3], h),
                )
                if x2 - x1 < 1 or y2 - y1 < 1:
                    stats["degenerate"] += 1
                    continue
                lines.append(
                    f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                )
            else:
                if not seg:
                    stats["empty_mask"] += 1
                    continue
                try:
                    mask = to_binary_mask(seg, h, w)
                except ValueError as e:
                    stats["size_mismatch"] += 1
                    print(f"[lewati mask] {name}: {e}")
                    continue
                poly, n_comp = mask_to_polygon(mask)
                if poly is None:
                    stats["empty_mask"] += 1
                    continue
                if n_comp > 1:
                    stats["multi_component"] += 1
                poly[:, 0] /= w
                poly[:, 1] /= h
                lines.append(
                    f"{cls} " + " ".join(f"{v:.6f}" for v in poly.flatten().clip(0, 1))
                )

        # citra tanpa instans tetap dapat label kosong -> dipelajari sebagai latar belakang / hard negative
        (lbl_out / f"{Path(name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        stats[f"{split}_images"] += 1
        stats[f"{split}_instances"] += len(lines)
        if not lines:
            stats[f"{split}_empty_images"] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset", type=Path)
    ap.add_argument("--out", default="dataset/detector", type=Path)
    ap.add_argument("--task", choices=["detect", "seg"], default="detect")
    args = ap.parse_args()
    task: Task = args.task

    coco: dict[str, CocoDataset] = {
        s: json.loads((args.root / "annotations" / f"{s}.json").read_text())
        for s in SPLITS
    }
    names = check_leakage(coco)
    check_split_csv(args.root, names)

    id_map, class_names = build_class_map(coco["train"])
    print("peta kelas (id COCO -> id YOLO):", id_map, class_names)

    for sub in ("images", "labels"):  # jangan hapus runs/ (bobot terlatih)
        shutil.rmtree(args.out / sub, ignore_errors=True)
    args.out.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    bbox_ious: list[float] = []
    sizes: Counter[tuple[str, tuple[int, int]]] = Counter()
    for s in SPLITS:
        convert_split(
            s, coco, args.root, args.out, task, id_map, stats, bbox_ious, sizes
        )

    cfg: dict[str, Any] = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("\nRingkasan:")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")

    print("\nUkuran citra per sumber (lima teratas):")
    by_src: dict[str, Counter[str]] = {}
    for (src, wh), n in sizes.items():
        by_src.setdefault(src, Counter())[f"{wh[0]}x{wh[1]}"] += n
    for src, c in sorted(by_src.items()):
        print(f"  {src}: {dict(c.most_common(5))}")

    if bbox_ious:
        arr = np.array(bbox_ious)
        print(
            f"\nbbox dari mask vs bbox COCO: IoU rerata {arr.mean():.3f}, min {arr.min():.3f}, "
            f"{int((arr < 0.9).sum())} instans dengan IoU < 0.9"
        )
    if stats["train_images"] and stats["train_empty_images"] == 0:
        print(
            "\n[peringatan] subset latih tidak punya citra tanpa instans: tidak ada hard negative "
            "(panduan 3.1 dan 5.1 mengharapkannya)."
        )

    print(f"\ndata.yaml -> {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
