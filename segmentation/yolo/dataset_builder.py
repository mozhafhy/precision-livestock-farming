"""Konversi COCO (Roboflow) -> dataset YOLO-seg untuk satu kombinasi (variant, mode).

mask_based : seluruh citra + seluruh polygon -> resize ke size x size.
bbox_based : crop ke union bbox semua anotasi pada gambar itu -> resize ke size x size.
Keduanya memakai jalur kode yang sama: mask_based adalah crop dengan kotak penuh.
Satu gambar = satu sampel pada kedua mode, sehingga val/test sebanding.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import yaml
from cv2.typing import MatLike

from .coco import group_by_image, load_coco
from .config import (
    CLASS_MAP,
    CLASS_NAMES,
    MODES,
    SPLITS,
    VARIANTS,
    Config,
    check_choice,
)
from .geometry import (
    Box,
    crop_resize,
    full_box,
    to_crop_space,
    to_polys,
    to_yolo_line,
    union_box,
)
from .history import MANIFEST, SampleRecord, write_manifest
from .logger import get_logger, stage


def read_bgr(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Gagal membaca citra: {path}")
    return image


def check_image_size(
    info: dict,
    source: Path,
    width: int,
    height: int,
) -> None:
    """Polygon JSON hanya valid bila ukuran file sama dengan ukuran di JSON."""
    if (width, height) != (info["width"], info["height"]):
        raise ValueError(
            f"{source}: ukuran file {width}x{height} != JSON {info['width']}x{info['height']}"
        )


def choose_crop_box(
    mode: str,
    anns: list[dict],
    width: int,
    height: int,
) -> tuple[Box, bool]:
    """Kembalikan (box, fallback). fallback=True bila bbox_based tak punya bbox valid."""
    if mode == "mask_based":
        return full_box(width, height), False
    box = union_box(anns, width, height)
    return (box, False) if box else (full_box(width, height), True)


def labeled_polygons(anns: list[dict], box: Box, size: int) -> list[tuple[int, object]]:
    """[(class_id YOLO, polygon di ruang input model)]; polygon degenerat dibuang."""
    result = []
    for ann in anns:
        class_id = CLASS_MAP[ann["category_id"]]
        for poly in to_polys(ann):
            moved = to_crop_space(poly, box, size)
            if moved is not None:
                result.append((class_id, moved))
    return result


def write_image(path: Path, image: MatLike, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"Gagal menulis citra: {path}")


def write_label(path: Path, labeled: list, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(to_yolo_line(c, p, size) + "\n" for c, p in labeled), encoding="utf-8"
    )


def build_sample(
    cfg: Config,
    variant: str,
    mode: str,
    split: str,
    info: dict,
    anns: list[dict],
    out_dir: Path,
) -> SampleRecord:
    source = cfg.variant_image_path(variant, split, info["file_name"])
    image = read_bgr(source)
    height, width = image.shape[:2]
    check_image_size(info, source, width, height)

    box, fallback = choose_crop_box(mode, anns, width, height)
    labeled = labeled_polygons(anns, box, cfg.size)

    stem = Path(info["file_name"]).stem
    image_path = out_dir / "images" / split / f"{stem}.jpg"
    label_path = out_dir / "labels" / split / f"{stem}.txt"
    write_image(image_path, crop_resize(image, box, cfg.size), cfg.jpeg_quality)
    write_label(label_path, labeled, cfg.size)

    return SampleRecord(
        split=split,
        stem=stem,
        image_id=info["id"],
        variant=variant,
        mode=mode,
        raw_image=cfg.relative(cfg.raw_image_path(info["file_name"])),
        source_image=cfg.relative(source),
        orig_size=(width, height),
        crop_box=box,
        crop_fallback=fallback,
        out_size=cfg.size,
        n_polygons=len(labeled),
        image=cfg.relative(image_path),
        label=cfg.relative(label_path),
    )


def build_split(
    cfg: Config, variant: str, mode: str, split: str, out_dir: Path, log
) -> list[SampleRecord]:
    coco = load_coco(cfg.annotation_path(split))
    anns_by_image = group_by_image(coco)
    records = [
        build_sample(
            cfg, variant, mode, split, info, anns_by_image.get(info["id"], []), out_dir
        )
        for info in coco["images"]
    ]
    empty = sum(r.n_polygons == 0 for r in records)
    log.info(
        "%s: %d sampel, %d polygon, %d tanpa polygon (background)",
        split,
        len(records),
        sum(r.n_polygons for r in records),
        empty,
    )
    return records


def write_data_yaml(out_dir: Path) -> None:
    config = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }
    (out_dir / "data.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


def build_dataset(cfg: Config, variant: str, mode: str, log=None) -> Path:
    """Bangun ulang dataset YOLO-seg + manifest. Folder keluaran dihapus dulu (hasil generate)."""
    check_choice(variant, VARIANTS, "variant")
    check_choice(mode, MODES, "mode")
    log = log or get_logger()

    out_dir = cfg.dataset_dir(variant, mode)
    _remove_generated(out_dir, cfg)
    records: list[SampleRecord] = []
    for split in SPLITS:
        with stage(log, f"Build {variant}/{mode}/{split}"):
            records += build_split(cfg, variant, mode, split, out_dir, log)

    write_manifest(out_dir / MANIFEST, records)
    write_data_yaml(out_dir)
    log.info("Dataset siap: %s (%d sampel)", out_dir, len(records))
    return out_dir


def _remove_generated(out_dir: Path, cfg: Config) -> None:
    """Hapus hanya folder di bawah yolo_data/ (pengaman agar tak menghapus sembarang path)."""
    import shutil

    if out_dir.exists() and (cfg.root / "yolo_data") in out_dir.parents:
        shutil.rmtree(out_dir)
