"""Pembacaan dan ringkasan file COCO."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

Coco = dict[str, Any]

def load_coco(path: Path) -> Coco:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def group_by_image(coco: Coco) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        grouped[ann["image_id"]].append(ann)
    return grouped


def summarize(coco: Coco) -> dict[str, Any]:
    """Ringkasan cepat untuk memeriksa asumsi (format segmentasi, id kategori, dst)."""
    anns = coco["annotations"]
    per_image = Counter(ann["image_id"] for ann in anns)
    return {
        "images": len(coco["images"]),
        "annotations": len(anns),
        "images_without_annotation": len(coco["images"]) - len(per_image),
        "annotations_per_image": dict(sorted(Counter(per_image.values()).items())),
        "segmentation_types": dict(
            Counter(type(a["segmentation"]).__name__ for a in anns)
        ),
        "category_ids": dict(Counter(a["category_id"] for a in anns)),
        "image_sizes": dict(Counter((i["width"], i["height"]) for i in coco["images"])),
    }
