"""E0: bersihkan nama gambar Roboflow dan sinkronkan COCO JSON.

Jalankan dari root proyek:
    python -m scripts.e0.p0_clean_dataset kgl
    python -m scripts.e0.p0_clean_dataset abc --classes cocci healthy --strip-prefix ""
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

RES_ROOT = Path("tmp/res")
DATASET_ROOT = Path("dataset")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ROBOFLOW_SEPARATOR = ".rf."
INDEX_WIDTH = 4
DEFAULT_CLASSES = ("cocci", "healthy", "salmo", "ncd")
DEFAULT_STRIP_PREFIX = "pcr"


@dataclass(frozen=True)
class Config:
    source: str
    raw_dir: Path
    image_out_dir: Path
    annotation_out_file: Path
    metadata_file: Path
    class_pattern: re.Pattern

    @classmethod
    def build(
        cls, source: str, classes=DEFAULT_CLASSES, strip_prefix=DEFAULT_STRIP_PREFIX
    ) -> Config:
        if not classes:
            raise ValueError("Daftar kelas tidak boleh kosong")
        alternatives = "|".join(re.escape(c) for c in classes)
        optional_prefix = f"(?:{re.escape(strip_prefix)})?" if strip_prefix else ""
        return cls(
            source=source,
            raw_dir=RES_ROOT / source,
            image_out_dir=DATASET_ROOT / "raw" / source,
            annotation_out_file=DATASET_ROOT
            / "annotations"
            / source
            / "annotations.coco.json",
            metadata_file=DATASET_ROOT / f"{source}_metadata.csv",
            class_pattern=re.compile(
                rf"^{optional_prefix}({alternatives})", re.IGNORECASE
            ),
        )


@dataclass(frozen=True)
class CleanImage:
    raw_name: str
    image_id: str
    new_name: str
    label: str | None = None


def extract_prefix(raw_name: str) -> str:
    prefix, separator, _ = raw_name.partition(ROBOFLOW_SEPARATOR)
    if not separator:
        raise ValueError(f"Bukan format Roboflow: {raw_name}")
    return prefix


def extract_label(prefix: str, class_pattern: re.Pattern) -> str | None:
    match = class_pattern.match(prefix)
    if not match:
        return None
    return match.group(1).lower()


def index_images_on_disk(raw_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.name in images:
            raise ValueError(f"Nama gambar ganda: {path.name}")
        images[path.name] = path
    return images


def load_coco_documents(raw_dir: Path) -> list[dict]:
    paths = sorted(raw_dir.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"Tidak ada COCO JSON di {raw_dir}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def shared_categories(documents: list[dict]) -> list[dict]:
    categories = documents[0]["categories"]
    if any(document["categories"] != categories for document in documents):
        raise ValueError("Kategori antar-COCO JSON berbeda")
    return categories


def index_coco_entries(documents: list[dict]) -> dict[str, tuple[dict, list[dict]]]:
    """file_name -> (data gambar, daftar anotasinya)."""
    entries: dict[str, tuple[dict, list[dict]]] = {}
    for document in documents:
        annotations_by_image = defaultdict(list)
        for annotation in document["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)
        for image in document["images"]:
            name = Path(image["file_name"]).name
            if name in entries:
                raise ValueError(f"file_name ganda di COCO JSON: {name}")
            entries[name] = (image, annotations_by_image[image["id"]])
    return entries


def plan_clean_images(raw_names, config: Config) -> list[CleanImage]:
    plan = []
    for index, raw_name in enumerate(sorted(raw_names)):
        image_id = f"{config.source.upper()}_{index:0{INDEX_WIDTH}d}"
        plan.append(
            CleanImage(
                raw_name=raw_name,
                image_id=image_id,
                new_name=image_id + Path(raw_name).suffix.lower(),
                label=extract_label(extract_prefix(raw_name), config.class_pattern),
            )
        )
    return plan


def build_coco(categories, plan, coco_entries) -> dict:
    images, annotations = [], []
    for index, item in enumerate(plan):
        source_image, source_annotations = coco_entries[item.raw_name]
        images.append({**source_image, "id": index, "file_name": item.new_name})
        for annotation in source_annotations:
            annotations.append(
                {**annotation, "id": len(annotations), "image_id": index}
            )
    return {"categories": categories, "images": images, "annotations": annotations}


def refuse_overwrite(config: Config) -> None:
    image_dir_in_use = config.image_out_dir.exists() and any(
        config.image_out_dir.iterdir()
    )
    if (
        image_dir_in_use
        or config.annotation_out_file.exists()
        or config.metadata_file.exists()
    ):
        raise FileExistsError(
            f"Keluaran untuk '{config.source}' sudah ada; "
            "hapus manual sebelum menjalankan ulang"
        )


def copy_images(
    plan: list[CleanImage], images_on_disk: dict[str, Path], config: Config
) -> None:
    config.image_out_dir.mkdir(parents=True, exist_ok=True)
    for item in plan:
        shutil.copy2(
            images_on_disk[item.raw_name], config.image_out_dir / item.new_name
        )


def write_coco(coco: dict, config: Config) -> None:
    config.annotation_out_file.parent.mkdir(parents=True, exist_ok=True)
    config.annotation_out_file.write_text(json.dumps(coco), encoding="utf-8")


def write_metadata(plan: list[CleanImage], config: Config) -> None:
    config.metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with config.metadata_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "class", "raw_name"])
        writer.writerows((i.image_id, i.label, i.raw_name) for i in plan)


def report(plan, only_on_disk, only_in_json) -> None:
    print(f"Disinkronkan : {len(plan)} gambar")
    print(f"Per kelas    : {dict(Counter(item.label for item in plan))}")
    print(f"Tanpa JSON   : {len(only_on_disk)} gambar (dilewati)")
    print(f"Tanpa gambar : {len(only_in_json)} entri JSON (dilewati)")
    prefix_counts = Counter(extract_prefix(item.raw_name) for item in plan)
    duplicates = {p: n for p, n in prefix_counts.items() if n > 1}
    if duplicates:
        print(
            f"PERINGATAN   : {len(duplicates)} prefix muncul >1x (kemungkinan "
            "augmentasi Roboflow); periksa sebelum split"
        )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "Cleaner").splitlines()[0])
    parser.add_argument("source", help="nama subfolder di tmp/res/ (mis. kgl)")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help=f"daftar kelas valid (default: {' '.join(DEFAULT_CLASSES)})",
    )
    parser.add_argument(
        "--strip-prefix",
        default=DEFAULT_STRIP_PREFIX,
        help="awalan opsional sebelum nama kelas; string kosong untuk menonaktifkan",
    )
    return parser.parse_args(argv)


def run(config: Config) -> None:
    if not config.raw_dir.is_dir():
        raise FileNotFoundError(f"Source tidak ditemukan: {config.raw_dir}")
    refuse_overwrite(config)
    images_on_disk = index_images_on_disk(config.raw_dir)
    documents = load_coco_documents(config.raw_dir)
    coco_entries = index_coco_entries(documents)

    synced_names = images_on_disk.keys() & coco_entries.keys()
    plan = plan_clean_images(synced_names, config)

    copy_images(plan, images_on_disk, config)
    write_coco(build_coco(shared_categories(documents), plan, coco_entries), config)
    write_metadata(plan, config)
    report(
        plan, images_on_disk.keys() - synced_names, coco_entries.keys() - synced_names
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    run(Config.build(args.source, args.classes, args.strip_prefix))


if __name__ == "__main__":
    main()
