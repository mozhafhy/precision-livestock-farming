"""E0: bersihkan nama gambar Roboflow dan sinkronkan COCO JSON.

Jalankan dari root proyek:  python -m scripts.e0_clean_dataset
"""

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

SOURCE = "kgl"
RAW_DIR = Path("tmp/res") / SOURCE
IMAGE_OUT_DIR = Path("dataset/raw") / SOURCE
ANNOTATION_OUT_FILE = Path("dataset/annotations") / SOURCE / "annotations.coco.json"
METADATA_FILE = Path("dataset") / f"{SOURCE}_metadata.csv"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ROBOFLOW_SEPARATOR = ".rf."
INDEX_WIDTH = 4
# "pcr" = awalan opsional (mis. pcrncd, pcrsalmo); kelas tetap bagian setelahnya.
CLASS_PATTERN = re.compile(r"^(?:pcr)?(cocci|healthy|salmo|ncd)", re.IGNORECASE)


@dataclass(frozen=True)
class CleanImage:
    raw_name: str
    image_id: str
    new_name: str
    label: str


def extract_prefix(raw_name: str) -> str:
    prefix, separator, _ = raw_name.partition(ROBOFLOW_SEPARATOR)
    if not separator:
        raise ValueError(f"Bukan format Roboflow: {raw_name}")
    return prefix


def extract_label(prefix: str) -> str:
    match = CLASS_PATTERN.match(prefix)
    if not match:
        raise ValueError(f"Kelas tidak dikenali dari prefix: {prefix}")
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


def plan_clean_images(raw_names) -> list[CleanImage]:
    plan = []
    for index, raw_name in enumerate(sorted(raw_names)):
        image_id = f"{SOURCE.upper()}_{index:0{INDEX_WIDTH}d}"
        plan.append(
            CleanImage(
                raw_name=raw_name,
                image_id=image_id,
                new_name=image_id + Path(raw_name).suffix.lower(),
                label=extract_label(extract_prefix(raw_name)),
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


def refuse_overwrite() -> None:
    image_dir_in_use = IMAGE_OUT_DIR.exists() and any(IMAGE_OUT_DIR.iterdir())
    if image_dir_in_use or ANNOTATION_OUT_FILE.exists() or METADATA_FILE.exists():
        raise FileExistsError(
            "Keluaran sudah ada; hapus manual sebelum menjalankan ulang"
        )


def copy_images(plan: list[CleanImage], images_on_disk: dict[str, Path]) -> None:
    IMAGE_OUT_DIR.mkdir(parents=True)
    for item in plan:
        shutil.copy2(images_on_disk[item.raw_name], IMAGE_OUT_DIR / item.new_name)


def write_coco(coco: dict) -> None:
    ANNOTATION_OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATION_OUT_FILE.write_text(json.dumps(coco), encoding="utf-8")


def write_metadata(plan: list[CleanImage]) -> None:
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_FILE.open("w", newline="", encoding="utf-8") as file:
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


def main() -> None:
    refuse_overwrite()
    images_on_disk = index_images_on_disk(RAW_DIR)
    documents = load_coco_documents(RAW_DIR)
    coco_entries = index_coco_entries(documents)

    synced_names = images_on_disk.keys() & coco_entries.keys()
    plan = plan_clean_images(synced_names)

    copy_images(plan, images_on_disk)
    write_coco(build_coco(shared_categories(documents), plan, coco_entries))
    write_metadata(plan)
    report(
        plan, images_on_disk.keys() - synced_names, coco_entries.keys() - synced_names
    )


if __name__ == "__main__":
    main()
