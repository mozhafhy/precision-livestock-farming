from collections import Counter
from pathlib import Path

import pandas as pd

from utils.file_managements import load_json, save_json

ALLOWED_SPLITS = {"train", "val", "test", "tfub"}
REQUIRED_SPLITS = ("train", "val", "test")
PUBLIC_SOURCES = {"KGL", "MDY"}


def _preview(items, n: int = 10) -> str:
    items = sorted(items, key=str)
    extra = f" ... (+{len(items) - n} lainnya)" if len(items) > n else ""
    return f"{items[:n]}{extra}"


def _load_and_validate_split_csv(split_csv_path: Path) -> pd.DataFrame:
    """Memuat split.csv dan memvalidasi struktur serta konsistensi source/split."""

    # dtype=str: cegah pandas menginterpretasi ulang nilai.
    df = pd.read_csv(split_csv_path, dtype=str)

    required = {"image_id", "source", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom CSV tidak lengkap: {sorted(missing)}")

    for column in required:
        if df[column].isna().any() or (df[column].str.strip() == "").any():
            raise ValueError(f"Ditemukan {column} kosong pada split.csv.")
        df[column] = df[column].str.strip()

    duplicated = df.loc[df["image_id"].duplicated(keep=False), "image_id"]
    if not duplicated.empty:
        raise ValueError(
            f"Duplicate image_id pada split.csv: {_preview(set(duplicated))}"
        )

    invalid_splits = set(df["split"]) - ALLOWED_SPLITS
    if invalid_splits:
        raise ValueError(f"Split tidak dikenal: {sorted(invalid_splits)}")

    # image_id = <SOURCE>_<NOMOR>; kolom source harus sama dengan prefiksnya.
    prefix = df["image_id"].str.split("_", n=1).str[0]
    mismatch = df.loc[prefix != df["source"], "image_id"]
    if not mismatch.empty:
        raise ValueError(
            "Kolom source tidak sama dengan prefiks image_id: "
            f"{_preview(set(mismatch))}"
        )

    # Spesifikasi 3.3 butir 3: TFUB tidak masuk subset mana pun,
    # dan citra publik tidak boleh masuk subset tfub.
    bad_tfub = df[(df["source"] == "TFUB") != (df["split"] == "tfub")]
    if not bad_tfub.empty:
        raise ValueError(
            "Split 'tfub' hanya boleh untuk source TFUB (dan sebaliknya): "
            f"{_preview(set(bad_tfub['image_id']))}"
        )

    bad_public = df[
        df["source"].isin(PUBLIC_SOURCES) & ~df["split"].isin(REQUIRED_SPLITS)
    ]
    if not bad_public.empty:
        raise ValueError(
            "Citra KGL/MDY harus berada di train/val/test: "
            f"{_preview(set(bad_public['image_id']))}"
        )

    present = set(df["split"])
    missing_splits = set(REQUIRED_SPLITS) - present
    if missing_splits:
        raise ValueError(f"Split wajib tidak memiliki data: {sorted(missing_splits)}")

    return df


def _validate_coco(coco_data: dict) -> None:
    """Validasi integritas master COCO (ID unik, referensi valid)."""

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    if not images:
        raise ValueError("Master JSON tidak memiliki data images.")

    if not categories:
        raise ValueError("Master JSON tidak memiliki categories.")

    for name, items in (
        ("image", images),
        ("annotation", annotations),
        ("category", categories),
    ):
        ids = [item.get("id") for item in items]
        if any(i is None for i in ids):
            raise ValueError(f"Ada {name} tanpa field 'id' pada master JSON.")
        duplicates = [i for i, n in Counter(ids).items() if n > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate {name} id pada master JSON: {_preview(duplicates)}"
            )

    image_ids = {img["id"] for img in images}
    category_ids = {cat["id"] for cat in categories}

    for ann in annotations:
        if "image_id" not in ann or "category_id" not in ann:
            raise ValueError(f"Annotation {ann.get('id')} tanpa image_id/category_id.")

    orphan_images = {a["image_id"] for a in annotations} - image_ids
    if orphan_images:
        raise ValueError(
            f"Annotation menunjuk ke image yang tidak ada: {_preview(orphan_images)}"
        )

    unknown_categories = {a["category_id"] for a in annotations} - category_ids
    if unknown_categories:
        raise ValueError(
            "Annotation memakai category_id yang tidak terdaftar: "
            f"{_preview(unknown_categories)}"
        )

    # Tugas ini instance segmentation: anotasi tanpa polygon tidak bisa
    # dijadikan mask pada Tahap 4. Tidak fatal di sini, tetapi harus terlihat.
    no_mask = [a["id"] for a in annotations if not a.get("segmentation")]
    if no_mask:
        print(
            f"PERINGATAN: {len(no_mask)} anotasi tanpa segmentation (mask kosong), "
            f"contoh id: {_preview(no_mask, 5)}"
        )


def split_coco_annotations(
    cleaned_json_path: Path,
    split_csv_path: Path,
    annotations_dir: Path,
) -> None:
    """
    Memecah master COCO JSON menjadi train/val/test/tfub.json berdasarkan
    split.csv (sumber kebenaran pembagian; hanya dibaca, tidak diubah).

    Citra tanpa anotasi tetap dipertahankan (hard negative, spesifikasi 3.1).
    Seluruh categories disalin ke setiap subset agar category_id konsisten.
    """

    if not cleaned_json_path.exists():
        raise FileNotFoundError(f"Master JSON tidak ditemukan: {cleaned_json_path}")

    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV tidak ditemukan: {split_csv_path}")

    df_split = _load_and_validate_split_csv(split_csv_path)
    coco_data = load_json(cleaned_json_path)
    _validate_coco(coco_data)

    images = coco_data["images"]
    annotations = coco_data.get("annotations", [])

    # split.csv.image_id == stem file_name (mis. KGL_0001)
    stem_to_image = {}
    for img in images:
        stem = Path(img["file_name"]).stem
        if stem in stem_to_image:
            raise ValueError(f"Duplicate image_id (stem file_name) pada master: {stem}")
        stem_to_image[stem] = img

    csv_ids, json_ids = set(df_split["image_id"]), set(stem_to_image)

    if csv_ids - json_ids:
        raise ValueError(
            "Ada di split.csv tetapi tidak di master JSON: "
            f"{_preview(csv_ids - json_ids)}"
        )

    if json_ids - csv_ids:
        raise ValueError(
            "Ada di master JSON tetapi tidak di split.csv: "
            f"{_preview(json_ids - csv_ids)}"
        )

    annotations_by_image = {}
    for ann in annotations:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    annotations_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    for split_name in sorted(ALLOWED_SPLITS):
        split_ids = df_split.loc[df_split["split"] == split_name, "image_id"].tolist()

        if not split_ids:
            print(f"PERINGATAN: split '{split_name}' kosong; berkas tidak dibuat.")
            continue

        split_imgs = [stem_to_image[i] for i in split_ids]
        split_anns = [
            ann for img in split_imgs for ann in annotations_by_image.get(img["id"], [])
        ]

        subset = {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "categories": coco_data["categories"],
            "images": split_imgs,
            "annotations": split_anns,
        }

        out_json = annotations_dir / f"{split_name}.json"
        save_json(subset, out_json)
        outputs[split_name] = out_json

        empty = sum(1 for img in split_imgs if img["id"] not in annotations_by_image)
        print(
            f"Selesai [{out_json.name}]: {len(split_imgs)} gambar "
            f"({empty} tanpa anotasi), {len(split_anns)} anotasi"
        )

    _verify_outputs(outputs, coco_data)


def _verify_outputs(outputs: dict, coco_data: dict) -> None:
    """Baca ulang berkas keluaran dan pastikan tidak ada kebocoran/kehilangan."""

    seen_images, seen_anns = {}, set()

    for split_name, path in outputs.items():
        subset = load_json(path)

        for img in subset["images"]:
            if img["id"] in seen_images:
                raise RuntimeError(
                    f"Image {img['file_name']} muncul di '{seen_images[img['id']]}' "
                    f"dan '{split_name}' (kebocoran data)."
                )
            seen_images[img["id"]] = split_name

        img_ids = {img["id"] for img in subset["images"]}
        for ann in subset["annotations"]:
            if ann["image_id"] not in img_ids:
                raise RuntimeError(
                    f"Annotation {ann['id']} tidak sesuai subset {split_name}."
                )
            seen_anns.add(ann["id"])

    if len(seen_images) != len(coco_data["images"]):
        raise RuntimeError(
            f"Jumlah image keluaran ({len(seen_images)}) != master "
            f"({len(coco_data['images'])})."
        )

    if len(seen_anns) != len(coco_data.get("annotations", [])):
        raise RuntimeError(
            f"Jumlah annotation keluaran ({len(seen_anns)}) != master "
            f"({len(coco_data.get('annotations', []))})."
        )

    print(
        "Verifikasi keluaran lolos: tidak ada image di >1 subset, tidak ada yang hilang."
    )


if __name__ == "__main__":
    cleaned_json = Path("dataset/annotations/master_cleaned.json")
    split_csv = Path("dataset/split.csv")
    annotations_dir = Path("dataset/annotations")

    split_coco_annotations(
        cleaned_json,
        split_csv,
        annotations_dir,
    )
