from pathlib import Path

import pandas as pd

from utils.file_managements import load_json, save_json

ALLOWED_SPLITS = {"train", "val", "test", "tfub"}


def split_coco_annotations(
    cleaned_json_path: Path,
    split_csv_path: Path,
    annotations_dir: Path,
) -> None:
    """Memecah master COCO JSON berdasarkan split.csv."""

    if not cleaned_json_path.exists():
        raise FileNotFoundError(f"Master JSON tidak ditemukan: {cleaned_json_path}")

    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV tidak ditemukan: {split_csv_path}")

    df_split = pd.read_csv(split_csv_path)
    coco_data = load_json(cleaned_json_path)

    # ------------------------------------------------------------------
    # 1. Validasi struktur split.csv
    # ------------------------------------------------------------------
    required_columns = {
        "image_id",
        "source",
        "split",
    }

    missing_columns = required_columns - set(df_split.columns)

    if missing_columns:
        raise ValueError(f"Kolom CSV tidak lengkap: {sorted(missing_columns)}")

    if df_split["image_id"].isna().any():
        raise ValueError("Ditemukan image_id kosong pada split.csv.")

    if df_split["source"].isna().any():
        raise ValueError("Ditemukan source kosong pada split.csv.")

    if df_split["split"].isna().any():
        raise ValueError("Ditemukan split kosong pada split.csv.")

    if df_split["image_id"].duplicated().any():
        duplicates = df_split.loc[
            df_split["image_id"].duplicated(keep=False),
            "image_id",
        ].tolist()

        raise ValueError(f"Ditemukan duplicate image_id pada split.csv: {duplicates}")

    invalid_splits = set(df_split["split"].unique()) - ALLOWED_SPLITS

    if invalid_splits:
        raise ValueError(f"Split tidak dikenal: {sorted(invalid_splits)}")

    # ------------------------------------------------------------------
    # 2. Ambil data COCO
    # ------------------------------------------------------------------
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])

    if not images:
        raise ValueError("Master JSON tidak memiliki data images.")

    # ------------------------------------------------------------------
    # 3. Petakan image_id berdasarkan filename
    #
    # split.csv menggunakan image_id = filename tanpa extension.
    # Source tidak diturunkan dari filename. Source sudah ditentukan
    # sebelumnya oleh penanggung jawab dataset dan hanya disimpan
    # sebagai metadata di split.csv.
    # ------------------------------------------------------------------
    image_id_to_image = {}

    for img in images:
        image_id = Path(img["file_name"]).stem

        if image_id in image_id_to_image:
            raise ValueError(f"Duplicate image_id dalam master JSON: {image_id}")

        image_id_to_image[image_id] = img

    # ------------------------------------------------------------------
    # 4. Pastikan isi CSV dan master JSON konsisten
    # ------------------------------------------------------------------
    csv_image_ids = set(df_split["image_id"])
    json_image_ids = set(image_id_to_image)

    missing_from_json = csv_image_ids - json_image_ids

    if missing_from_json:
        raise ValueError(
            "Image berikut terdapat di split.csv tetapi "
            "tidak ditemukan dalam master JSON: "
            f"{sorted(missing_from_json)}"
        )

    missing_from_csv = json_image_ids - csv_image_ids

    if missing_from_csv:
        raise ValueError(
            "Image berikut terdapat dalam master JSON tetapi "
            "tidak terdapat di split.csv: "
            f"{sorted(missing_from_csv)}"
        )

    # ------------------------------------------------------------------
    # 5. Petakan COCO image ID -> annotations
    # ------------------------------------------------------------------
    image_id_to_annotations = {}

    for ann in annotations:
        annotation_image_id = ann["image_id"]

        image_id_to_annotations.setdefault(
            annotation_image_id,
            [],
        ).append(ann)

    # Pastikan setiap annotation menunjuk ke image yang valid.
    coco_image_ids = {img["id"] for img in images}

    orphan_annotation_image_ids = set(image_id_to_annotations) - coco_image_ids

    if orphan_annotation_image_ids:
        raise ValueError(
            "Ditemukan annotation yang menunjuk ke image_id "
            "yang tidak ada dalam master JSON: "
            f"{sorted(orphan_annotation_image_ids)}"
        )

    # ------------------------------------------------------------------
    # 6. Buat JSON untuk setiap split
    # ------------------------------------------------------------------
    annotations_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name in sorted(df_split["split"].unique()):
        split_df = df_split[df_split["split"] == split_name]

        # Menggunakan urutan split.csv agar output deterministik.
        split_image_ids = split_df["image_id"].tolist()

        split_imgs = [image_id_to_image[image_id] for image_id in split_image_ids]

        split_anns = []

        for img in split_imgs:
            split_anns.extend(
                image_id_to_annotations.get(
                    img["id"],
                    [],
                )
            )

        subset_coco = {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "categories": coco_data.get("categories", []),
            "images": split_imgs,
            "annotations": split_anns,
        }

        out_json = annotations_dir / f"{split_name}.json"

        save_json(
            subset_coco,
            out_json,
        )

        print(
            f"Selesai [{out_json.name}]: "
            f"{len(split_imgs)} gambar, "
            f"{len(split_anns)} anotasi"
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
