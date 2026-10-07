from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.file_managements import (
    clean_roboflow_filename,
    load_json,
    save_json,
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_SOURCES = {"PUBLIC", "TFUB"}


def clean_raw_images_and_master_json(
    raw_dir: Path,
    json_path: Path,
) -> Path:
    """Membersihkan filename pada raw/ dan master COCO JSON."""

    if not raw_dir.exists():
        raise FileNotFoundError(f"Direktori raw tidak ditemukan: {raw_dir}")

    if not json_path.exists():
        raise FileNotFoundError(f"Master JSON tidak ditemukan: {json_path}")

    coco_data = load_json(json_path)

    filename_map: dict[str, str] = {}
    cleaned_names: set[str] = set()

    # 1. Bersihkan filename pada JSON
    for img_obj in coco_data.get("images", []):
        old_name = img_obj["file_name"]
        clean_name = clean_roboflow_filename(old_name)

        if clean_name in cleaned_names:
            raise ValueError(f"Filename collision setelah cleaning: {clean_name}")

        filename_map[old_name] = clean_name
        cleaned_names.add(clean_name)
        img_obj["file_name"] = clean_name

    # 2. Rename file fisik
    renamed_count = 0

    for img_file in raw_dir.iterdir():
        if not img_file.is_file():
            continue

        new_name = filename_map.get(img_file.name)

        if new_name is None:
            continue

        new_path = raw_dir / new_name

        if new_path.exists() and new_path != img_file:
            raise FileExistsError(f"Target rename sudah ada: {new_path}")

        if img_file != new_path:
            img_file.rename(new_path)
            renamed_count += 1

    print(f"Berhasil mengubah nama {renamed_count} file di {raw_dir}")

    # 3. Simpan master JSON yang sudah dibersihkan
    cleaned_master_json = json_path.parent / "master_cleaned.json"

    save_json(
        coco_data,
        cleaned_master_json,
    )

    print(f"Master JSON bersih disimpan di: {cleaned_master_json}")

    return cleaned_master_json


def generate_split_csv(
    raw_dir: Path,
    source_csv: Path,
    output_csv: Path,
    seed: int = 42,
) -> None:
    """
    Membuat split.csv berdasarkan source yang telah
    ditentukan secara manual oleh penanggung jawab dataset.
    """

    if not raw_dir.exists():
        raise FileNotFoundError(f"Direktori raw tidak ditemukan: {raw_dir}")

    if not source_csv.exists():
        raise FileNotFoundError(f"Source metadata tidak ditemukan: {source_csv}")

    # ------------------------------------------------------------------
    # 1. Baca source metadata
    # ------------------------------------------------------------------
    df_source = pd.read_csv(source_csv)

    required_columns = {"image_id", "source"}
    missing_columns = required_columns - set(df_source.columns)

    if missing_columns:
        raise ValueError(f"Kolom source CSV tidak lengkap: {sorted(missing_columns)}")

    if df_source["image_id"].isna().any():
        raise ValueError("Ditemukan image_id kosong pada source CSV.")

    if df_source["source"].isna().any():
        raise ValueError("Ditemukan source kosong pada source CSV.")

    if df_source["image_id"].duplicated().any():
        duplicates = df_source.loc[
            df_source["image_id"].duplicated(keep=False),
            "image_id",
        ].tolist()

        raise ValueError(f"Ditemukan duplicate image_id pada source CSV: {duplicates}")

    df_source["source"] = df_source["source"].astype(str).str.strip().str.upper()

    invalid_sources = set(df_source["source"].unique()) - ALLOWED_SOURCES

    if invalid_sources:
        raise ValueError(f"Source tidak dikenal: {sorted(invalid_sources)}")

    # ------------------------------------------------------------------
    # 2. Baca image aktual dari raw/
    # ------------------------------------------------------------------
    records = []

    for img_file in sorted(raw_dir.iterdir()):
        if not img_file.is_file():
            continue

        if img_file.suffix.lower() not in VALID_EXTENSIONS:
            continue

        records.append(
            {
                "image_id": img_file.stem,
                "filename": img_file.name,
            }
        )

    df_images = pd.DataFrame(records)

    if df_images.empty:
        raise ValueError(f"Tidak ada file gambar ditemukan di {raw_dir}")

    if df_images["image_id"].duplicated().any():
        duplicates = df_images.loc[
            df_images["image_id"].duplicated(keep=False),
            "image_id",
        ].tolist()

        raise ValueError(f"Ditemukan duplicate image_id pada raw/: {duplicates}")

    # ------------------------------------------------------------------
    # 3. Pastikan source metadata dan raw/ konsisten
    # ------------------------------------------------------------------
    raw_ids = set(df_images["image_id"])
    source_ids = set(df_source["image_id"])

    missing_source = raw_ids - source_ids

    if missing_source:
        raise ValueError(
            "Image berikut ada di raw/ tetapi belum memiliki "
            f"source: {sorted(missing_source)}"
        )

    missing_images = source_ids - raw_ids

    if missing_images:
        raise ValueError(
            "Image berikut ada di source CSV tetapi tidak "
            f"ditemukan di raw/: {sorted(missing_images)}"
        )

    # Gabungkan metadata source dengan file aktual
    df = df_images.merge(
        df_source,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------------
    # 4. Pisahkan TFUB sebagai external/test set
    # ------------------------------------------------------------------
    df_tfub = df[df["source"] == "TFUB"].copy()
    df_tfub["split"] = "tfub"

    df_public = df[df["source"] == "PUBLIC"].copy()

    if df_public.empty:
        raise ValueError("Tidak ada image PUBLIC untuk train/val/test.")

    # ------------------------------------------------------------------
    # 5. Stratified 70/15/15
    # ------------------------------------------------------------------
    # Saat ini seluruh PUBLIC memiliki source yang sama.
    # Karena itu stratifikasi berdasarkan source tidak bermakna.
    train_df, temp_df = train_test_split(
        df_public,
        test_size=0.30,
        random_state=seed,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # ------------------------------------------------------------------
    # 6. Gabungkan seluruh split
    # ------------------------------------------------------------------
    final_df = pd.concat(
        [
            train_df,
            val_df,
            test_df,
            df_tfub,
        ],
        ignore_index=True,
    )

    final_df = (
        final_df[["image_id", "source", "split"]]
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        output_csv,
        index=False,
    )

    print(f"split.csv berhasil disimpan di: {output_csv}")

    print(
        pd.crosstab(
            final_df["source"],
            final_df["split"],
            margins=True,
        )
    )


if __name__ == "__main__":
    raw_path = Path("dataset/raw")
    master_json = Path("dataset/annotations/_annotations.coco.json")
    source_csv = Path("dataset/metadata/sources.csv")
    split_csv = Path("dataset/split.csv")

    clean_raw_images_and_master_json(
        raw_path,
        master_json,
    )

    generate_split_csv(
        raw_path,
        source_csv,
        split_csv,
    )
