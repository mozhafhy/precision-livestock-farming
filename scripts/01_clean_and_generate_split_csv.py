import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.file_managements import (
    load_json,
    save_json,
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_SOURCES = {"KGL", "MDY", "TFUB"}
PUBLIC_SOURCES = {"KGL", "MDY"}

# Lebar zero-padding nomor pada nama file / image_id, mis. KGL_0001.
ID_WIDTH = 4


def _format_number(image_id: str) -> str:
    """ID COCO -> nomor ber-zero-padding (selalu minimal ID_WIDTH digit)."""

    if not image_id.isdigit():
        raise ValueError(
            f"ID COCO harus berupa bilangan bulat, ditemukan: {image_id!r}"
        )

    return image_id.zfill(ID_WIDTH)


def load_source_metadata(
    source_csv: Path,
) -> pd.DataFrame:
    """Memuat dan memvalidasi metadata source."""

    if not source_csv.exists():
        raise FileNotFoundError(f"Source metadata tidak ditemukan: {source_csv}")

    # dtype=str: mencegah pandas membuang leading zero / mengubah ID
    # menjadi float, yang akan membuat lookup terhadap COCO ID meleset.
    df = pd.read_csv(source_csv, dtype=str)

    required_columns = {"image_id", "source"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Kolom source CSV tidak lengkap: {sorted(missing_columns)}")

    if df["image_id"].isna().any() or df["source"].isna().any():
        raise ValueError("Ditemukan image_id atau source kosong pada source CSV.")

    # Kolom image_id pada sources.csv = stem nama file ASLI (tanpa ekstensi),
    # mis. "cocci-0_jpg.rf.70uD...". Ini BUKAN ID integer COCO.
    # Jika ekstensi ikut tertulis, dibuang agar kunci konsisten.
    def _strip_ext(value: str) -> str:
        value = value.strip()
        if Path(value).suffix.lower() in VALID_EXTENSIONS:
            return value[: -len(Path(value).suffix)]
        return value

    df["image_id"] = df["image_id"].map(_strip_ext)
    df["source"] = df["source"].str.strip().str.upper()

    if (df["image_id"] == "").any() or (df["source"] == "").any():
        raise ValueError("Ditemukan image_id atau source kosong pada source CSV.")

    if df["image_id"].duplicated().any():
        duplicates = df.loc[df["image_id"].duplicated(keep=False), "image_id"]
        raise ValueError(
            f"Ditemukan duplicate image_id pada source CSV: {sorted(set(duplicates))}"
        )

    invalid_sources = set(df["source"].unique()) - ALLOWED_SOURCES

    if invalid_sources:
        raise ValueError(f"Source tidak dikenal: {sorted(invalid_sources)}")

    return df


def _get_image_id(image: dict) -> str:
    """Mengambil image ID (integer COCO) sebagai string."""

    if "id" not in image:
        raise ValueError("Image pada COCO JSON tidak memiliki field 'id'.")

    return str(image["id"])


def _resolve_source_map(
    images: list,
    df_source: pd.DataFrame,
) -> dict:
    """
    Memetakan COCO image id -> source.

    Kolom image_id di sources.csv bisa berisi COCO id, nama file, atau
    nama file tanpa ekstensi. Strategi dengan kecocokan terbanyak dipilih;
    jika tidak ada yang cocok penuh, error menampilkan contoh kedua sisi.
    """

    raw = dict(zip(df_source["image_id"], df_source["source"]))

    ids = [_get_image_id(image) for image in images]
    names = [Path(image["file_name"]).name for image in images]

    strategies = {
        "COCO id": ids,
        "nama file": names,
        "nama file tanpa ekstensi": [Path(n).stem for n in names],
    }

    label, keys = max(
        strategies.items(),
        key=lambda item: sum(k in raw for k in item[1]),
    )

    missing = [i for i, k in zip(ids, keys) if k not in raw]
    extra = sorted(set(raw) - set(keys))

    if missing or extra:
        raise ValueError(
            f"sources.csv tidak cocok penuh dengan COCO JSON "
            f"(strategi terbaik: {label}).\n"
            f"  Tidak ada di sources.csv: {len(missing)} image, contoh id: {missing[:5]}\n"
            f"  Tidak ada di COCO JSON  : {len(extra)} baris, contoh: {extra[:5]}\n"
            f"  Contoh image_id sources.csv: {list(raw)[:3]}\n"
            f"  Contoh COCO id/file_name   : {ids[:3]} / {names[:3]}"
        )

    print(f"sources.csv dicocokkan berdasarkan: {label}")

    return {i: raw[k] for i, k in zip(ids, keys)}


def clean_raw_images_and_master_json(
    raw_dir: Path,
    json_path: Path,
    source_csv: Path,
) -> Path:
    """
    Rename file di raw/ menjadi <SOURCE>_<COCO_IMAGE_ID zero-padded>.<EXT>
    (mis. KGL_0304.jpg); file berformat lama (KGL_304.jpg) ikut dimigrasi.
    Selanjutnya
    tulis master_cleaned.json dengan file_name yang diperbarui.

    Idempotent. Seluruh rencana rename divalidasi lebih dulu; tidak ada
    file yang disentuh sebelum semua validasi lolos.
    """

    if not raw_dir.exists():
        raise FileNotFoundError(f"Direktori raw tidak ditemukan: {raw_dir}")

    if not json_path.exists():
        raise FileNotFoundError(f"Master JSON tidak ditemukan: {json_path}")

    df_source = load_source_metadata(source_csv)

    coco_data = load_json(json_path)
    images = coco_data.get("images", [])

    if not images:
        raise ValueError("Master JSON tidak memiliki data images.")

    coco_ids = [_get_image_id(image) for image in images]

    # 1. Duplicate ID pada COCO
    duplicates = sorted(i for i, n in Counter(coco_ids).items() if n > 1)
    if duplicates:
        raise ValueError(f"Duplicate image ID pada COCO JSON: {duplicates}")

    # 2. COCO <-> sources.csv harus identik (dua arah)
    source_map = _resolve_source_map(images, df_source)

    # 3. Susun rencana rename (belum ada side effect)
    plan = []
    target_names = set()

    for image in images:
        image_id = _get_image_id(image)
        source = source_map[image_id]

        current_name = Path(image["file_name"]).name
        extension = Path(current_name).suffix.lower()

        if extension not in VALID_EXTENSIONS:
            raise ValueError(
                f"Extension tidak didukung untuk image {image_id}: {current_name}"
            )

        target_name = f"{source}_{_format_number(image_id)}{extension}"

        if target_name in target_names:
            raise ValueError(f"Filename target duplicate: {target_name}")
        target_names.add(target_name)

        target_path = raw_dir / target_name

        # Kandidat sumber rename, berurutan:
        #   1. nama asli (belum pernah dibersihkan)
        #   2. nama format lama tanpa padding, mis. KGL_1.jpg (run versi lama)
        legacy_name = f"{source}_{image_id}{extension}"
        candidates = [raw_dir / current_name]
        if legacy_name != target_name:
            candidates.append(raw_dir / legacy_name)

        existing = [p for p in candidates if p.exists() and p != target_path]

        if target_path.exists():
            if existing:
                raise FileExistsError(
                    f"Target sudah ada untuk image {image_id}: {target_path}, "
                    f"tetapi file sumber juga masih ada: {existing[0]}"
                )
            action, current_path = "keep", target_path  # sudah bersih
        elif existing:
            action, current_path = "rename", existing[0]
        else:
            raise FileNotFoundError(
                f"File untuk image {image_id} tidak ditemukan. Diharapkan salah "
                f"satu dari: {[str(p) for p in candidates]} | {target_path}"
            )

        plan.append((image, action, current_path, target_path, target_name))

    # 4. Eksekusi rename + update JSON dalam satu pass (O(n))
    renamed_count = 0
    already_clean_count = 0

    for image, action, current_path, target_path, target_name in plan:
        if action == "rename":
            current_path.rename(target_path)
            renamed_count += 1
        else:
            already_clean_count += 1

        image["file_name"] = target_name

    print(f"Berhasil mengubah nama {renamed_count} file di {raw_dir}")
    print(f"File yang sudah sesuai: {already_clean_count}")

    # 5. File yatim di raw/ (tidak terdaftar di COCO) -> berpotensi bocor/kotor
    orphans = sorted(
        p.name for p in raw_dir.iterdir() if p.is_file() and p.name not in target_names
    )
    if orphans:
        print(
            f"PERINGATAN: {len(orphans)} file di raw/ tidak ada di COCO JSON, "
            f"contoh: {orphans[:5]}"
        )

    cleaned_master_json = json_path.parent / "master_cleaned.json"
    save_json(coco_data, cleaned_master_json)

    print(f"Master JSON bersih disimpan di: {cleaned_master_json}")

    return cleaned_master_json


def generate_split_csv(
    raw_dir: Path,
    cleaned_json: Path,
    output_csv: Path,
    seed: int = 42,
    overwrite: bool = False,
) -> None:
    """
    Membuat split.csv (kolom: image_id, source, split).

    image_id berformat <SOURCE>_<COCO_ID zero-padded> (mis. KGL_0304).

    KGL + MDY : train / val / test = 70 / 15 / 15, stratified per source
    TFUB      : "tfub" (tidak masuk subset mana pun)

    split.csv bersifat beku: tidak ditimpa kecuali overwrite=True.
    """

    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"{output_csv} sudah ada dan dibekukan. "
            "Hapus manual atau gunakan overwrite=True jika memang disengaja."
        )

    if not raw_dir.exists():
        raise FileNotFoundError(f"Direktori raw tidak ditemukan: {raw_dir}")

    if not cleaned_json.exists():
        raise FileNotFoundError(f"Master JSON tidak ditemukan: {cleaned_json}")

    # Setelah cleaning, <SOURCE> dan <COCO_ID> sudah tertanam di file_name,
    # sehingga source dibaca dari sana (sources.csv sudah dipakai di cleaning,
    # dan kuncinya mungkin nama file lama yang kini sudah tidak ada).
    images = load_json(cleaned_json).get("images", [])

    if not images:
        raise ValueError("Master JSON tidak memiliki data images.")

    records = []

    for image in images:
        coco_id = _get_image_id(image)
        filename = Path(image["file_name"]).name
        stem = Path(filename).stem
        source, _, number = stem.partition("_")

        if source not in ALLOWED_SOURCES or number != _format_number(coco_id):
            raise ValueError(
                f"file_name '{filename}' tidak mengikuti format "
                f"<SOURCE>_{_format_number(coco_id)}.<ext>. Gunakan master_cleaned.json "
                "hasil clean_raw_images_and_master_json."
            )

        if not (raw_dir / filename).exists():
            raise FileNotFoundError(f"File tidak ditemukan di raw/: {filename}")

        records.append({"image_id": stem, "source": source})

    df = pd.DataFrame(records)

    if df["image_id"].duplicated().any():
        raise ValueError("image_id duplikat terdeteksi saat membangun split.")

    df_tfub = df[df["source"] == "TFUB"].copy()
    df_tfub["split"] = "tfub"

    df_public = df[df["source"].isin(PUBLIC_SOURCES)].copy()

    if df_public.empty:
        raise ValueError("Tidak ada image KGL/MDY untuk train/val/test.")

    # Split 70 / 30, stratified per source.
    # train_test_split sudah melempar ValueError jika kelas terlalu kecil,
    # tetapi pesan eksplisit lebih mudah didiagnosis.
    for name, frame in (("public", df_public),):
        counts = frame["source"].value_counts()
        if (counts < 4).any():
            raise ValueError(
                f"Tiap source di subset {name} butuh minimal 4 image agar "
                f"val/test (15/15) dapat terstratifikasi: {counts.to_dict()}"
            )

    train_df, temp_df = train_test_split(
        df_public,
        test_size=0.30,
        random_state=seed,
        stratify=df_public["source"],
    )

    # 30% -> 15% val + 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["source"],
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    final_df = (
        pd.concat([train_df, val_df, test_df, df_tfub], ignore_index=True)
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    # Invarian: tidak ada image_id di >1 subset, tidak ada yang hilang.
    if final_df["image_id"].duplicated().any():
        raise RuntimeError("image_id muncul pada lebih dari satu subset.")

    if len(final_df) != len(df):
        raise RuntimeError(
            f"Jumlah baris split ({len(final_df)}) != jumlah image ({len(df)})."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    # Seed dicatat sesuai spesifikasi (Bagian 3.3 butir 4).
    meta_path = output_csv.with_name("split_meta.json")
    meta = {
        "seed": seed,
        "ratio_public": {"train": 0.70, "val": 0.15, "test": 0.15},
        "counts": {
            split: int(n) for split, n in final_df["split"].value_counts().items()
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"split.csv berhasil disimpan di: {output_csv}")
    print(f"Metadata split (seed) disimpan di: {meta_path}")
    print(pd.crosstab(final_df["source"], final_df["split"], margins=True))


if __name__ == "__main__":
    raw_path = Path("dataset/raw")
    master_json = Path("dataset/annotations/_annotations.coco.json")
    source_csv = Path("dataset/metadata/sources.csv")
    split_csv = Path("dataset/split.csv")

    cleaned_json = clean_raw_images_and_master_json(
        raw_path,
        master_json,
        source_csv,
    )

    generate_split_csv(
        raw_path,
        cleaned_json,
        split_csv,
    )
