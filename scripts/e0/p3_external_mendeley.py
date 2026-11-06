"""E0 langkah 5: pilih 100 citra Mendeley terstratifikasi, tanpa kembaran citra Kaggle."""

import pandas as pd
from common import (
    DATASET_DIR,
    EXTERNAL_SIZE,
    MAX_HAMMING,
    SPLIT_SEED,
    SPLITS_DIR,
    close_pairs,
    compute_phashes,
    hashes_to_bits,
    index_images,
    read_hash_groups,
    read_metadata,
    sample_stratified,
    write_csv,
)

MDY_IMAGE_DIR = DATASET_DIR / "raw" / "mdy"
OUTPUT_FILE = SPLITS_DIR / "external_mendeley.csv"


def find_leaked_ids(pool: pd.DataFrame, kgl_hashes: list[str]) -> set[str]:
    """Citra Mendeley yang hampir kembar dengan citra Kaggle mana pun."""
    pairs = close_pairs(
        hashes_to_bits(pool["phash"].tolist()), hashes_to_bits(kgl_hashes), MAX_HAMMING
    )
    leaked_rows = pd.Index(sorted({i for i, _ in pairs}))
    return set(pool["image_id"].iloc[leaked_rows])


def main() -> None:
    if OUTPUT_FILE.exists():
        raise FileExistsError(f"{OUTPUT_FILE} sudah ada (dibekukan)")
    pool = read_metadata("mdy")
    paths = index_images(MDY_IMAGE_DIR)
    pool["phash"] = compute_phashes([paths[image_id] for image_id in pool["image_id"]])

    leaked = find_leaked_ids(pool, read_hash_groups()["phash"].tolist())
    clean_pool = pool[~pool["image_id"].isin(leaked)]
    external = sample_stratified(clean_pool, EXTERNAL_SIZE, SPLIT_SEED)

    write_csv(
        external[["image_id", "class"]].rename(columns={"class": "kelas"}), OUTPUT_FILE
    )
    print(f"Pool {len(pool)} citra; kembar dengan Kaggle (dibuang): {len(leaked)}")
    print(external["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
