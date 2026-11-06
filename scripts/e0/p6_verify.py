"""E0: verifikasi kriteria selesai dan keutuhan file beku."""
import pandas as pd
import yaml
from common import PROTOCOL_FILE, SPLIT_SIZES, SPLITS_DIR, sha256_of

PROPORTION_TOLERANCE = 0.02


def load_splits() -> pd.DataFrame:
    parts = [pd.read_csv(SPLITS_DIR / f"{name}.csv").assign(split=name) for name in SPLIT_SIZES]
    return pd.concat(parts, ignore_index=True)


def check_no_leakage(splits: pd.DataFrame) -> None:
    if not splits["image_id"].is_unique:
        raise ValueError("Ada image_id di lebih dari satu split")
    spread = splits.groupby("hash_group")["split"].nunique()
    if (spread > 1).any():
        raise ValueError(f"{(spread > 1).sum()} kelompok hash ada di lebih dari satu split")


def check_sizes(splits: pd.DataFrame) -> None:
    actual = splits["split"].value_counts().to_dict()
    if actual != SPLIT_SIZES:
        raise ValueError(f"Ukuran split {actual} != {SPLIT_SIZES}")


def check_class_proportions(splits: pd.DataFrame) -> None:
    per_split = pd.crosstab(splits["split"], splits["kelas"], normalize="index")
    overall = splits["kelas"].value_counts(normalize=True)
    print(per_split.round(3).to_string())
    worst = (per_split - overall).abs().to_numpy().max()
    if worst > PROPORTION_TOLERANCE:
        raise ValueError(f"Simpangan proporsi kelas {worst:.3f} > {PROPORTION_TOLERANCE}")


def check_frozen_hashes() -> None:
    protocol = yaml.safe_load(PROTOCOL_FILE.read_text(encoding="utf-8"))
    for name, expected in protocol["split_files_sha256"].items():
        if sha256_of(SPLITS_DIR / name) != expected:
            raise ValueError(f"{name} berubah sejak protokol dibekukan")


def main() -> None:
    splits = load_splits()
    check_no_leakage(splits)
    check_sizes(splits)
    check_class_proportions(splits)
    check_frozen_hashes()
    print("E0 OK: tanpa kebocoran, ukuran 1050/225/225, proporsi kelas seimbang, file utuh")


if __name__ == "__main__":
    main()
    