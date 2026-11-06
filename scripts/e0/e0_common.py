"""Konstanta dan fungsi bersama untuk skrip E0."""

import hashlib
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

E0_RESULTS_DIR = Path("results/E0_protokol")
DATASET_DIR = Path("dataset")

SPLITS_DIR = E0_RESULTS_DIR / "splits"

CONFIG_DIR = E0_RESULTS_DIR / "config"
PROTOCOL_FILE = CONFIG_DIR / "protocol.yaml"

HASH_GROUPS_FILE = SPLITS_DIR / "hash_groups.csv"
FROZEN_SPLIT_FILES = [
    "train.csv",
    "val.csv",
    "test.csv",
    "external_mendeley.csv",
    "hash_groups.csv",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_SEED = 42
MAX_HAMMING = 5
SUBSET_SIZE = 1500
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SIZES = {name: round(SUBSET_SIZE * ratio) for name, ratio in SPLIT_RATIOS.items()}
EXTERNAL_SIZE = 100
CHUNK_ROWS = 1024


def read_metadata(source: str) -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / f"{source}_metadata.csv")


def read_hash_groups() -> pd.DataFrame:
    return pd.read_csv(HASH_GROUPS_FILE)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """LF eksplisit agar SHA-256 sama di semua OS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_images(image_dir: Path) -> dict[str, Path]:
    """image_id (nama tanpa ekstensi) -> path."""
    paths = sorted(image_dir.rglob("*"))
    return {path.stem: path for path in paths if path.suffix.lower() in IMAGE_SUFFIXES}


def compute_phash(path: Path) -> str:
    with Image.open(path) as image:
        return str(imagehash.phash(image))


def compute_phashes(paths: list[Path]) -> list[str]:
    return [compute_phash(path) for path in tqdm(paths, desc="pHash")]


def hashes_to_bits(hex_hashes: list[str]) -> np.ndarray:
    raw = np.frombuffer(bytes.fromhex("".join(hex_hashes)), dtype=np.uint8)
    return np.unpackbits(raw.reshape(len(hex_hashes), -1), axis=1).astype(np.float32)


def close_pairs(bits_a: np.ndarray, bits_b: np.ndarray, max_distance: int):
    """Hasilkan pasangan indeks (i, j) dengan jarak Hamming <= max_distance."""
    ones_b = bits_b.sum(axis=1)
    for start in range(0, len(bits_a), CHUNK_ROWS):
        block = bits_a[start : start + CHUNK_ROWS]
        distances = (
            block.sum(axis=1)[:, None] + ones_b[None, :] - 2 * (block @ bits_b.T)
        )
        rows, cols = np.nonzero(distances <= max_distance)
        yield from zip((rows + start).tolist(), cols.tolist())


def allocate_quotas(class_counts: dict[Any, int], total: int) -> dict[Any, int]:
    """Jatah per kelas proporsional (largest remainder), jumlahnya tepat `total`."""
    population = sum(class_counts.values())

    # Perbaikan: Tangani kasus populasi 0 untuk menghindari ZeroDivisionError
    if population == 0 or total > population:
        return {label: 0 for label in class_counts}

    exact = {label: total * count / population for label, count in class_counts.items()}
    quotas = {label: int(value) for label, value in exact.items()}

    by_remainder = sorted(
        exact, key=lambda label: exact[label] - quotas[label], reverse=True
    )

    for label in by_remainder[: total - sum(quotas.values())]:
        quotas[label] += 1

    return quotas


def sample_stratified(frame: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    # Early exit jika dataframe kosong atau total tidak valid
    if frame.empty or total <= 0:
        return frame.iloc[:0]

    # KISS: Buat Series sementara dimana null diubah menjadi teks biasa.
    # Ini menjamin null terikut dalam groupby dan aman dari KeyError pada dictionary.
    stratify_col = frame["class"].fillna("__MISSING__")

    # Hitung kuota menggunakan Series sementara
    quotas = allocate_quotas(stratify_col.value_counts().to_dict(), total)

    # Lakukan sampling
    parts = [
        members.sample(n=quotas[label], random_state=seed)
        for label, members in frame.groupby(stratify_col)
        if quotas.get(label, 0) > 0  # Pastikan tidak sampling 0 data
    ]

    # Tangani kasus jika semua kuota adalah 0 (misal: total > population)
    if not parts:
        return frame.iloc[:0]

    return pd.concat(parts).sort_values("image_id").reset_index(drop=True)
