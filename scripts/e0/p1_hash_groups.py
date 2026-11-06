"""E0 langkah 1-2: hitung citra per kelas, pHash, dan kelompok citra kembar."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .e0_common import (
    DATASET_DIR,
    HASH_GROUPS_FILE,
    IMAGE_SUFFIXES,
    MAX_HAMMING,
    close_pairs,
    compute_phashes,
    hashes_to_bits,
    index_images,
    read_metadata,
    write_csv,
)

KGL_IMAGE_DIR = DATASET_DIR / "raw" / "kgl"


def group_by_phash(hex_hashes: list[str]) -> np.ndarray:
    """Label kelompok per citra; kembar = komponen terhubung (Hamming <= MAX_HAMMING)."""
    bits = hashes_to_bits(hex_hashes)
    pairs = [(i, j) for i, j in close_pairs(bits, bits, MAX_HAMMING) if i < j]
    rows = np.array([i for i, _ in pairs], dtype=int)
    cols = np.array([j for _, j in pairs], dtype=int)
    size = len(hex_hashes)
    graph = csr_matrix((np.ones(len(pairs)), (rows, cols)), shape=(size, size))
    _, labels = connected_components(graph, directed=False)
    return labels


def build_group_table(metadata: pd.DataFrame) -> pd.DataFrame:
    paths = index_images(KGL_IMAGE_DIR)
    hashes = compute_phashes([paths[image_id] for image_id in metadata["image_id"]])
    labels = group_by_phash(hashes)
    return pd.DataFrame(
        {
            "image_id": metadata["image_id"],
            "hash_group": [f"G{label:05d}" for label in labels],
            "phash": hashes,
        }
    )


def report(metadata: pd.DataFrame, groups: pd.DataFrame) -> None:
    members = metadata.merge(groups, on="image_id").groupby("hash_group")
    sizes = members.size()
    mixed_class_groups = (members["class"].nunique() > 1).sum()
    print("Citra per kelas:\n" + metadata["class"].value_counts().to_string())
    print(f"Kelompok hash         : {len(sizes)}")
    print(
        f"Kelompok kembar (>1)  : {(sizes > 1).sum()} ({sizes[sizes > 1].sum()} citra)"
    )
    print(f"Kelompok terbesar     : {sizes.max()} citra")
    print(f"Kelompok lintas-kelas : {mixed_class_groups}")


def estimate_original_count(image_dir: Path) -> None:
    paths = sorted(
        p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    group_count = len(set(group_by_phash(compute_phashes(paths))))
    print(
        f"{image_dir}: {len(paths)} berkas, {group_count} kelompok pHash "
        "(batas atas jumlah citra asli; flip/rotasi tidak terdeteksi)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, help="folder dataset rencana awal")
    args = parser.parse_args()

    metadata = read_metadata("kgl")
    groups = build_group_table(metadata)
    write_csv(groups, HASH_GROUPS_FILE)
    report(metadata, groups)
    if args.legacy_dir:
        estimate_original_count(args.legacy_dir)


if __name__ == "__main__":
    main()
