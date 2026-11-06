"""E0 langkah 3-4: subset terstratifikasi 1.500 citra, split 70:15:15 per kelompok hash."""

import random
from collections import Counter
from typing import Any

import pandas as pd

from .e0_common import (
    SPLIT_RATIOS,
    SPLIT_SEED,
    SPLIT_SIZES,
    SPLITS_DIR,
    SUBSET_SIZE,
    read_hash_groups,
    read_metadata,
    sample_stratified,
    write_csv,
)


def class_targets(subset: pd.DataFrame) -> dict[str, dict[Any, float]]:
    """Jumlah ideal per kelas di tiap split."""
    totals = subset["class"].value_counts()
    return {
        split: {label: ratio * count for label, count in totals.items()}
        for split, ratio in SPLIT_RATIOS.items()
    }


def imbalance_increase(counts: dict, assigned: Counter, target: dict) -> float:
    """Kenaikan simpangan kuadrat terbobot bila kelompok ini masuk ke split."""
    return sum(
        n * (n + 2 * (assigned[label] - target[label])) / target[label]
        for label, n in counts.items()
    )


def assign_splits(subset: pd.DataFrame) -> dict[str, str]:
    """Greedy: kelompok terbesar dulu, ke split yang paling kekurangan kelas itu."""
    targets = class_targets(subset)
    capacity = dict(SPLIT_SIZES)
    assigned = {split: Counter() for split in SPLIT_SIZES}
    groups = list(subset.groupby("hash_group"))
    random.Random(SPLIT_SEED).shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    split_of: dict[str, str] = {}
    for group_id, members in groups:
        counts = members["class"].value_counts().to_dict()
        fitting = [split for split in capacity if capacity[split] >= len(members)]
        if not fitting:
            raise ValueError(f"Kelompok {group_id} ({len(members)} citra) tidak muat")
        chosen = min(
            fitting, key=lambda s: imbalance_increase(counts, assigned[s], targets[s])
        )
        capacity[chosen] -= len(members)
        assigned[chosen].update(counts)
        split_of.update({image_id: chosen for image_id in members["image_id"]})
    return split_of


def write_splits(subset: pd.DataFrame) -> None:
    for split in SPLIT_SIZES:
        part = subset[subset["split"] == split]
        columns = part[["image_id", "class", "hash_group"]].rename(
            columns={"class": "kelas"}
        )
        write_csv(columns, SPLITS_DIR / f"{split}.csv")


def e0a_table(pool: pd.DataFrame, subset: pd.DataFrame) -> pd.DataFrame:
    """Isian E0-a: tersedia, kelompok kembar, latih/validasi/uji per kelas."""
    group_sizes = pool.groupby("hash_group")["image_id"].transform("size")
    twin_groups = pool[group_sizes > 1].groupby("class")["hash_group"].nunique()
    table = pd.DataFrame(
        {"tersedia": pool["class"].value_counts(), "kelompok_kembar": twin_groups}
    )
    table = (
        table.join(pd.crosstab(subset["class"], subset["split"])).fillna(0).astype(int)
    )
    table = table[["tersedia", "kelompok_kembar", *SPLIT_SIZES]]
    table.loc["Total"] = table.sum()
    return table


def refuse_overwrite() -> None:
    frozen = [SPLITS_DIR / f"{split}.csv" for split in SPLIT_SIZES]
    if any(path.exists() for path in frozen):
        raise FileExistsError(
            "Split sudah ada (dibekukan); hapus manual bila sengaja diulang"
        )


def main() -> None:
    refuse_overwrite()
    pool = read_metadata("kgl").merge(read_hash_groups(), on="image_id")
    subset = sample_stratified(pool, SUBSET_SIZE, SPLIT_SEED)
    subset["split"] = subset["image_id"].map(assign_splits(subset))
    write_splits(subset)
    print(e0a_table(pool, subset).to_string())


if __name__ == "__main__":
    main()
