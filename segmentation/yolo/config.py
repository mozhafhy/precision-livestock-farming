"""Konfigurasi terpusat: path proyek, konstanta, dan konvensi penamaan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VARIANTS = ("raw", "gray_world")
MODES = ("mask_based", "bbox_based")
SPLITS = ("train", "val", "test")

# Id COCO -> id YOLO. Id 0 pada export Roboflow adalah kategori induk
# ("Ground-Truth"), bukan objek, sehingga tidak dipetakan.
CLASS_MAP = {1: 0}
CLASS_NAMES = {0: "excreta"}


def check_choice(value: str, allowed: tuple, name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name} tidak valid: {value!r}. Pilihan: {allowed}")


def run_name(variant: str, mode: str) -> str:
    return f"{variant}_{mode}"


@dataclass(frozen=True)
class Config:
    root: Path = Path(".")
    yolo_path: Path = Path("segmentation/yolo")
    size: int = 512
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    # --- sumber data ---------------------------------------------------
    def annotation_path(self, split: str) -> Path:
        return self.root / "dataset" / "annotations" / f"{split}.json"

    def raw_image_path(self, file_name: str) -> Path:
        return self.root / "dataset" / "raw" / file_name

    def variant_image_path(self, variant: str, split: str, file_name: str) -> Path:
        check_choice(variant, VARIANTS, "variant")
        if variant == "raw":
            return self.raw_image_path(file_name)
        return (
            self.root
            / "dataset"
            / "rawGrayworld"
            / split
            / f"{Path(file_name).stem}.jpg"
        )

    # --- keluaran ------------------------------------------------------
    def dataset_dir(self, variant: str, mode: str) -> Path:
        return self.yolo_dir / "yolo_data" / run_name(variant, mode)

    def best_weights(self, variant: str, mode: str) -> Path:
        return self.runs_dir / run_name(variant, mode) / "weights" / "best.pt"

    @property
    def runs_dir(self) -> Path:
        return self.yolo_dir / "runs"

    @property
    def logs_dir(self) -> Path:
        return self.yolo_dir / "logs"
    
    @property
    def yolo_dir(self) -> Path:
        return self.root / self.yolo_path

    def relative(self, path: Path) -> str:
        """Path relatif terhadap root, agar manifest portabel antar mesin."""
        return Path(path).relative_to(self.root).as_posix()
