import re
import shutil
from pathlib import Path

import cv2
from cv2.typing import MatLike


def get_img(img_path: Path | str, flags: int = cv2.IMREAD_COLOR_RGB) -> MatLike:
    img = cv2.imread(img_path, flags=flags)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak bisa dibaca: {img_path}")

    return img


def get_file_number(file_path: Path) -> int:
    match = re.search(r"\d+", file_path.name)
    return int(match.group()) if match else 0


def reset_author_dir(author_dir: Path) -> None:
    """Hapus total isi lama sebelum copy — menjamin idempotensi run."""
    if author_dir.exists():
        shutil.rmtree(author_dir)
    author_dir.mkdir(parents=True)