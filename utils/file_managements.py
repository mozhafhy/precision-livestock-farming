import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
from cv2.typing import MatLike


def load_img(img_path: Path, flags: int = cv2.IMREAD_COLOR_RGB) -> MatLike:
    """Memuat gambar pada path yang diberikan dalam format RGB sebagai default"""
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
    

def load_json(filepath: Path):
    """Membaca berkas JSON dari path yang diberikan."""
    with open(filepath, "r") as f:
        return json.load(f)
    

def save_json(data: dict[str, Any], filepath: Path) -> None:
    """Menyimpan data ke dalam berkas JSON di path yang diberikan."""
    os.makedirs(filepath.parent, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def clean_roboflow_filename(filename):
    """Membersihkan suffix .rf.<UUID> dan tag format Roboflow.

    Contoh: 'KGL_0001_jpg.rf.1a2b3c4d5e6f7g.jpg' -> 'KGL_0001.jpg'
    """
    pattern = r"(_[a-zA-Z0-9]+)?\.rf\.[a-f0-9A-Za-z]+"
    return re.sub(pattern, "", filename)