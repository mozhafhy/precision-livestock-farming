"""Konstanta, path, dan utilitas bersama untuk eksperimen E1 (perbandingan arsitektur)."""

from __future__ import annotations

import hashlib
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(".")
RAW_DIR = ROOT / "dataset/raw/kgl"
COCO_JSON = ROOT / "dataset/annotations/kgl/annotations.coco.json"
E0 = ROOT / "results/E0_protokol"
PROTOCOL_YAML = E0 / "config/protocol.yaml"
SPLITS_DIR = E0 / "splits"
E1 = ROOT / "results/E1_arsitektur"

MODELS = ("yolov8n-seg", "yolov8s-seg", "unet-mobilenetv2")
SPLITS = ("train", "val", "test")

# Konfigurasi C = normalisasi spasial (ROI crop + letterbox) + spektral (Gray World)
CONFIGS = {"C": {"spatial": True, "spectral": True}}

ROI_MARGIN = 0.10  # margin relatif di sekeliling union bbox saat auto-crop
PAD_VALUE = 114  # warna bingkai letterbox (konvensi YOLO)


def load_protocol() -> dict:
    # protocol.yaml berisi JSON, dan JSON adalah YAML yang valid
    with open(PROTOCOL_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def input_size(proto: dict) -> int:
    r = proto["input_resolution"]
    if isinstance(r, (list, tuple)):
        assert len(set(r)) == 1, "Resolusi masukan harus persegi"
        r = r[0]
    return int(r)


def read_split(name: str) -> list[str]:
    """Daftar nama file citra pada results/E0_protokol/splits/<name>.csv."""
    df = pd.read_csv(SPLITS_DIR / f"{name}.csv")
    for col in ("file_name", "filename", "image", "image_path", "path"):
        if col in df.columns:
            return df[col].astype(str).tolist()
    raise KeyError(f"{name}.csv: tidak ada kolom nama file (kolom: {list(df.columns)})")


def run_id(model: str, cfg: str, seed: int) -> str:
    return f"E1_{model}_cfg{cfg}_s{seed}"


def git_info() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], check=False, capture_output=True, text=True
            ).stdout.strip()
        except OSError:
            return ""

    return {
        "commit": run("rev-parse", "HEAD") or "unknown",
        "dirty": bool(run("status", "--porcelain")),
    }


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
