"""E0 langkah 7: bekukan protokol ke config/protocol.yaml."""

import importlib
import sys
from datetime import datetime

import yaml
from common import (
    CONFIG_DIR,
    EXTERNAL_SIZE,
    FROZEN_SPLIT_FILES,
    MAX_HAMMING,
    PROTOCOL_FILE,
    SPLIT_SEED,
    SPLIT_SIZES,
    SPLITS_DIR,
    SUBSET_SIZE,
    sha256_of,
)
from p4_degrade import DOWNSCALE_NAME, MAX_SIDE, UPSCALE_NAME

# USULAN: ubah sesuai keputusanmu SEBELUM dijalankan; nilai ini lalu dibekukan.
TRAINING_SEEDS = [0, 1, 2]
INPUT_RESOLUTION = 256
MAX_EPOCHS = 100
PATIENCE = 20
OPTIMIZER = "AdamW"
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
AUGMENTATION = {
    "horizontal_flip_p": 0.5,
    "rotation_degrees": 15,
    "brightness_contrast_limit": 0.2,
}
UNET_THRESHOLD = 0.5
YOLO_CONF_GRID = [round(0.10 + 0.05 * step, 2) for step in range(17)]  # 0,10 .. 0,90
TRACKED_LIBRARIES = [
    "numpy",
    "python",
    "opencv",
    "ipykernel",
    "matplotlib",
    "scikit-learn",
    "pandas",
    "pyyaml",
    "pycocotools",
    "ultralytics",
    "torchvision",
    "segmentation-models-pytorch",
    "albumentations",
    "imagehash",
]


def library_version(module_name: str) -> str:
    try:
        return importlib.import_module(module_name).__version__
    except (ImportError, AttributeError):
        return "tidak terdeteksi"


def gpu_name() -> str:
    try:
        import torch
    except ImportError:
        return "torch tidak terpasang"
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def build_protocol() -> dict:
    return {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": TRAINING_SEEDS,
        "data": {
            "split_seed": SPLIT_SEED,
            "subset_size": SUBSET_SIZE,
            "split_sizes": SPLIT_SIZES,
            "phash_max_hamming": MAX_HAMMING,
            "external_size": EXTERNAL_SIZE,
            "degradation": {
                "max_side": MAX_SIDE,
                "downscale": DOWNSCALE_NAME,
                "upscale": UPSCALE_NAME,
            },
        },
        "input_resolution": INPUT_RESOLUTION,
        "augmentation": AUGMENTATION,
        "training": {
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "optimizer": OPTIMIZER,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
        },
        "inference": {
            "unet_threshold": UNET_THRESHOLD,
            "yolo_conf_grid": YOLO_CONF_GRID,
        },
        "models": {
            "yolov8n-seg": {"pretrained": "COCO", "loss": "bawaan YOLOv8-Seg"},
            "yolov8s-seg": {"pretrained": "COCO", "loss": "bawaan YOLOv8-Seg"},
            "unet-mobilenetv2": {
                "pretrained": "ImageNet (encoder)",
                "loss": "BCE + Dice",
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "gpu": gpu_name(),
            "libraries": {name: library_version(name) for name in TRACKED_LIBRARIES},
        },
        "split_files_sha256": {
            name: sha256_of(SPLITS_DIR / name) for name in FROZEN_SPLIT_FILES
        },
    }


def main() -> None:
    if PROTOCOL_FILE.exists():
        raise FileExistsError(
            f"{PROTOCOL_FILE} sudah ada (dibekukan); hapus manual bila sengaja"
        )
    CONFIG_DIR.mkdir(exist_ok=True)
    text = yaml.safe_dump(build_protocol(), sort_keys=False, allow_unicode=True)
    PROTOCOL_FILE.write_text(text, encoding="utf-8")
    print(f"Protokol ditulis ke {PROTOCOL_FILE}")


if __name__ == "__main__":
    main()
