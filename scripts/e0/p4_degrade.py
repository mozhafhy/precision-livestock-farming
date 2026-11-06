"""E0 langkah 6: himpunan degradasi sintetis dari split uji."""
import cv2
import numpy as np
import pandas as pd

from .e0_common import DATASET_DIR, SPLITS_DIR, index_images

MAX_SIDE = 100
DOWNSCALE_NAME = "INTER_AREA"
UPSCALE_NAME = "INTER_CUBIC"
OUTPUT_DIR = DATASET_DIR / "degraded" / "sintetis100"


def degrade(image: np.ndarray) -> np.ndarray:
    """Perkecil sisi terpanjang ke MAX_SIDE, lalu kembalikan ke ukuran asli."""
    height, width = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(height, width))
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = cv2.resize(image, small_size, interpolation=getattr(cv2, DOWNSCALE_NAME))
    return cv2.resize(small, (width, height), interpolation=getattr(cv2, UPSCALE_NAME))


def main() -> None:
    test_ids = pd.read_csv(SPLITS_DIR / "test.csv")["image_id"]
    paths = index_images(DATASET_DIR / "raw" / "kgl")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_id in test_ids:
        image = cv2.imread(str(paths[image_id]))
        if image is None:
            raise OSError(f"Gagal membaca {paths[image_id]}")
        cv2.imwrite(str(OUTPUT_DIR / f"{image_id}.png"), degrade(image))
    print(f"{len(test_ids)} citra -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()