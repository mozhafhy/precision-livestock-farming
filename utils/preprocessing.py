from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from .file_managements import get_img
from .types import AutoCropResult, BoundingBox
from .visualization import draw_detection_overlay


def find_blurring_kernel_size(
    img: MatLike,
    percentage: float = 0.01,
) -> tuple[int, int]:
    h, w = img.shape[:2]
    min_dim = min(h, w)

    k = int(min_dim * percentage)

    k += (k + 1) % 2  # memastikan k selalu ganjil
    k = max(3, k)

    return (k, k)


# HELPER FUNCTIONS
def _get_crop_from_contours(
    img: MatLike,
    contours: Sequence[MatLike],
    padding: int = 15,
    min_area_ratio: float = 0.005
) -> tuple[Sequence[MatLike], BoundingBox, MatLike]:
    img_h, img_w = img.shape[:2]
    total_area = img_h * img_w

    min_area = total_area * min_area_ratio

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Fallback ke kontur terbesar
    if not valid_contours:
        largest_contour = max(contours, key=cv2.contourArea)
        valid_contours = [largest_contour]

    # 7. Bounding box dari kontur valid
    combined_contours = np.vstack(valid_contours)

    x, y, w, h = cv2.boundingRect(combined_contours)

    # 8. Padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    bbox = BoundingBox(x1, y1, x2, y2)

    cropped = img[y1:y2, x1:x2]
    
    return valid_contours, bbox, cropped
    

###################


# AUTO CROP 1
def auto_crop(
    img: MatLike,
    padding: int = 15,
    min_area_ratio: float = 0.005,
    morph_k_size: tuple[int, int] = (11, 11)
) -> AutoCropResult:
    # 1. Konversi ke HSV dan ambil kanal Saturation (S)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    # 2. Gaussian Blur
    k_size = find_blurring_kernel_size(img)
    blurred = cv2.GaussianBlur(saturation, k_size, 0)

    # 3. Otsu Thresholding
    thresh_value, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Morphological Closing
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_k_size)

    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)

    # 5. Cari kontur
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Nilai default jika tidak ada kontur yang valid
    valid_contours: Sequence[MatLike] = []
    bbox = None
    cropped = img
    
    if contours:
        valid_contours, bbox, cropped = _get_crop_from_contours(
            img=img,
            contours=contours,
            padding=padding,
            min_area_ratio=min_area_ratio,
        )
        
    overlay = draw_detection_overlay(img, valid_contours, bbox)

    return AutoCropResult(
        images={
            "Original": img,
            "Saturation": saturation,
            "Gaussian Blur": blurred,
            f"Otsu Threshold ({thresh_value = })": thresh,
            "Morphological Closing": closed,
            "Detection Overlay": overlay,
            "Final Crop": cropped,
        },
        contours=contours,
        valid_contours=valid_contours,
        bbox=bbox,
    )


def apply_gray_world(img: MatLike) -> MatLike:
    img_float = img.astype(np.float32)

    channel_means = np.mean(img_float, axis=(0, 1))
    avg_gray = np.mean(channel_means)
    scale_factors = avg_gray / (channel_means + 1e-8)

    img_float = np.clip(img_float * scale_factors, 0, 255)

    return img_float.astype(img.dtype)


def process_single_img(
    inpath: Path, outpath: Path, return_result: bool = False
) -> MatLike | None:
    img = get_img(inpath)

    cropped = auto_crop(img)["images"]["Final Crop"]
    processed = apply_gray_world(cropped)

    cv2.imwrite(outpath, processed)
    print(f"Berhasil memproses: {inpath.name} -> {outpath}")

    if return_result:
        return get_img(outpath)

    return None
