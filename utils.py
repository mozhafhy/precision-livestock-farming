import re
import shutil
from math import ceil
from pathlib import Path
from typing import TypedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from matplotlib.axes import Axes


class HistCfg(TypedDict, total=False):
    show_hist: bool
    log: bool


def get_img(img_path: Path | str) -> MatLike:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak bisa dibaca: {img_path}")

    return img


_DEFAULT_HIST_CONFIG: HistCfg = {"show_hist": True, "log": False}


def show_img(
    img: MatLike,
    title: str,
    axes: tuple[Axes, Axes] | Axes | None = None,
    hist_cfg: HistCfg | None = None,
    cmap: str | None = "gray",
    figsize: tuple[int, int] = (6, 6),
) -> None:

    hist_cfg = _DEFAULT_HIST_CONFIG if hist_cfg is None else hist_cfg

    show_hist = hist_cfg.get("show_hist", True)
    standalone = axes is None

    img_ax: Axes
    hist_ax: Axes | None = None

    # Inisialisasi axes berdasarkan mode standalone dan konfigurasi histogram
    if standalone:
        if show_hist:
            _, (img_ax, hist_ax) = plt.subplots(2, 1, figsize=figsize)
        else:
            _, img_ax = plt.subplots(1, 1, figsize=figsize)
    else:
        if isinstance(axes, tuple):
            img_ax, hist_ax = axes
        else:
            img_ax = axes

    # Tampilkan gambar
    img_ax.imshow(img, cmap=cmap)
    img_ax.set_title(title)

    # Tampilkan histogram jika show_hist bernilai True
    if show_hist and hist_ax is not None:
        log = hist_cfg.get("log", False)

        hist_ax.hist(img.ravel(), bins=256, range=(0, 256), color="r", log=log)
        hist_ax.set_xlim(-10, 265)
        hist_ax.set_xticks(np.arange(0, 256, 25))
        hist_ax.tick_params(labelsize=7)
        hist_ax.set_title(
            f"Histogram - {title}" if not log else f"Histogram (skala log) - {title}"
        )

    if standalone:
        plt.tight_layout()

        if show_hist and hist_ax is not None:
            img_pos = img_ax.get_position()
            hist_pos = hist_ax.get_position()
            hist_ax.set_position(
                (img_pos.x0, hist_pos.y0, img_pos.width, hist_pos.height)
            )

        plt.show()


def show_images(
    img_dict: dict[str, MatLike],
    ncols: int = 3,
    hist_cfg: HistCfg | None = None,
    cmap: str | None = "gray",
) -> None:

    hist_cfg = _DEFAULT_HIST_CONFIG if hist_cfg is None else hist_cfg

    show_hist = hist_cfg.get("show_hist", True)

    N_IMAGES = len(img_dict)
    num_image_rows = ceil(N_IMAGES / ncols)

    # Tentukan jumlah baris grid: 2 baris per gambar jika histogram aktif, 1 baris jika tidak
    nrows = (2 * num_image_rows) if show_hist else num_image_rows

    fig_width = 5 * ncols
    fig_height = 4 * nrows

    _, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    for i, (title, img) in enumerate(img_dict.items()):
        if show_hist:
            row = (i // ncols) * 2
            col = i % ncols
            axes = (axs[row, col], axs[row + 1, col])
        else:
            row = i // ncols
            col = i % ncols
            axes = axs[row, col]

        show_img(
            img=img,
            axes=axes,
            hist_cfg=hist_cfg,
            title=title,
            cmap=cmap,
        )

    # Sembunyikan subplot yang tidak terpakai
    for j in range(N_IMAGES, num_image_rows * ncols):
        if show_hist:
            row = (j // ncols) * 2
            col = j % ncols
            axs[row, col].set_visible(False)
            axs[row + 1, col].set_visible(False)
        else:
            row = j // ncols
            col = j % ncols
            axs[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()


def get_file_number(file_path: Path) -> int:
    match = re.search(r"\d+", file_path.name)
    return int(match.group()) if match else 0


def reset_author_dir(author_dir: Path) -> None:
    """Hapus total isi lama sebelum copy — menjamin idempotensi run."""
    if author_dir.exists():
        shutil.rmtree(author_dir)
    author_dir.mkdir(parents=True)


def get_optimal_kernel_size(
    img: MatLike,
    percentage: float = 0.01,
) -> tuple[int, int]:
    h, w = img.shape[:2]
    min_dim = min(h, w)

    k = int(min_dim * percentage)

    k += (k + 1) % 2  # memastikan k selalu ganjil
    k = max(3, k)

    return (k, k)


def auto_crop(
    img: MatLike,
    padding: int = 15,
    min_area_ratio: float = 0.005,
) -> MatLike:
    img_h, img_w = img.shape[:2]
    total_area = img_h * img_w

    # 1. Konversi ke HSV dan ambil kanal Saturation (S)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # 2. Gaussian Blur pada kanal Saturation
    k_size = get_optimal_kernel_size(img)
    blurred = cv2.GaussianBlur(saturation, k_size, 0)

    # 3. Otsu Thresholding pada kanal Saturation
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Morphological Closing untuk menyatukan area kotoran yang terpisah
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)

    # 5. Cari kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    # 6. FILTERING: HANYA ambil kontur yang luasnya > min_area_ratio dari total gambar
    valid_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > (total_area * min_area_ratio)
    ]

    # Jika tidak ada kontur yang memenuhi syarat, ambil 1 kontur terbesar
    if not valid_contours:
        largest_contour = max(contours, key=cv2.contourArea)
        valid_contours = [largest_contour]

    # 7. Gabungkan HANYA kontur-kontur valid yang sudah difilter
    combined_contours = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(combined_contours)

    # 8. Terapkan padding aman
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    return img[y1:y2, x1:x2]


def apply_gray_world(img: MatLike) -> MatLike:
    img_float = img.astype(np.float32)

    channel_means = np.mean(img_float, axis=(0, 1))
    avg_gray = np.mean(channel_means)
    scale_factors = avg_gray / (channel_means + 1e-8)

    img_float = np.clip(img_float * scale_factors, 0, 255)

    return img_float.astype(img.dtype)
