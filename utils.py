from math import ceil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .types import HistCfg


def get_gray_img(img_path: Path | str) -> cv2.typing.MatLike:
    img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak bisa dibaca: {img_path}")

    return img


_DEFAULT_HIST_CONFIG: HistCfg = {"show_hist": True, "log": False}


def show_img(
    img: cv2.typing.MatLike,
    title: str,
    axes: tuple[Axes, Axes] | Axes | None = None,
    hist_cfg: HistCfg | None = None,
    cmap: str = "gray",
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
    img_dict: dict[str, cv2.typing.MatLike],
    ncols: int = 3,
    hist_cfg: HistCfg | None = None,
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
