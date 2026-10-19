"""Preset 'teman' (versi awal ROI 1-0): mengikuti 4 parameter tuning dari teman
(color space, kernel blur, kernel closing, ukuran kontur minimum) dengan nilai baseline umum.

Pemakaian:
    python "ROI/ROI 1-0/roi_otsu.py"                       # evaluasi split test + output tiap tahap
    python "ROI/ROI 1-0/roi_otsu.py" --split val
    python "ROI/ROI 1-0/roi_otsu.py" --image path/foto.jpg # visualisasi satu citra bebas
"""
import sys

sys.dont_write_bytecode = True

from roi_core import ROIParams, run_preset

PARAMS = ROIParams(
    color_space="gray",        # 1. color space
    channel=0,
    blur_ksize=5,              # 2. kernel Gaussian blur
    close_ksize=5,             # 3. kernel morphological closing
    min_area_frac=0.002,       # 4. ukuran kontur minimum (0.2% luas citra, ~100 px pada 224x224)
    # sisanya default: polaritas otomatis, tanpa padding, gabungan semua kontur
)

if __name__ == "__main__":
    run_preset("otsu", PARAMS)
