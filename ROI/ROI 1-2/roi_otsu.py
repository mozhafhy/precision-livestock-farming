"""Implementasi Detektor ROI Ekskreta berbasis Threshold Otsu (5 Tahap Tunable).

Parameter di bawah merupakan hasil tuning terbaik pada subset validasi (lihat ROI/output/thresholds.json)
dan dapat diubah langsung di variabel PARAMS untuk eksperimen manual.

Pemakaian:
    python ROI/roi_otsu.py                             # evaluasi subset test + simpan gambar tiap tahap
    python ROI/roi_otsu.py --split val                 # evaluasi subset validasi
    python ROI/roi_otsu.py --image dataset/raw/KGL_0020.jpg
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from roi_core import ROIParams, run_detector

# Parameter tunable untuk ke-5 tahap deteksi ROI Otsu
PARAMS = ROIParams(
    # Tahap 1: Normalisasi warna ("none" [off], "gray_world", atau "clahe")
    norm_mode="gray_world",
    # Tahap 2: Konversi ruang warna ("gray", "lab", "hsv") & ukuran kernel Gaussian blur
    color_space="hsv",
    channel=1,              # kanal 1 pada HSV = Saturation (S)
    blur_ksize=7,           # kernel Gaussian blur 7x7
    # Tahap 3: Segmentasi kasar dengan Otsu (invert False / True)
    invert=False,
    # Tahap 4: Operasi morfologi berurutan (Opening lalu Closing)
    open_ksize=3,           # kernel Opening 3x3 (hapus bintik kecil di luar objek)
    close_ksize=9,          # kernel Closing 9x9 (tambal lubang di dalam ekskreta)
    # Tahap 5: Pencarian kontur terbesar, luas minimum, dan penambahan ruang (padding)
    min_area=100,           # luas kontur minimum dalam piksel
    padding=0,              # margin tambahan dalam piksel di keempat sisi kotak (misal 0, 10, 20)
)

if __name__ == "__main__":
    run_detector(PARAMS)
