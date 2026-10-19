"""Detektor ROI Otsu (ROI 1-1) yang ditingkatkan agar hasil mendekati master_cleaned.json.

Selain 4 parameter dasar (color space, Gaussian blur, morphological closing, ukuran kontur minimum),
ditambahkan 3 metode pengolahan citra di sekitar Otsu:
  1. Background Color Contrast & Difference of Gaussians (DoG) di ruang LAB
  2. Center-Surround Gaussian Spatial Prior (meredam derau/bayangan tepi frame)
  3. Penilaian skor kontur (luas + kedekatan ke pusat + penalti kontur yang menempel di tepi frame)

Pemakaian:
    python "ROI/ROI 1-1/roi_otsu.py"                       # evaluasi split test + simpan gambar tiap tahap
    python "ROI/ROI 1-1/roi_otsu.py" --split val
    python "ROI/ROI 1-1/roi_otsu.py" --image path/foto.jpg
"""
import sys

sys.dont_write_bytecode = True

from roi_core import ROIParams, run_preset

PARAMS = ROIParams(
    color_space="lab",          # 1. color space ("lab", "hsv", "gray")
    channel=0,
    blur_ksize=7,               # 2. kernel Gaussian blur
    open_ksize=3,               # kernel morphological opening (hapus bintik sekam kecil)
    close_ksize=7,              # 3. kernel morphological closing (satukan bagian urat & feses)
    min_area_frac=0.002,        # 4. ukuran kontur minimum (0.2% luas citra)
    use_bg_contrast=True,       # metode tambahan 1: selisih warna terhadap latar tepi & lokal (DoG)
    border_margin=0.12,         # lebar tepi referensi warna latar (12% dari sisi citra)
    center_sigma=0.28,          # metode tambahan 2: prior spasial Gaussian terpusat
    merge_score_ratio=0.55,     # metode tambahan 3: gabungkan kontur inti dengan skor >= 55% skor tertinggi
    pad_frac=0.04,              # margin tambahan di sekeliling kotak ROI (4% ukuran kotak)
)

if __name__ == "__main__":
    run_preset("otsu", PARAMS)
