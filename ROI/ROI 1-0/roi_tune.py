"""Grid search parameter detektor ROI Otsu (versi awal ROI 1-0).

Tuning dilakukan HANYA pada data train; hasil terbaik lalu dilaporkan pada val dan test
supaya skor akhir tidak bias (prinsip pembagian data pada Tahap 1 modul).

Pemakaian:  python "ROI/ROI 1-0/roi_tune.py"
"""
import sys

sys.dont_write_bytecode = True

import csv
import itertools
from dataclasses import asdict
from pathlib import Path

from roi_core import ROIParams, evaluate, load_ground_truth, load_images, load_split

MIN_COVERAGE = 0.85    # syarat: ROI harus memuat rata-rata >= 85% objek, baru dipilih berdasarkan IoU

# ruang pencarian (kombinasi color space + kanal, lalu parameter lain)
CHANNELS = [("gray", 0), ("lab", 0), ("lab", 1), ("lab", 2), ("hsv", 0), ("hsv", 1), ("hsv", 2)]
GRID = {
    "blur_ksize": [3, 5, 9, 15],
    "close_ksize": [3, 7, 15, 25],
    "min_area_frac": [0.002, 0.01, 0.03],
    "polarity": ["auto", "dark", "bright"],
    "pad_frac": [0.0, 0.1],
    "roi_mode": ["union", "largest"],
}


def main():
    gt = load_ground_truth()
    split = load_split()
    train = load_images(split["train"])

    results = []
    keys = list(GRID)
    for (cs, ch), vals in itertools.product(CHANNELS, itertools.product(*GRID.values())):
        p = ROIParams(color_space=cs, channel=ch, **dict(zip(keys, vals)))
        results.append((p, evaluate(p, train, gt)))

    # urutkan: yang memenuhi syarat coverage dulu, lalu IoU rata-rata tertinggi
    results.sort(key=lambda r: (r[1]["mean_coverage"] >= MIN_COVERAGE, r[1]["mean_iou"]), reverse=True)

    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    with open(out / "tuning_results.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(list(asdict(results[0][0])) + list(results[0][1]))
        for p, s in results:
            w.writerow(list(asdict(p).values()) + [f"{v:.4f}" for v in s.values()])

    print("Top 10 (data train):")
    for p, s in results[:10]:
        print(f"  IoU={s['mean_iou']:.3f} cov={s['mean_coverage']:.3f} roi={s['mean_roi_frac']:.3f} | {p}")

    best = results[0][0]
    print("\nParameter terbaik pada val/test:")
    for name in ("val", "test"):
        s = evaluate(best, load_images(split[name]), gt)
        print(f"  {name}: IoU={s['mean_iou']:.3f} cov={s['mean_coverage']:.3f} "
              f"hit@cov0.9={s['hit_rate_cov90']:.3f} roi={s['mean_roi_frac']:.3f}")


if __name__ == "__main__":
    main()
