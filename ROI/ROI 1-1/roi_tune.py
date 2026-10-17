"""Grid search parameter detektor ROI Otsu (ROI 1-1) terhadap master_cleaned.json.

Pemakaian:  python "ROI/ROI 1-1/roi_tune.py"
"""
import sys

sys.dont_write_bytecode = True

import csv
import itertools
from dataclasses import asdict
from pathlib import Path

from roi_core import ROIParams, evaluate, load_ground_truth, load_images, load_split

GRID = {
    "color_space": ["lab", "hsv"],
    "blur_ksize": [5, 7],
    "close_ksize": [7, 11],
    "center_sigma": [0.28, 0.32],
    "merge_score_ratio": [0.55, 1.0],
}


def main():
    gt = load_ground_truth()
    split = load_split()
    val_imgs = load_images(split["val"])

    results = []
    keys = list(GRID)
    for vals in itertools.product(*GRID.values()):
        p = ROIParams(**dict(zip(keys, vals)))
        results.append((p, evaluate(p, val_imgs, gt)))

    results.sort(key=lambda r: (r[1]["mean_iou"], r[1]["hit_rate_iou50"]), reverse=True)

    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    with open(out / "tuning_results.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(list(asdict(results[0][0])) + list(results[0][1]))
        for p, s in results:
            w.writerow(list(asdict(p).values()) + [f"{v:.4f}" for v in s.values()])

    print("Top 5 (data val):")
    for p, s in results[:5]:
        print(f"  IoU={s['mean_iou']:.3f} hit@0.5={s['hit_rate_iou50']:.3f} cov={s['mean_coverage']:.3f} | {p}")

    best = results[0][0]
    print("\nParameter terbaik pada train/val/test:")
    for name in ("train", "val", "test"):
        s = evaluate(best, load_images(split[name]), gt)
        print(
            f"  {name:5s}: mean_iou={s['mean_iou']:.4f} median_iou={s['median_iou']:.4f} "
            f"hit@iou0.5={s['hit_rate_iou50']:.4f} cov={s['mean_coverage']:.4f} roi_frac={s['mean_roi_frac']:.4f}"
        )


if __name__ == "__main__":
    main()
