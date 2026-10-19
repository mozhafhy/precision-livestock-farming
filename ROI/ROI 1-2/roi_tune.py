"""Penyetelan parameter (Grid Search) Detektor ROI Otsu 5 Tahap.

Menyapu kombinasi parameter dari 5 tahap:
  1. Normalisasi warna   : none (off), gray_world, clahe
  2. Ruang warna & blur  : gray, lab (L/a/b), hsv (H/S/V) + ukuran kernel Gaussian blur
  3. Segmentasi Otsu     : invert (False / True)
  4. Morfologi           : ukuran kernel Opening & Closing
  5. Kontur & Padding    : luas kontur minimum (piksel) & nilai padding (piksel)

Seleksi parameter terbaik dilakukan pada subset validasi berdasarkan F1-score (IoU >= 0.5)
dan rata-rata IoU (selaras dengan scripts/calibrate_detector.py), lalu diuji sekali pada subset test.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import cv2

from roi_core import (
    ROIParams,
    _odd,
    evaluate,
    iou_xyxy,
    load_split_dataset,
    normalize_color,
    select_channel,
)

# Ruang pencarian parameter untuk ke-5 tahap
NORM_MODES = ["none", "gray_world", "clahe"]
CHANNELS = [
    ("gray", 0),
    ("lab", 0),
    ("lab", 1),
    ("lab", 2),
    ("hsv", 0),
    ("hsv", 1),
    ("hsv", 2),
]
BLUR_KSIZES = [3, 5, 7]
INVERT_MODES = [False, True]
OPEN_KSIZES = [0, 3, 5]
CLOSE_KSIZES = [3, 5, 9, 15]
MIN_AREAS = [50, 100, 250]
PADDINGS = [0, 10, 20, 30]


def _fast_sweep(dataset):
    """Mengevaluasi seluruh kombinasi grid secara hierarkis tanpa mengulang konversi citra yang sama."""
    n_img = len(dataset)
    results: list[tuple[ROIParams, dict[str, float]]] = []

    kernels = {
        k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k), _odd(k)))
        for k in set(OPEN_KSIZES + CLOSE_KSIZES)
        if k > 0
    }

    for norm_mode in NORM_MODES:
        normed = [(stem, normalize_color(img, norm_mode), gt, img.shape[:2]) for stem, img, gt in dataset]

        for color_space, channel in CHANNELS:
            ch_imgs = [(stem, select_channel(nimg, color_space, channel), gt, hw) for stem, nimg, gt, hw in normed]

            for blur_ksize in BLUR_KSIZES:
                kb = _odd(blur_ksize)
                blurred = [
                    (stem, cv2.GaussianBlur(c, (kb, kb), 0) if kb > 0 else c, gt, hw)
                    for stem, c, gt, hw in ch_imgs
                ]

                for invert in INVERT_MODES:
                    flag = (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
                    otsu_imgs = [(stem, cv2.threshold(b, 0, 255, flag)[1], gt, hw) for stem, b, gt, hw in blurred]

                    for open_ksize in OPEN_KSIZES:
                        if open_ksize > 0:
                            ko = kernels[open_ksize]
                            opened_imgs = [
                                (stem, cv2.morphologyEx(o, cv2.MORPH_OPEN, ko), gt, hw)
                                for stem, o, gt, hw in otsu_imgs
                            ]
                        else:
                            opened_imgs = otsu_imgs

                        for close_ksize in CLOSE_KSIZES:
                            if close_ksize > 0:
                                kc = kernels[close_ksize]
                                closed_imgs = [
                                    (stem, cv2.morphologyEx(op, cv2.MORPH_CLOSE, kc), gt, hw)
                                    for stem, op, gt, hw in opened_imgs
                                ]
                            else:
                                closed_imgs = opened_imgs

                            # Ekstrak daftar (luas, x, y, bw, bh) per citra satu kali untuk pasangan (min_area, padding)
                            contour_table = []
                            for _, cl, gt, (h, w) in closed_imgs:
                                cnts, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cinfo = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in cnts]
                                contour_table.append((cinfo, gt, w, h))

                            for min_area, padding in itertools.product(MIN_AREAS, PADDINGS):
                                tp = fp = fn = 0
                                tp_iou_sum = 0.0
                                all_iou_sum = 0.0
                                count_ok = 0

                                for cinfo, gt, w, h in contour_table:
                                    valid = [item for item in cinfo if item[0] >= min_area]
                                    if valid:
                                        _, (x, y, bw, bh) = max(valid, key=lambda it: it[0])
                                        b = (
                                            float(max(0, x - padding)),
                                            float(max(0, y - padding)),
                                            float(min(w, x + bw + padding)),
                                            float(min(h, y + bh + padding)),
                                        )
                                        best_iou = 0.0
                                        best_gi = -1
                                        for gi, g in enumerate(gt):
                                            v = iou_xyxy(b, g)
                                            if v > best_iou:
                                                best_iou, best_gi = v, gi
                                        if best_gi >= 0 and best_iou >= 0.5:
                                            tp += 1
                                            tp_iou_sum += best_iou
                                        else:
                                            fp += 1
                                        fn += len(gt) - (1 if (best_gi >= 0 and best_iou >= 0.5) else 0)
                                        all_iou_sum += best_iou
                                        count_ok += int(1 == len(gt))
                                    else:
                                        fn += len(gt)
                                        count_ok += int(0 == len(gt))

                                prec = tp / (tp + fp) if (tp + fp) else 0.0
                                rec = tp / (tp + fn) if (tp + fn) else 0.0
                                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                                p = ROIParams(
                                    norm_mode=norm_mode,
                                    color_space=color_space,
                                    channel=channel,
                                    blur_ksize=blur_ksize,
                                    invert=invert,
                                    open_ksize=open_ksize,
                                    close_ksize=close_ksize,
                                    min_area=min_area,
                                    padding=padding,
                                )
                                results.append((
                                    p,
                                    {
                                        "f1": f1,
                                        "precision": prec,
                                        "recall": rec,
                                        "mean_iou": (tp_iou_sum / tp) if tp else 0.0,
                                        "mean_iou_all": all_iou_sum / n_img,
                                        "count_acc": count_ok / n_img,
                                    },
                                ))
    return results


def main() -> None:
    val_data = load_split_dataset("val")
    test_data = load_split_dataset("test")

    results = _fast_sweep(val_data)
    # Urutkan berdasarkan F1 tertinggi, lalu mean_iou_all tertinggi, lalu mean_iou (TP) tertinggi
    results.sort(key=lambda r: (r[1]["f1"], r[1]["mean_iou_all"], r[1]["mean_iou"]), reverse=True)

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "tuning_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        p_keys = list(asdict(results[0][0]).keys())
        m_keys = list(results[0][1].keys())
        w.writerow(p_keys + m_keys)
        for p, m in results:
            w.writerow(list(asdict(p).values()) + [f"{v:.4f}" for v in m.values()])

    best_params = results[0][0]
    val_metrics = evaluate(best_params, val_data)
    test_metrics = evaluate(best_params, test_data)

    # Simpan parameter terpilih dan hasil evaluasi dengan struktur selaras calibrate_detector.py
    (out_dir / "thresholds.json").write_text(
        json.dumps(
            {
                "params": asdict(best_params),
                "f1_val": val_metrics["f1"],
                "val": val_metrics,
                "test": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Top 5 Konfigurasi pada Subset Validasi:")
    for p, m in results[:5]:
        print(
            f"  F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"IoU(TP)={m['mean_iou']:.3f} IoU(all)={m['mean_iou_all']:.3f} | {p}"
        )
    print("\nEvaluasi Parameter Terbaik pada Subset Val & Test:")
    print(json.dumps({"best_params": asdict(best_params), "val": val_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
