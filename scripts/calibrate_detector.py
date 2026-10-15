"""Tahap 3: kalibrasi detektor ROI pada subset VALIDASI, lalu evaluasi subset uji dengan ambang beku.

    # 1. kalibrasi (hanya val): τconf pada F1 maksimum, ambang IoU NMS dari sapuan
    python -m scripts.calibrate_detector --weights dataset/detector/runs/train/weights/best.pt

    # 2. evaluasi uji, SEKALI, memakai thresholds.json hasil langkah 1
    python -m scripts.calibrate_detector --weights dataset/detector/runs/train/weights/best.pt --evaluate-test

Metrik (sebagian dari Tabel 4 panduan): precision, recall, F1, rata-rata IoU kotak (pasangan TP, IoU >= 0.5),
count accuracy, tingkat no_detection, mAP@0.5 (hanya uji). Crop success rate butuh Tahap 4 dan belum ada di sini.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from ultralytics.engine.results import Results

if TYPE_CHECKING:
    from ultralytics import YOLO

IMG_EXT: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
DEFAULT_NMS_IOU: float = 0.7

Box = tuple[float, float, float, float]  # xyxy, piksel


class Item(TypedDict):
    """Prediksi dan ground truth satu citra; pred/conf sudah urut menurun menurut confidence."""

    name: str
    pred: NDArray[np.float64]  # (N, 4) xyxy
    conf: NDArray[np.float64]  # (N,)
    gt: list[Box]


class Metrics(TypedDict):
    precision: float
    recall: float
    f1: float
    mean_iou: float
    count_acc: float
    crowded_n: int
    crowded_count_acc: float | None
    no_detection_all: float
    no_detection_on_positives: float | None  # citra BER-ekskreta yang tak terdeteksi
    fp_on_negatives: float | None  # citra tanpa ekskreta yang punya deteksi
    n_images: int
    n_positive: int
    n_negative: int


def iou_xyxy(a: Box | NDArray[np.float64], b: Box | NDArray[np.float64]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def split_dirs(data_yaml: str | Path, split: str) -> tuple[Path, Path]:
    cfg: dict[str, Any] = yaml.safe_load(Path(data_yaml).read_text())
    rel: str = cfg[split]  # mis. images/val
    root = Path(cfg["path"])
    return root / rel, root / rel.replace("images", "labels", 1)


def load_gt(path: Path, w: int, h: int) -> list[Box]:
    """Label YOLO detect (5 kolom, ternormalisasi) -> daftar kotak xyxy piksel."""
    boxes: list[Box] = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:])
            boxes.append(
                (
                    (cx - bw / 2) * w,
                    (cy - bh / 2) * h,
                    (cx + bw / 2) * w,
                    (cy + bh / 2) * h,
                )
            )
    return boxes


def collect(
    model: YOLO, img_dir: Path, lbl_dir: Path, conf: float, iou: float, imgsz: int
) -> list[Item]:
    """Prediksi seluruh citra pada conf rendah; ambang τconf diterapkan kemudian di score()."""
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    items: list[Item] = []
    gen = model.predict(
        source=[str(p) for p in paths],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        stream=True,
        verbose=False,
    )

    gen = cast(list[Results], gen)
    for p, r in zip(paths, gen):
        h, w = r.orig_shape

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = cast(torch.Tensor, r.boxes.xyxy).cpu().numpy()
            confs = cast(torch.Tensor, r.boxes.conf).cpu().numpy()
            o = np.argsort(-confs)
            boxes = boxes[o]
            confs = confs[o]
        else:
            boxes = np.empty((0, 4), dtype=np.float64)
            confs = np.empty((0,), dtype=np.float64)

        items.append(
            {
                "name": p.name,
                "pred": boxes,
                "conf": confs,
                "gt": load_gt(lbl_dir / f"{p.stem}.txt", w, h),
            }
        )
    return items


def score(items: list[Item], tau: float) -> Metrics:
    tp = fp = fn = 0
    ious: list[float] = []
    n_img = n_pos = n_neg = 0
    count_ok = crowded = crowded_ok = 0
    nodet_all = miss_pos = fp_neg = 0
    for it in items:
        pb = it["pred"][it["conf"] >= tau]  # urut menurun berdasarkan confidence
        gt = it["gt"]
        matched: set[int] = set()
        for b in pb:
            best, bi = 0.0, -1
            for gi, g in enumerate(gt):
                if gi in matched:
                    continue
                v = iou_xyxy(b, g)
                if v > best:
                    best, bi = v, gi
            if bi >= 0 and best >= 0.5:
                tp += 1
                matched.add(bi)
                ious.append(best)
            else:
                fp += 1
        fn += len(gt) - len(matched)
        n, g_n = len(pb), len(gt)
        n_img += 1
        count_ok += int(n == g_n)
        nodet_all += int(n == 0)
        if g_n >= 1:
            n_pos += 1
            miss_pos += int(n == 0)
        else:
            n_neg += 1
            fp_neg += int(n > 0)
        if g_n >= 2:
            crowded += 1
            crowded_ok += int(n == g_n)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "count_acc": count_ok / n_img if n_img else 0.0,
        "crowded_n": crowded,
        "crowded_count_acc": crowded_ok / crowded if crowded else None,
        "no_detection_all": nodet_all / n_img if n_img else 0.0,
        "no_detection_on_positives": miss_pos / n_pos if n_pos else None,
        "fp_on_negatives": fp_neg / n_neg if n_neg else None,
        "n_images": n_img,
        "n_positive": n_pos,
        "n_negative": n_neg,
    }


def pick_conf(items: list[Item]) -> tuple[float, float]:
    """τconf = titik F1 maksimum (kurva dihaluskan jendela 5 agar tidak memilih puncak tunggal)."""
    taus = np.round(np.arange(0.01, 1.0, 0.01), 2)
    f1 = np.array([score(items, float(t))["f1"] for t in taus])
    sm = np.convolve(f1, np.ones(5) / 5, mode="same")
    i = int(sm.argmax())
    return float(taus[i]), float(f1[i])


def pct(v: float | None) -> str:
    return "  n/a " if v is None else f"{100 * v:5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="dataset/detector/data.yaml")
    ap.add_argument(
        "--imgsz", type=int, default=640, help="harus sama dengan saat training"
    )
    ap.add_argument(
        "--iou-sweep", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9]
    )
    ap.add_argument("--evaluate-test", action="store_true")
    ap.add_argument(
        "--out", default=None, help="default: <folder data.yaml>/thresholds.json"
    )
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    out = Path(args.out) if args.out else Path(args.data).parent / "thresholds.json"

    if args.evaluate_test:
        if not out.exists():
            sys.exit(
                f"{out} belum ada. Jalankan kalibrasi pada subset validasi terlebih dahulu."
            )
        th: dict[str, Any] = json.loads(out.read_text())
        img_dir, lbl_dir = split_dirs(args.data, "test")
        items = collect(model, img_dir, lbl_dir, 0.001, th["iou"], th["imgsz"])
        res: dict[str, Any] = dict(score(items, th["conf"]))
        res["mAP50"] = float(
            model.val(
                data=args.data,
                split="test",
                imgsz=th["imgsz"],
                conf=0.001,
                iou=th["iou"],
                plots=False,
                verbose=False,
            ).box.map50
        )
        res["thresholds"] = {"conf": th["conf"], "iou": th["iou"], "imgsz": th["imgsz"]}
        print(json.dumps(res, indent=2))
        out.with_name("test_results.json").write_text(json.dumps(res, indent=2))
        return

    img_dir, lbl_dir = split_dirs(args.data, "val")
    base = collect(model, img_dir, lbl_dir, 0.001, DEFAULT_NMS_IOU, args.imgsz)
    tau, f1 = pick_conf(base)
    print(
        f"τconf = {tau:.2f} (F1 = {f1:.3f}, NMS IoU = {DEFAULT_NMS_IOU}, {len(base)} citra val)\n"
    )

    rows: list[tuple[float, Metrics]] = []
    for iou in args.iou_sweep:
        items = (
            base
            if iou == DEFAULT_NMS_IOU
            else collect(model, img_dir, lbl_dir, 0.001, iou, args.imgsz)
        )
        rows.append((iou, score(items, tau)))

    print(
        "NMS IoU |  P     R     F1   | count acc | count acc (>=2 objek) | no_det (positif) | FP pada negatif"
    )
    for iou, s in rows:
        print(
            f"  {iou:.2f}  | {s['precision']:.3f} {s['recall']:.3f} {s['f1']:.3f} |  {pct(s['count_acc'])}  |"
            f"  {pct(s['crowded_count_acc'])} (n={s['crowded_n']:>3})      |  {pct(s['no_detection_on_positives'])}         |"
            f"  {pct(s['fp_on_negatives'])}"
        )

    def key(r: tuple[float, Metrics]) -> tuple[float, float, float]:
        s = r[1]
        primary = (
            s["crowded_count_acc"]
            if s["crowded_count_acc"] is not None
            else s["count_acc"]
        )
        return (primary, s["count_acc"], -abs(r[0] - DEFAULT_NMS_IOU))

    best_iou, best = max(rows, key=key)
    if rows[0][1]["crowded_n"] == 0:
        print(
            "\n[peringatan] tidak ada citra val dengan >= 2 objek; ambang NMS dipilih dari count accuracy keseluruhan."
        )
    elif rows[0][1]["crowded_n"] < 20:
        print(
            f"\n[peringatan] hanya {rows[0][1]['crowded_n']} citra val berobjek >= 2: perbedaan antar ambang NMS rapuh secara statistik."
        )

    out.write_text(
        json.dumps(
            {
                "conf": tau,
                "iou": best_iou,
                "imgsz": args.imgsz,
                "weights": str(args.weights),
                "f1_val_at_conf": f1,
                "val": best,
            },
            indent=2,
        )
    )
    print(f"\nDipilih: conf={tau:.2f}, NMS IoU={best_iou:.2f} -> {out}")


if __name__ == "__main__":
    main()
