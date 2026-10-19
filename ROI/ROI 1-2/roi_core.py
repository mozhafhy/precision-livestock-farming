"""Modul inti detektor ROI berbasis Threshold Otsu (5 tahap tunable) dan evaluasi setara calibrate_detector.py."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DETECTOR_DIR = ROOT / "dataset" / "detector"
RAW_DIR = ROOT / "dataset" / "raw"
SPLIT_CSV = ROOT / "dataset" / "split.csv"
COCO_JSON = ROOT / "dataset" / "annotations" / "master_cleaned.json"

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2) dalam piksel


# ----------------------------------------------------------------------------
# Parameter Tunable (menggabungkan spesifikasi teman & 5 tahap saran Gemini)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ROIParams:
    # Tahap 1: Normalisasi warna (toggle on/off & pilihan metode)
    norm_mode: str = "none"        # "none" (off) | "gray_world" | "clahe"
    # Tahap 2: Konversi ruang warna & penghalusan (Gaussian blur)
    color_space: str = "gray"      # "gray" | "lab" | "hsv"
    channel: int = 0               # indeks kanal untuk lab (0=L,1=a,2=b) / hsv (0=H,1=S,2=V)
    blur_ksize: int = 5            # ukuran kernel Gaussian blur (ganjil: 3, 5, 7, ...; 0 = off)
    # Tahap 3: Segmentasi kasar Otsu
    invert: bool = False           # False = objek terang (THRESH_BINARY), True = objek gelap (THRESH_BINARY_INV)
    # Tahap 4: Operasi morfologi (Opening & Closing)
    open_ksize: int = 3            # ukuran kernel Opening untuk hapus bintik luar (ganjil; 0 = off)
    close_ksize: int = 5           # ukuran kernel Closing untuk tambal lubang objek (ganjil; 0 = off)
    # Tahap 5: Pencarian kontur & penambahan ruang (ukuran kontur minimum & padding)
    min_area: int = 100            # luas kontur minimum dalam piksel untuk menyaring sisa derau
    padding: int = 20              # margin tambahan (piksel) di keempat sisi kotak ROI


@dataclass
class ROIResult:
    boxes: list[Box]               # daftar kotak prediksi (kosong bila tidak ada kontur yang lolos)
    threshold: float               # nilai ambang Otsu yang terpilih otomatis
    stages: dict[str, np.ndarray]  # gambar hasil tiap tahap (terisi bila return_stages=True)


def _odd(k: int) -> int:
    """Pastikan ukuran kernel bernilai ganjil positif (atau 0 jika dimatikan)."""
    return k if k <= 0 or k % 2 == 1 else k + 1


# ----------------------------------------------------------------------------
# Tahap 1: Normalisasi Warna (sesuai Modul Bimbingan Tahap 3)
# ----------------------------------------------------------------------------
def normalize_color(img_bgr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return img_bgr
    if mode == "gray_world":
        f = img_bgr.astype(np.float32)
        means = f.reshape(-1, 3).mean(axis=0)
        return np.clip(f * (means.mean() / (means + 1e-6)), 0, 255).astype(np.uint8)
    if mode == "clahe":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    raise ValueError(f"norm_mode tidak dikenal: {mode}")


# ----------------------------------------------------------------------------
# Tahap 2a: Konversi Ruang Warna & Pemilihan Kanal
# ----------------------------------------------------------------------------
def select_channel(img_bgr: np.ndarray, color_space: str, channel: int) -> np.ndarray:
    if color_space == "gray":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if color_space == "lab":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., channel]
    if color_space == "hsv":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., channel]
    raise ValueError(f"color_space tidak dikenal: {color_space}")


# ----------------------------------------------------------------------------
# Pipeline 5 Tahap Deteksi ROI Otsu
# ----------------------------------------------------------------------------
def detect_roi(img_bgr: np.ndarray, p: ROIParams, return_stages: bool = False) -> ROIResult:
    h, w = img_bgr.shape[:2]
    st: dict[str, np.ndarray] = {}
    if return_stages:
        st["01_input"] = img_bgr.copy()

    # Tahap 1: Normalisasi warna (toggle on/off & metode)
    norm_img = normalize_color(img_bgr, p.norm_mode)
    if return_stages:
        st["02_norm"] = norm_img

    # Tahap 2: Konversi ruang warna/grayscale & penghalusan Gaussian blur
    ch = select_channel(norm_img, p.color_space, p.channel)
    if return_stages:
        st["03_channel"] = ch

    kb = _odd(p.blur_ksize)
    blur = cv2.GaussianBlur(ch, (kb, kb), 0) if kb > 0 else ch
    if return_stages:
        st["04_blur"] = blur

    # Tahap 3: Segmentasi kasar dengan Otsu (dengan opsi invert)
    flag = (cv2.THRESH_BINARY_INV if p.invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
    thr, otsu = cv2.threshold(blur, 0, 255, flag)
    if return_stages:
        st["05_otsu"] = otsu

    # Tahap 4: Operasi morfologi berurutan (Opening lalu Closing)
    ko = _odd(p.open_ksize)
    if ko > 0:
        kern_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
        opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kern_o)
    else:
        opened = otsu
    if return_stages:
        st["06_opening"] = opened

    kc = _odd(p.close_ksize)
    if kc > 0:
        kern_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kern_c)
    else:
        closed = opened
    if return_stages:
        st["07_closing"] = closed

    # Tahap 5: Pencarian kontur terbesar, filter luas minimum, dan penambahan padding
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= p.min_area]

    boxes: list[Box] = []
    best_contour = None
    if valid:
        best_contour = max(valid, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(best_contour)
        pad = max(0, int(p.padding))
        x1 = float(max(0, x - pad))
        y1 = float(max(0, y - pad))
        x2 = float(min(w, x + bw + pad))
        y2 = float(min(h, y + bh + pad))
        boxes.append((x1, y1, x2, y2))

    if return_stages:
        vis = img_bgr.copy()
        if best_contour is not None:
            cv2.drawContours(vis, [best_contour], -1, (0, 255, 255), 1)
        for bx in boxes:
            cv2.rectangle(vis, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 255, 0), 2)
        st["08_roi"] = vis

    return ROIResult(boxes=boxes, threshold=float(thr), stages=st)


# ----------------------------------------------------------------------------
# Pemuatan Dataset & Ground Truth (kompatibel dengan dataset/detector & calibrate_detector.py)
# ----------------------------------------------------------------------------
def iou_xyxy(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def coverage_xyxy(roi: Box, gt: Box) -> float:
    """Proporsi luas kotak GT yang tercakup di dalam kotak ROI."""
    ix = max(0.0, min(roi[2], gt[2]) - max(roi[0], gt[0]))
    iy = max(0.0, min(roi[3], gt[3]) - max(roi[1], gt[1]))
    a_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return float((ix * iy) / a_gt) if a_gt > 0 else 0.0


def load_yolo_gt(label_path: Path, w: int, h: int) -> list[Box]:
    """Membaca label YOLO (cls cx cy bw bh ternormalisasi) menjadi daftar kotak piksel xyxy."""
    boxes: list[Box] = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def load_split_dataset(split: str) -> list[tuple[str, np.ndarray, list[Box]]]:
    """Memuat (image_id, img_bgr, gt_boxes) untuk subset train/val/test."""
    img_dir = DETECTOR_DIR / "images" / split
    lbl_dir = DETECTOR_DIR / "labels" / split
    if img_dir.exists() and lbl_dir.exists():
        items: list[tuple[str, np.ndarray, list[Box]]] = []
        for p in sorted(img_dir.glob("*.jpg")):
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            items.append((p.stem, img, load_yolo_gt(lbl_dir / f"{p.stem}.txt", w, h)))
        return items

    # Fallback ke dataset/raw + master_cleaned.json bila folder detector belum dibuat
    coco = json.loads(COCO_JSON.read_text(encoding="utf-8"))
    id_to_stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    gt_map: dict[str, list[Box]] = {}
    for a in coco["annotations"]:
        x, y, bw, bh = a["bbox"]
        gt_map.setdefault(id_to_stem[a["image_id"]], []).append((float(x), float(y), float(x + bw), float(y + bh)))
    with open(SPLIT_CSV, encoding="utf-8") as f:
        stems = [r["image_id"] for r in csv.DictReader(f) if r["split"] == split]
    return [(s, cv2.imread(str(RAW_DIR / f"{s}.jpg")), gt_map.get(s, [])) for s in stems]


# ----------------------------------------------------------------------------
# Evaluasi Metrik (identik dengan score() di scripts/calibrate_detector.py + metrik cakupan ROI)
# ----------------------------------------------------------------------------
def evaluate(p: ROIParams, dataset: list[tuple[str, np.ndarray, list[Box]]], collect: bool = False):
    tp = fp = fn = 0
    tp_ious: list[float] = []
    all_ious: list[float] = []
    coverages: list[float] = []
    roi_fracs: list[float] = []
    times_ms: list[float] = []

    n_img = n_pos = n_neg = 0
    count_ok = crowded = crowded_ok = 0
    nodet_all = miss_pos = fp_neg = 0
    rows: list[dict[str, Any]] = []

    for stem, img, gt in dataset:
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        res = detect_roi(img, p, return_stages=False)
        ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(ms)

        pb = res.boxes
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
                tp_ious.append(best)
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

        # Metrik per citra untuk analisis kualitas ROI (IoU terbaik, coverage, dan rasio luas ROI)
        if pb and gt:
            best_pair_iou = max(iou_xyxy(b, g) for b in pb for g in gt)
            best_cov = max(coverage_xyxy(b, g) for b in pb for g in gt)
        else:
            best_pair_iou = 0.0
            best_cov = 0.0
        roi_frac = sum((b[2] - b[0]) * (b[3] - b[1]) for b in pb) / (w * h) if pb else 0.0
        all_ious.append(best_pair_iou)
        if g_n >= 1:
            coverages.append(best_cov)
        roi_fracs.append(roi_frac)

        if collect:
            rows.append({
                "image_id": stem,
                "iou": best_pair_iou,
                "coverage": best_cov,
                "roi_frac": roi_frac,
                "n_pred": n,
                "n_gt": g_n,
                "ms": ms,
                "pred_box": pb[0] if pb else None,
                "gt_boxes": gt,
            })

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    metrics: dict[str, Any] = {
        # Metrik utama yang sama persis dengan scripts/calibrate_detector.py
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_iou": float(np.mean(tp_ious)) if tp_ious else 0.0,
        "count_acc": count_ok / n_img if n_img else 0.0,
        "crowded_n": crowded,
        "crowded_count_acc": (crowded_ok / crowded) if crowded else None,
        "no_detection_all": nodet_all / n_img if n_img else 0.0,
        "no_detection_on_positives": (miss_pos / n_pos) if n_pos else None,
        "fp_on_negatives": (fp_neg / n_neg) if n_neg else None,
        "n_images": n_img,
        "n_positive": n_pos,
        "n_negative": n_neg,
        # Metrik tambahan pemantauan pemotongan ROI
        "mean_iou_all": float(np.mean(all_ious)) if all_ious else 0.0,
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "crop_hit_cov90": float(np.mean([c >= 0.9 for c in coverages])) if coverages else 0.0,
        "mean_roi_frac": float(np.mean(roi_fracs)) if roi_fracs else 0.0,
        "ms_per_image": float(np.mean(times_ms)) if times_ms else 0.0,
    }
    return (metrics, rows) if collect else metrics


# ----------------------------------------------------------------------------
# Visualisasi Tahap demi Tahap
# ----------------------------------------------------------------------------
def _label_tile(img: np.ndarray, title: str) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(out, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_montage(stages: dict[str, np.ndarray], gt_boxes: list[Box] | None = None, cols: int = 4) -> np.ndarray:
    """Menyusun 8 panel tahapan ke dalam satu gambar grid (2x4). Kotak GT digambar biru di panel 08_roi."""
    tiles: list[np.ndarray] = []
    for name, im in stages.items():
        tile = _label_tile(im, name)
        if name == "08_roi" and gt_boxes:
            for g in gt_boxes:
                cv2.rectangle(tile, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (255, 0, 0), 2)
        tiles.append(tile)
    while len(tiles) % cols != 0:
        tiles.append(np.zeros_like(tiles[0]))
    grid_rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(grid_rows)


# ----------------------------------------------------------------------------
# Runner CLI untuk roi_otsu.py
# ----------------------------------------------------------------------------
def _apply_autocrop_to_dataset(dataset: list[tuple[str, np.ndarray, list[Box]]]) -> list[tuple[str, np.ndarray, list[Box]]]:
    ac_dir = Path(__file__).resolve().parent.parent / "autocropping"
    meta = json.loads((ac_dir / "crop_meta.json").read_text(encoding="utf-8"))["images"]
    out: list[tuple[str, np.ndarray, list[Box]]] = []
    for stem, _, gt_list in dataset:
        im = cv2.imread(str(ac_dir / "images" / f"{stem}.jpg"))
        if im is None:
            continue
        if stem in meta:
            cx1, cy1, cx2, cy2 = meta[stem]["crop_box"]
            cw, ch = cx2 - cx1, cy2 - cy1
            shifted = [
                (
                    float(max(0.0, g[0] - cx1)),
                    float(max(0.0, g[1] - cy1)),
                    float(min(cw, g[2] - cx1)),
                    float(min(ch, g[3] - cy1)),
                )
                for g in gt_list
            ]
        else:
            shifted = gt_list
        out.append((stem, im, shifted))
    return out


def run_detector(params: ROIParams) -> None:
    ap = argparse.ArgumentParser(description="Evaluasi dan visualisasi tahapan Detektor ROI Otsu")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Subset evaluasi")
    ap.add_argument("--image", help="Jalankan pada 1 citra saja dan simpan tahapan visualnya")
    ap.add_argument("--n-stages", type=int, default=10, help="Jumlah citra yang disimpan per file tahap")
    ap.add_argument("--autocrop", action="store_true", help="Gunakan citra dari ROI/autocropping/images dan simpan ke output_autocrop/")
    ap.add_argument("--out", default=None, help="Folder keluaran")
    args = ap.parse_args()

    default_out = "output_autocrop" if args.autocrop else "output"
    out_dir = Path(args.out) if args.out else (Path(__file__).resolve().parent / default_out)
    (out_dir / "montage").mkdir(parents=True, exist_ok=True)

    if args.image:
        img_path = Path(args.image)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Citra tidak ditemukan: {img_path}")
        res = detect_roi(img, params, return_stages=True)
        cv2.imwrite(str(out_dir / "montage" / f"{img_path.stem}.png"), make_montage(res.stages))
        stage_dir = out_dir / "stages" / img_path.stem
        stage_dir.mkdir(parents=True, exist_ok=True)
        for sname, simg in res.stages.items():
            cv2.imwrite(str(stage_dir / f"{sname}.png"), simg)
        return

    dataset = load_split_dataset(args.split)
    if args.autocrop:
        dataset = _apply_autocrop_to_dataset(dataset)
    metrics, rows = evaluate(params, dataset, collect=True)

    # Simpan gambar montage (seluruh citra pada split) dan file terpisah per tahap (n_stages citra)
    for idx, (stem, img, gt) in enumerate(dataset):
        res = detect_roi(img, params, return_stages=True)
        cv2.imwrite(str(out_dir / "montage" / f"{stem}.png"), make_montage(res.stages, gt))
        if idx < args.n_stages:
            sdir = out_dir / "stages" / stem
            sdir.mkdir(parents=True, exist_ok=True)
            for sname, simg in res.stages.items():
                cv2.imwrite(str(sdir / f"{sname}.png"), simg)

    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "iou", "coverage", "roi_frac", "n_pred", "n_gt", "ms", "pred_box", "gt_boxes"])
        for r in sorted(rows, key=lambda x: x["iou"]):
            w.writerow([
                r["image_id"],
                f"{r['iou']:.4f}",
                f"{r['coverage']:.4f}",
                f"{r['roi_frac']:.4f}",
                r["n_pred"],
                r["n_gt"],
                f"{r['ms']:.2f}",
                r["pred_box"],
                r["gt_boxes"],
            ])

    summary = {"split": args.split, "params": asdict(params), "metrics": metrics}
    (out_dir / f"{args.split}_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
