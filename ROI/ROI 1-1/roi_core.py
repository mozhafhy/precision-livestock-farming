"""Detektor ROI ekskreta berbasis Threshold Otsu + Kontras Latar & Prior Spasial Terpusat.

Alur deteksi (setiap tahap disimpan sebagai gambar):
    01_input        : citra asli BGR
    02_channel      : konversi ruang warna (LAB/HSV/Gray) + Gaussian blur
    03_bg_contrast  : selisih warna terhadap latar tepi & latar lokal (Difference of Gaussians)
    04_center_prior : pembobotan spasial Gaussian terpusat (menekan derau/bayangan pinggir frame)
    05_otsu         : binarisasi otomatis dengan Threshold Otsu
    06_closing      : operasi morfologi Opening + Closing untuk menyatukan bagian urat & feses
    07_contours     : penyaringan ukuran kontur minimum & penilaian skor kontur (luas + posisi + tepi)
    08_roi          : kotak ROI prediksi (hijau) dibandingkan dengan master_cleaned.json (biru)
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent   # folder precision-livestock-farming
IMG_DIR = ROOT / "dataset" / "raw"
COCO_JSON = ROOT / "dataset" / "annotations" / "master_cleaned.json"
SPLIT_CSV = ROOT / "dataset" / "split.csv"


# ----------------------------------------------------------------------------
# Parameter Tunable (4 parameter dasar + metode tambahan agar mendekati master_cleaned.json)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ROIParams:
    color_space: str = "lab"         # 1. color space: "lab" | "hsv" | "gray"
    channel: int = 0                 # indeks kanal bila memakai 1 kanal (diabaikan saat use_bg_contrast=True)
    blur_ksize: int = 5              # 2. kernel Gaussian blur (ganjil; 0 = tanpa blur)
    open_ksize: int = 3              # kernel morphological opening untuk membuang bintik sekam kecil
    close_ksize: int = 15            # 3. kernel morphological closing untuk menyatukan gumpalan ekskreta
    min_area_frac: float = 0.002     # 4. ukuran kontur minimum (fraksi luas citra)
    max_area_frac: float = 0.85      # batas atas luas kontur agar latar penuh tidak terpilih
    use_bg_contrast: bool = True     # metode tambahan 1: pengurangan warna latar tepi & kontras lokal (DoG)
    border_margin: float = 0.12      # lebar tepi citra (fraksi) yang dijadikan referensi warna latar
    center_sigma: float = 0.30       # metode tambahan 2: lebar Gaussian prior pusat citra (0 = matikan)
    merge_score_ratio: float = 0.55  # metode tambahan 3: gabungkan kontur dekat pusat dengan skor >= rasio ini
    pad_frac: float = 0.04           # margin tambahan di sekeliling kotak ROI (fraksi ukuran kotak)


@dataclass
class ROIResult:
    box: tuple             # (x1, y1, x2, y2) dalam piksel
    fallback: bool         # True bila tidak ada kontur lolos -> fallback ke kotak tengah
    threshold: float       # nilai ambang Otsu
    stages: dict           # nama tahap -> citra untuk visualisasi


def _odd(k: int) -> int:
    """Pastikan ukuran kernel ganjil positif (0 dibiarkan 0)."""
    return k if k <= 0 or k % 2 == 1 else k + 1


def _to_color_space(img_bgr: np.ndarray, color_space: str) -> np.ndarray:
    """Konversi citra BGR ke ruang warna yang dipilih."""
    if color_space == "lab":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    if color_space == "hsv":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if color_space == "gray":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"color_space tidak dikenal: {color_space}")


def _compute_bg_contrast(img_cs: np.ndarray, border_margin: float) -> np.ndarray:
    """Hitung peta kontras objek terhadap warna latar tepi dan latar lokal (Difference of Gaussians)."""
    h, w = img_cs.shape[:2]
    f = img_cs.astype(np.float32)
    if f.ndim == 2:
        f = f[..., None]

    # Masker piksel pinggir bingkai sebagai sampel warna latar belakang (lantai/sekam)
    b_mask = np.zeros((h, w), dtype=bool)
    by, bx = max(1, int(h * border_margin)), max(1, int(w * border_margin))
    b_mask[:by, :] = b_mask[-by:, :] = b_mask[:, :bx] = b_mask[:, -bx:] = True

    bg_med = np.median(f[b_mask], axis=0)
    bg_std = np.std(f[b_mask], axis=0) + 10.0

    # Latar lokal frekuensi rendah (Gaussian lebar) untuk meredam bayangan tidak merata
    bg_local = cv2.GaussianBlur(f, (0, 0), sigmaX=w * 0.18)
    if bg_local.ndim == 2:
        bg_local = bg_local[..., None]

    weights = np.array([0.7, 1.2, 1.2], dtype=np.float32) if f.shape[2] == 3 else np.array([1.0], dtype=np.float32)
    dist_global = np.sqrt((((f - bg_med) / bg_std) ** 2) @ weights)
    dist_local = np.sqrt((((f - bg_local) / bg_std) ** 2) @ weights)

    sal = 0.5 * dist_global + 0.5 * dist_local
    return cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def detect_roi(img_bgr: np.ndarray, p: ROIParams, return_stages: bool = False) -> ROIResult:
    h, w = img_bgr.shape[:2]
    st = {}
    if return_stages:
        st["01_input"] = img_bgr.copy()

    # 02: Konversi ruang warna & Gaussian blur
    cs_img = _to_color_space(img_bgr, p.color_space)
    kb = _odd(p.blur_ksize)
    blurred = cv2.GaussianBlur(cs_img, (kb, kb), 0) if kb > 0 else cs_img
    if return_stages:
        st["02_channel"] = blurred if blurred.ndim == 2 else blurred[..., p.channel]

    # 03: Peta kontras terhadap latar belakang (atau kanal tunggal jika dimatikan)
    if p.use_bg_contrast:
        contrast = _compute_bg_contrast(blurred, p.border_margin)
    else:
        contrast = blurred if blurred.ndim == 2 else blurred[..., p.channel]
    if return_stages:
        st["03_bg_contrast"] = contrast

    # 04: Pembobotan spasial Gaussian terpusat (Center-Surround Prior)
    cy, cx = h * 0.49, w * 0.50
    if p.center_sigma > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        w_map = np.exp(-0.5 * (((xx - cx) / (w * p.center_sigma)) ** 2 + ((yy - cy) / (h * p.center_sigma)) ** 2))
        sal_weighted = cv2.normalize(contrast.astype(np.float32) * w_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        sal_weighted = contrast
    if return_stages:
        st["04_center_prior"] = sal_weighted

    # 05: Threshold Otsu
    thr, binary = cv2.threshold(sal_weighted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not p.use_bg_contrast and (binary > 0).mean() > 0.5:
        binary = 255 - binary
    if return_stages:
        st["05_otsu"] = binary

    # 06: Morfologi Opening lalu Closing
    work = binary.copy()
    ko = _odd(p.open_ksize)
    if ko > 0:
        kern_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kern_o)
    kc = _odd(p.close_ksize)
    if kc > 0:
        kern_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kern_c)
    if return_stages:
        st["06_closing"] = work

    # 07: Penyaringan luas kontur & penilaian skor (luas, kedekatan ke pusat, penalti tepi frame)
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lo, hi = p.min_area_frac * h * w, p.max_area_frac * h * w
    scored = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (lo <= area <= hi):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        bcx, bcy = x + bw / 2.0, y + bh / 2.0
        center_w = np.exp(-0.5 * (((bcx - cx) / (w * 0.25)) ** 2 + ((bcy - cy) / (h * 0.25)) ** 2))
        border_touch = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 1) + int(y + bh >= h - 1)
        score = (area ** 0.7) * (center_w ** 1.5) * (0.6 ** border_touch)
        scored.append((score, c, (x, y, x + bw, y + bh)))

    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        top_score = scored[0][0]
        selected = [item for item in scored if item[0] >= top_score * p.merge_score_ratio]
        kept_contours = [item[1] for item in selected]
        kept_boxes = [item[2] for item in selected]
        x1 = min(b[0] for b in kept_boxes)
        y1 = min(b[1] for b in kept_boxes)
        x2 = max(b[2] for b in kept_boxes)
        y2 = max(b[3] for b in kept_boxes)
        px, py = (x2 - x1) * p.pad_frac, (y2 - y1) * p.pad_frac
        box = (
            int(max(0, x1 - px)),
            int(max(0, y1 - py)),
            int(min(w, x2 + px)),
            int(min(h, y2 + py)),
        )
        fallback = False
    else:
        kept_contours = []
        box = (int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))
        fallback = True

    if return_stages:
        vis_c = img_bgr.copy()
        cv2.drawContours(vis_c, [item[1] for item in scored], -1, (0, 165, 255), 1)
        cv2.drawContours(vis_c, kept_contours, -1, (0, 255, 0), 2)
        st["07_contours"] = vis_c

        vis_r = img_bgr.copy()
        cv2.rectangle(vis_r, box[:2], box[2:], (0, 255, 0), 2)
        st["08_roi"] = vis_r

    return ROIResult(box, fallback, float(thr), st)


# ----------------------------------------------------------------------------
# Ground truth & metrik evaluasi terhadap master_cleaned.json
# ----------------------------------------------------------------------------
def load_ground_truth():
    """Kembalikan {image_id: (x1,y1,x2,y2)}: gabungan semua bbox anotasi pada master_cleaned.json."""
    coco = json.load(open(COCO_JSON, encoding="utf8"))
    names = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    gt = {}
    for a in coco["annotations"]:
        x, y, bw, bh = a["bbox"]
        b = gt.setdefault(names[a["image_id"]], [x, y, x + bw, y + bh])
        b[:] = [min(b[0], x), min(b[1], y), max(b[2], x + bw), max(b[3], y + bh)]
    return {k: tuple(v) for k, v in gt.items()}


def load_split():
    """Kembalikan {split: [image_id, ...]} dari split.csv."""
    out = {}
    with open(SPLIT_CSV, encoding="utf8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["split"], []).append(r["image_id"])
    return out


def box_metrics(roi, gt, shape):
    """IoU, coverage (porsi GT yang masuk ROI), dan porsi luas ROI terhadap citra."""
    ix = max(0, min(roi[2], gt[2]) - max(roi[0], gt[0]))
    iy = max(0, min(roi[3], gt[3]) - max(roi[1], gt[1]))
    inter = ix * iy
    a_roi = (roi[2] - roi[0]) * (roi[3] - roi[1])
    a_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return {
        "iou": inter / (a_roi + a_gt - inter + 1e-9),
        "coverage": inter / (a_gt + 1e-9),
        "roi_frac": a_roi / (shape[0] * shape[1]),
    }


def load_images(ids):
    return {i: cv2.imread(str(IMG_DIR / f"{i}.jpg")) for i in ids}


def evaluate(p: ROIParams, images, gt, collect=False):
    """Evaluasi parameter p pada kumpulan citra terhadap master_cleaned.json."""
    rows = []
    for i, img in images.items():
        if i not in gt:
            continue
        t0 = time.perf_counter()
        r = detect_roi(img, p)
        ms = (time.perf_counter() - t0) * 1000
        m = box_metrics(r.box, gt[i], img.shape)
        rows.append({"image_id": i, **m, "fallback": int(r.fallback), "ms": ms, "box": r.box, "gt": gt[i]})
    a = lambda k: np.array([r[k] for r in rows])
    summary = {
        "n": len(rows),
        "mean_iou": float(a("iou").mean()),
        "median_iou": float(np.median(a("iou"))),
        "mean_coverage": float(a("coverage").mean()),
        "hit_rate_cov90": float((a("coverage") >= 0.9).mean()),
        "hit_rate_iou50": float((a("iou") >= 0.5).mean()),
        "mean_roi_frac": float(a("roi_frac").mean()),
        "fallback_rate": float(a("fallback").mean()),
        "ms_per_image": float(a("ms").mean()),
    }
    return (summary, rows) if collect else summary


# ----------------------------------------------------------------------------
# Visualisasi
# ----------------------------------------------------------------------------
def _label(img, text):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(img, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def montage(stages, gt_box=None, cols=4):
    """Gabungkan seluruh tahap ke satu gambar grid. Kotak GT (biru) digambar pada tahap akhir."""
    tiles = []
    for name, im in stages.items():
        im = _label(im, name)
        if gt_box is not None and name == "08_roi":
            cv2.rectangle(im, tuple(map(int, gt_box[:2])), tuple(map(int, gt_box[2:])), (255, 0, 0), 2)
        tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def load_autocrop_data(ids, gt):
    """Muat citra hasil autocrop dan geser koordinat GT ke sistem koordinat citra autocrop."""
    ac_dir = Path(__file__).resolve().parent.parent / "autocropping"
    meta = json.loads((ac_dir / "crop_meta.json").read_text(encoding="utf-8"))["images"]
    images, gt_shifted = {}, {}
    for i in ids:
        im = cv2.imread(str(ac_dir / "images" / f"{i}.jpg"))
        if im is None:
            continue
        images[i] = im
        if i in gt and i in meta:
            cx1, cy1, cx2, cy2 = meta[i]["crop_box"]
            cw, ch = cx2 - cx1, cy2 - cy1
            gx1, gy1, gx2, gy2 = gt[i]
            gt_shifted[i] = (max(0, gx1 - cx1), max(0, gy1 - cy1), min(cw, gx2 - cx1), min(ch, gy2 - cy1))
    return images, gt_shifted


# ----------------------------------------------------------------------------
# Runner untuk preset
# ----------------------------------------------------------------------------
def run_preset(name, params: ROIParams):
    ap = argparse.ArgumentParser(description=f"Detektor ROI Otsu - preset '{name}'")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--image", help="path satu citra bebas (tanpa evaluasi), hanya visualisasi tahap")
    ap.add_argument("--n-stages", type=int, default=10)
    ap.add_argument("--autocrop", action="store_true",
                    help="gunakan citra dari ROI/autocropping/images dan simpan ke output_autocrop/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    default_out = "output_autocrop" if args.autocrop else "output"
    out = (Path(args.out) if args.out else (Path(__file__).parent / default_out)) / name
    (out / "montage").mkdir(parents=True, exist_ok=True)

    if args.image:
        img = cv2.imread(args.image)
        r = detect_roi(img, params, return_stages=True)
        cv2.imwrite(str(out / "montage" / (Path(args.image).stem + ".png")), montage(r.stages))
        return

    gt = load_ground_truth()
    split = load_split()
    ids = sum(split.values(), []) if args.split == "all" else split[args.split]
    if args.autocrop:
        images, gt = load_autocrop_data(ids, gt)
        params = ROIParams(
            **{
                **asdict(params),
                "border_margin": 0.06,
                "center_sigma": 0.38,
                "pad_frac": 0.06,
            }
        )
    else:
        images = load_images(ids)
    summary, rows = evaluate(params, images, gt, collect=True)

    for k, r in enumerate(rows):
        res = detect_roi(images[r["image_id"]], params, return_stages=True)
        cv2.imwrite(str(out / "montage" / f"{r['image_id']}.png"), montage(res.stages, r["gt"]))
        if k < args.n_stages:
            d = out / "stages" / r["image_id"]
            d.mkdir(parents=True, exist_ok=True)
            for sname, im in res.stages.items():
                cv2.imwrite(str(d / f"{sname}.png"), im)

    with open(out / "per_image.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "iou", "coverage", "roi_frac", "fallback", "ms", "box", "gt_box"])
        for r in sorted(rows, key=lambda r: r["iou"]):
            w.writerow([
                r["image_id"],
                f"{r['iou']:.4f}",
                f"{r['coverage']:.4f}",
                f"{r['roi_frac']:.4f}",
                r["fallback"],
                f"{r['ms']:.2f}",
                r["box"],
                r["gt"],
            ])
    json.dump(
        {
            "preset": name,
            "split": args.split,
            "params": asdict(params),
            "metrics": {k: float(v) for k, v in summary.items()},
        },
        open(out / "summary.json", "w", encoding="utf8"),
        indent=2,
    )

    print(f"[{name}] split={args.split} n={summary['n']}")
    for k, v in summary.items():
        if k != "n":
            print(f"  {k:16s}: {v:.4f}")
    print(f"  output: {out}")
