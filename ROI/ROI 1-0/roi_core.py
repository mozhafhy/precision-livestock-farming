"""Detektor ROI ekskreta berbasis threshold Otsu (tunable) + evaluasi + visualisasi tiap tahap.

Alur deteksi (setiap tahap bisa disimpan sebagai gambar):
    01 input -> 02 pemilihan kanal warna -> 03 Gaussian blur -> 04 threshold Otsu
    -> 05 penyeragaman polaritas -> 06 morphological closing
    -> 07 filter kontur (ukuran minimum) -> 08 kotak ROI

Modul ini dipakai oleh roi_otsu.py dan roi_tune.py.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent   # folder precision-livestock-farming
IMG_DIR = ROOT / "dataset" / "raw"
COCO_JSON = ROOT / "dataset" / "annotations" / "master_cleaned.json"
SPLIT_CSV = ROOT / "dataset" / "split.csv"


# ----------------------------------------------------------------------------
# Parameter (semua yang bisa di-tuning ada di sini)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ROIParams:
    color_space: str = "gray"      # "gray" | "lab" | "hsv"
    channel: int = 0               # indeks kanal untuk lab (0=L,1=a,2=b) / hsv (0=H,1=S,2=V); diabaikan pada gray
    blur_ksize: int = 5            # kernel Gaussian blur (ganjil; 0 = tanpa blur)
    close_ksize: int = 5           # kernel morphological closing (ganjil; 0 = tanpa closing)
    min_area_frac: float = 0.002   # luas kontur minimum sebagai fraksi luas citra (tidak bergantung resolusi)
    max_area_frac: float = 0.95    # kontur yang hampir sebesar citra dianggap latar, dibuang
    polarity: str = "auto"         # "auto" (objek = sisi yang lebih kecil) | "dark" | "bright"
    pad_frac: float = 0.0          # margin tambahan di sekeliling kotak ROI (fraksi sisi kotak)
    roi_mode: str = "union"        # "union" (gabung semua kontur) | "largest" (kontur terbesar saja)


# ----------------------------------------------------------------------------
# Detektor
# ----------------------------------------------------------------------------
@dataclass
class ROIResult:
    box: tuple             # (x1, y1, x2, y2) dalam piksel
    fallback: bool         # True bila tidak ada kontur lolos -> ROI = seluruh citra
    threshold: float       # nilai ambang Otsu
    stages: dict           # nama tahap -> citra BGR (hanya terisi jika return_stages=True)


def _odd(k):
    """Pastikan ukuran kernel ganjil (0 dibiarkan 0)."""
    return k if k <= 0 or k % 2 == 1 else k + 1


def select_channel(img_bgr, color_space, channel):
    """Ubah ke color space yang dipilih dan ambil satu kanal uint8."""
    if color_space == "gray":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if color_space == "lab":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., channel]
    if color_space == "hsv":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., channel]
    raise ValueError(f"color_space tidak dikenal: {color_space}")


def detect_roi(img_bgr, p: ROIParams, return_stages=False) -> ROIResult:
    h, w = img_bgr.shape[:2]
    st = {}
    if return_stages:
        st["01_input"] = img_bgr.copy()

    # 02: kanal warna
    ch = select_channel(img_bgr, p.color_space, p.channel)
    if return_stages:
        st["02_channel"] = ch

    # 03: Gaussian blur untuk menekan derau/tekstur halus latar
    k = _odd(p.blur_ksize)
    blur = cv2.GaussianBlur(ch, (k, k), 0) if k > 0 else ch
    if return_stages:
        st["03_blur"] = blur

    # 04: threshold Otsu (ambang dicari otomatis dari histogram)
    thr, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if return_stages:
        st["04_otsu"] = binary

    # 05: penyeragaman polaritas agar objek = putih (255)
    if p.polarity == "dark":
        binary = 255 - binary
    elif p.polarity == "auto" and (binary > 0).mean() > 0.5:
        binary = 255 - binary                  # objek diasumsikan menempati area lebih kecil dari latar
    if return_stages:
        st["05_polarity"] = binary

    # 06: closing untuk menutup lubang dan menyambung pecahan objek
    k = _odd(p.close_ksize)
    if k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if return_stages:
        st["06_closing"] = binary

    # 07: filter kontur berdasarkan luas
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lo, hi = p.min_area_frac * h * w, p.max_area_frac * h * w
    kept = [c for c in contours if lo <= cv2.contourArea(c) <= hi]
    if return_stages:
        vis = img_bgr.copy()
        cv2.drawContours(vis, kept, -1, (0, 255, 0), 1)
        st["07_contours"] = vis

    # 08: kotak ROI
    if not kept:
        box, fallback = (0, 0, w, h), True     # tidak ada kandidat -> pakai seluruh citra
    else:
        if p.roi_mode == "largest":
            kept = [max(kept, key=cv2.contourArea)]
        rects = np.array([cv2.boundingRect(c) for c in kept])      # x, y, w, h
        x1, y1 = rects[:, 0].min(), rects[:, 1].min()
        x2, y2 = (rects[:, 0] + rects[:, 2]).max(), (rects[:, 1] + rects[:, 3]).max()
        px, py = (x2 - x1) * p.pad_frac, (y2 - y1) * p.pad_frac
        box = (int(max(0, x1 - px)), int(max(0, y1 - py)),
               int(min(w, x2 + px)), int(min(h, y2 + py)))
        fallback = False
    if return_stages:
        vis = img_bgr.copy()
        cv2.rectangle(vis, box[:2], box[2:], (0, 255, 0), 2)
        st["08_roi"] = vis
    return ROIResult(box, fallback, float(thr), st)


# ----------------------------------------------------------------------------
# Ground truth & metrik
# ----------------------------------------------------------------------------
def load_ground_truth():
    """Kembalikan {image_id: (x1,y1,x2,y2)}: gabungan semua bbox anotasi pada citra tsb."""
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
    return {"iou": inter / (a_roi + a_gt - inter + 1e-9),
            "coverage": inter / (a_gt + 1e-9),
            "roi_frac": a_roi / (shape[0] * shape[1])}


def load_images(ids):
    return {i: cv2.imread(str(IMG_DIR / f"{i}.jpg")) for i in ids}


def evaluate(p: ROIParams, images, gt, collect=False):
    """Evaluasi parameter p pada kumpulan citra. Hanya citra yang punya GT yang dihitung."""
    rows = []
    for i, img in images.items():
        if i not in gt:
            continue
        t0 = time.perf_counter()
        r = detect_roi(img, p)
        ms = (time.perf_counter() - t0) * 1000
        m = box_metrics(r.box, gt[i], img.shape)
        rows.append({"image_id": i, **m, "fallback": int(r.fallback), "ms": ms,
                     "box": r.box, "gt": gt[i]})
    a = lambda k: np.array([r[k] for r in rows])
    summary = {"n": len(rows),
               "mean_iou": a("iou").mean(),
               "median_iou": float(np.median(a("iou"))),
               "mean_coverage": a("coverage").mean(),
               "hit_rate_cov90": (a("coverage") >= 0.9).mean(),   # ROI memuat >=90% objek
               "hit_rate_iou50": (a("iou") >= 0.5).mean(),
               "mean_roi_frac": a("roi_frac").mean(),             # makin kecil = latar terbuang makin banyak
               "fallback_rate": a("fallback").mean(),
               "ms_per_image": a("ms").mean()}
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
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
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
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"],
                    help="subset data yang dievaluasi (default: test)")
    ap.add_argument("--image", help="path satu citra bebas (tanpa evaluasi), hanya visualisasi tahap")
    ap.add_argument("--n-stages", type=int, default=10,
                    help="jumlah citra yang tiap tahapnya disimpan sebagai file terpisah")
    ap.add_argument("--autocrop", action="store_true",
                    help="gunakan citra dari ROI/autocropping/images dan simpan ke output_autocrop/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    default_out = "output_autocrop" if args.autocrop else "output"
    out = (Path(args.out) if args.out else (Path(__file__).parent / default_out)) / name
    (out / "montage").mkdir(parents=True, exist_ok=True)

    if args.image:                                           # mode satu citra
        img = cv2.imread(args.image)
        r = detect_roi(img, params, return_stages=True)
        cv2.imwrite(str(out / "montage" / (Path(args.image).stem + ".png")), montage(r.stages))
        return

    gt = load_ground_truth()
    split = load_split()
    ids = sum(split.values(), []) if args.split == "all" else split[args.split]
    if args.autocrop:
        images, gt = load_autocrop_data(ids, gt)
    else:
        images = load_images(ids)
    summary, rows = evaluate(params, images, gt, collect=True)

    # montage seluruh tahap untuk semua citra + file terpisah per tahap untuk sebagian citra
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
        for r in sorted(rows, key=lambda r: r["iou"]):       # terburuk di atas -> mudah ditinjau
            w.writerow([r["image_id"], f"{r['iou']:.4f}", f"{r['coverage']:.4f}", f"{r['roi_frac']:.4f}",
                        r["fallback"], f"{r['ms']:.2f}", r["box"], r["gt"]])
    json.dump({"preset": name, "split": args.split, "params": asdict(params),
               "metrics": {k: float(v) for k, v in summary.items()}},
              open(out / "summary.json", "w", encoding="utf8"), indent=2)

    print(f"[{name}] split={args.split} n={summary['n']}")
    for k, v in summary.items():
        if k != "n":
            print(f"  {k:16s}: {v:.4f}")
    print(f"  output: {out}")
