"""Skenario E6: Analisis Galat (Tabel 12 & Gambar 3)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(".")
E1_DIR = ROOT / "results" / "E1_arsitektur"
E6_DIR = ROOT / "results" / "E6_galat"
RAW_DIR = ROOT / "dataset" / "raw" / "kgl"
COCO_JSON = ROOT / "dataset" / "annotations" / "kgl" / "annotations.coco.json"
IOU_THRESHOLD = 0.70

def find_best_run_dir() -> Path:
    csvs = list(E1_DIR.glob("**/metrics_per_image.csv"))
    if not csvs:
        raise FileNotFoundError("Belum ada metrics_per_image.csv di E1.")
    return max(csvs, key=lambda p: pd.read_csv(p)["iou_exc"].mean()).parent

def get_ground_truth_mask(coco_data: dict, file_name: str, h: int, w: int) -> np.ndarray:
    from pycocotools import mask as mask_utils
    img_id = None
    for im in coco_data["images"]:
        if im["file_name"] == file_name or Path(im["file_name"]).name == Path(file_name).name:
            img_id = im["id"]; h = im.get("height", h); w = im.get("width", w); break
    if img_id is None: return np.zeros((h, w), dtype=np.uint8)

    anns = [a for a in coco_data["annotations"] if a["image_id"] == img_id and a.get("category_id") == 1]
    mask = np.zeros((h, w), dtype=np.uint8)
    for a in anns:
        seg = a.get("segmentation")
        if not seg: continue
        try:
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                m = mask_utils.decode(mask_utils.merge(rles))
            else:
                m = mask_utils.decode(seg)
            if m.ndim == 3: m = np.any(m, axis=2)
            mask |= m.astype(np.uint8)
        except Exception: pass
    return mask

def make_diagnostic_overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, np.logical_not(gt))
    fn = np.logical_and(np.logical_not(pred), gt)
    overlay = rgb.copy().astype(np.float32)
    alpha = 0.55
    overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0, 255, 0], dtype=np.float32) * alpha       # Hijau = TP
    overlay[fp] = overlay[fp] * (1 - alpha) + np.array([255, 0, 0], dtype=np.float32) * alpha       # Merah = FP
    overlay[fn] = overlay[fn] * (1 - alpha) + np.array([0, 100, 255], dtype=np.float32) * alpha     # Biru = FN
    return np.clip(overlay, 0, 255).astype(np.uint8)

def classify_error_case(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> list[str]:
    categories = []
    tp, fp, fn = np.logical_and(pred, gt).sum(), np.logical_and(pred, ~gt).sum(), np.logical_and(~pred, gt).sum()
    num_labels, _ = cv2.connectedComponents(gt.astype(np.uint8))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fp_mask = np.logical_and(pred, ~gt)

    if (num_labels - 1) > 1:
        categories.append("Beberapa objek dalam satu citra")
    if gray.mean() < 85 or (gray.std() > 65 and np.sum(gray < 40) > 0.1 * gray.size):
        categories.append("Bayangan kuat")
    if fp > 0 and fp_mask.sum() > 50 and gray[fp_mask].mean() > 140:
        categories.append("Sekam/bulu putih terdeteksi sebagai ekskreta (FP)")
    if fn > fp and fn > 0.25 * gt.sum():
        categories.append("Tudung urat tipis terpotong (FN)")
    if fp > fn and fp > 0.20 * pred.sum():
        categories.append("Ekskreta basah menyebar ke sekam lembap (FP)")
    if not categories:
        categories.append("Ekskreta basah menyebar ke sekam lembap (FP)" if fp > fn else "Tudung urat tipis terpotong (FN)")
    return categories

def main():
    E6_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = find_best_run_dir()
    print(f"[E6] Menganalisis galat dari model terbaik: {run_dir.name}")
    df_metrics = pd.read_csv(run_dir / "metrics_per_image.csv")
    with open(COCO_JSON, encoding="utf-8") as f: coco_data = json.load(f)

    bad_cases = df_metrics[df_metrics["iou_exc"] < IOU_THRESHOLD].copy()
    print(f"Total citra uji: {len(df_metrics)}")
    print(f"Citra uji galat (IoU < {IOU_THRESHOLD:.2f}): {len(bad_cases)} citra ({len(bad_cases)/len(df_metrics)*100:.1f}%)")

    pred_dir = run_dir / "pred_masks"
    classified_records, analyzed_items = [], []
    category_counts = {
        "Ekskreta basah menyebar ke sekam lembap (FP)": [],
        "Tudung urat tipis terpotong (FN)": [],
        "Sekam/bulu putih terdeteksi sebagai ekskreta (FP)": [],
        "Bayangan kuat": [],
        "Beberapa objek dalam satu citra": [],
        "Lainnya: Oklusi kawat sangkar / Ekskreta sangat kecil": [],
    }

    for _, row in bad_cases.iterrows():
        stem = str(row["image"])
        file_name = str(row.get("file_name", f"{stem}.jpg"))
        iou = float(row["iou_exc"])

        img_path = RAW_DIR / file_name
        if not img_path.exists(): img_path = RAW_DIR / f"{stem}.jpg"
        if not img_path.exists():
            matches = list(RAW_DIR.glob(f"*{stem}*"))
            if matches: img_path = matches[0]
        if not img_path.exists(): continue

        bgr = cv2.imread(str(img_path))
        if bgr is None: continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        gt = get_ground_truth_mask(coco_data, file_name, h, w)

        pred_path = pred_dir / f"{stem}.png"
        if pred_path.exists():
            pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            if pred.shape[:2] != (h, w): pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            pred = (pred > 127).astype(np.uint8)
        else:
            pred = np.zeros((h, w), dtype=np.uint8)

        overlay = make_diagnostic_overlay(rgb, gt, pred)
        cv2.imwrite(str(E6_DIR / f"overlay_{stem}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        cats = classify_error_case(rgb, gt, pred)
        for c in cats: category_counts[c].append(stem)

        classified_records.append({"image_id": stem, "file_name": file_name, "iou_exc": iou, "kategori": "; ".join(cats)})
        analyzed_items.append({"stem": stem, "iou": iou, "rgb": rgb, "gt": gt, "pred": pred, "overlay": overlay, "cats": cats})

    # Simpan kategori_galat.csv (luaran wajib)
    pd.DataFrame(classified_records).to_csv(E6_DIR / "kategori_galat.csv", index=False)

    notes_map = {
        "Ekskreta basah menyebar ke sekam lembap (FP)": "Kontur meluber ke serbuk sekam akibat warna cokelat/basah yang identik",
        "Tudung urat tipis terpotong (FN)": "Batas asam urat tipis di tepi kotoran terlewat akibat kontras rendah",
        "Sekam/bulu putih terdeteksi sebagai ekskreta (FP)": "Bulu ayam putih/sekam terang disalahartikan sebagai asam urat",
        "Bayangan kuat": "Gradien bayangan kawat sangkar mengaburkan batas ekskreta",
        "Beberapa objek dalam satu citra": "Gumpalan kotoran terpisah-pisah mengalami miss pada objek kecil",
        "Lainnya: Oklusi kawat sangkar / Ekskreta sangat kecil": "Ekskreta tertutup kawat baterai atau ukuran < 15 piksel",
    }

    tabel12_rows = []
    for cat_name, img_list in category_counts.items():
        tabel12_rows.append({
            "Kategori galat": cat_name,
            "Jumlah citra": len(img_list),
            "Contoh image_id": ", ".join(img_list[:2]) if img_list else "-",
            "Catatan": notes_map.get(cat_name, "-"),
        })

    df_t12 = pd.DataFrame(tabel12_rows)
    df_t12.to_csv(E6_DIR / "tabel12.csv", index=False)
    try: (E6_DIR / "tabel12.md").write_text(df_t12.to_markdown(index=False), encoding="utf-8")
    except Exception: (E6_DIR / "tabel12.md").write_text(df_t12.to_string(index=False), encoding="utf-8")

    # Generate Gambar 3: 3 Kasus Representatif Multi-panel
    selected_cases = []
    for target in ["Ekskreta basah menyebar ke sekam lembap (FP)", "Tudung urat tipis terpotong (FN)", "Sekam/bulu putih terdeteksi sebagai ekskreta (FP)"]:
        for it in analyzed_items:
            if target in it["cats"] and it not in selected_cases:
                selected_cases.append((target, it)); break
    while len(selected_cases) < 3 and len(analyzed_items) > len(selected_cases):
        for it in sorted(analyzed_items, key=lambda x: x["iou"]):
            if it not in [s[1] for s in selected_cases]:
                selected_cases.append((it["cats"][0], it)); break

    if len(selected_cases) >= 3:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        col_titles = ["Citra Asli", "Ground Truth", "Prediksi Model", "Overlay (Hijau=TP, Merah=FP, Biru=FN)"]
        for r_idx, (cat_label, item) in enumerate(selected_cases[:3]):
            axes[r_idx, 0].imshow(item["rgb"])
            axes[r_idx, 0].set_title(f"{col_titles[0]}\n(ID: {item['stem']}, IoU={item['iou']:.3f})", fontsize=10)
            axes[r_idx, 0].set_ylabel(f"Kasus {r_idx+1}\n[{cat_label.split('(')[0].strip()}]", fontsize=11, fontweight="bold")
            axes[r_idx, 1].imshow(item["gt"], cmap="gray"); axes[r_idx, 1].set_title(col_titles[1], fontsize=10)
            axes[r_idx, 2].imshow(item["pred"], cmap="gray"); axes[r_idx, 2].set_title(col_titles[2], fontsize=10)
            axes[r_idx, 3].imshow(item["overlay"]); axes[r_idx, 3].set_title(col_titles[3], fontsize=10)
            for c_idx in range(4): axes[r_idx, c_idx].set_xticks([]); axes[r_idx, c_idx].set_yticks([])
        plt.suptitle("Gambar 3: Tiga Contoh Kasus Galat Representatif (Skenario E6)", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(E6_DIR / "gambar3_representatif.png", dpi=200, bbox_inches="tight")
        plt.close()

    print("\n================ ISIAN E6: KATEGORI GALAT (TABEL 12) ================")
    print(df_t12.to_string(index=False))
    print(f"\nBerkas resmi tersimpan di:")
    print(f" - {E6_DIR / 'kategori_galat.csv'} (Luaran Wajib Panduan Dosen)")
    print(f" - {E6_DIR / 'tabel12.csv'} dan {E6_DIR / 'tabel12.md'}")
    print(f" - {E6_DIR / 'gambar3_representatif.png'} (Gambar 3 Laporan)")

if __name__ == "__main__":
    main()
