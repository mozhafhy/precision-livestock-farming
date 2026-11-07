"""Skenario E2: Ablasi Normalisasi Citra (Tabel 8).

=============================================================================
KETERANGAN KEBUTUHAN DATA DARI DRIVE / HASIL TRAINING:
- Membutuhkan hasil E1 selesai (terutama Tabel 7 / metrics_per_image.csv) 
  untuk mengetahui Model Pemenang (Best Model: misal yolov8n-seg atau yolov8s-seg)
  serta hasil Config C yang sudah terlatih di E1.
- Menjalankan pelatihan Best Model pada:
    * Konfigurasi A: Tanpa normalisasi spasial & spektral (3 seeds: 0, 1, 2)
    * Konfigurasi B: Normalisasi spasial saja (3 seeds: 0, 1, 2)
- Menggabungkan hasil Config A, Config B, dan Config C (dari E1) ke Tabel 8.
=============================================================================

Tabel 8 Laporan:
- Konfigurasi Normalisasi (A, B, C)
- IoU Ekskreta (Rerata ± SD)
- DSC Ekskreta (Rerata ± SD)
- mIoU (Rerata ± SD)
- Presisi (Rerata ± SD)
- Recall (Rerata ± SD)

Output:
- results/E2_ablasi/tabel8.csv
- results/E2_ablasi/tabel8.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.e1.build_datasets import build_all
from scripts.e1.common import CONFIGS, load_protocol, run_id
from scripts.e1.metrics import METRIC_COLS
from scripts.e1.run import execute_run

ROOT = Path(".")
E1_DIR = ROOT / "results" / "E1_arsitektur"
E2_DIR = ROOT / "results" / "E2_ablasi"


def get_best_model_from_e1() -> str:
    """Membaca Tabel 7 E1 untuk memilih model dengan IoU uji tertinggi."""
    t7_csv = E1_DIR / "tabel7.csv"
    if t7_csv.exists():
        df = pd.read_csv(t7_csv)
        # Ambil kolom IoU ekskreta
        col = next(c for c in df.columns if "IoU ekskreta" in c or "iou_exc" in c)
        # Parse nilai mean sebelum simbol ±
        means = df[col].apply(lambda s: float(str(s).split("±")[0].strip()))
        best_idx = means.idxmax()
        best_model = df.loc[best_idx, "Model"]
        print(f"[E2] Model Terbaik terpilih dari E1: {best_model} (IoU: {means[best_idx]:.4f})")
        return str(best_model)
    
    # Fallback default jika tabel7 belum digenerate: cari dari metrics_summary
    summaries = list(E1_DIR.glob("**/metrics_summary.json"))
    if summaries:
        scores = {}
        for s in summaries:
            data = json.loads(s.read_text(encoding="utf-8"))
            m = data["model"]
            iou = data["test"]["mean"]["iou_exc"]
            scores.setdefault(m, []).append(iou)
        mean_scores = {m: np.mean(v) for m, v in scores.items()}
        best_model = max(mean_scores, key=mean_scores.get)
        print(f"[E2] Model Terbaik terpilih dari summary JSON: {best_model} (IoU: {mean_scores[best_model]:.4f})")
        return best_model

    print("[E2 Peringatan] Hasil E1 belum lengkap di folder lokal. Menggunakan default: yolov8n-seg")
    return "yolov8n-seg"


def load_metrics_for_config(model: str, cfg: str, seeds: list[int]) -> dict[str, list[float]]:
    """Memuat metrik dari E1 (untuk cfg C) atau E2 (untuk cfg A dan B)."""
    metrics = {c: [] for c in METRIC_COLS}
    for s in seeds:
        rid = run_id(model, cfg, s)
        # Cek di E2 atau E1
        p = E2_DIR / rid / "metrics_per_image.csv"
        if not p.exists():
            p = E1_DIR / rid / "metrics_per_image.csv"
        
        if not p.exists():
            raise FileNotFoundError(f"Berkas metrik tidak ditemukan untuk {rid} di E1 maupun E2.")
        
        df = pd.read_csv(p)
        for c in METRIC_COLS:
            metrics[c].append(float(df[c].mean()))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Menjalankan Skenario E2: Ablasi Normalisasi Citra")
    parser.add_argument("--model", default=None, help="Model yang diuji (default: Best Model dari E1)")
    parser.add_argument("--device", default=None, help="Device komputasi (cuda atau cpu)")
    parser.add_argument("--skip-train", action="store_true", help="Lewati training, langsung agregasi tabel jika sudah ada")
    args = parser.parse_args()

    E2_DIR.mkdir(parents=True, exist_ok=True)
    proto = load_protocol()
    seeds = proto["seeds"]

    model_name = args.model or get_best_model_from_e1()
    print(f"\n=======================================================")
    print(f"  SKENARIO E2: ABLASI NORMALISASI PADA MODEL: {model_name}")
    print(f"=======================================================\n")

    if not args.skip_train:
        for cfg in ["A", "B"]:
            print(f"\n>>> Memeriksa / Membangun Dataset Config {cfg}...")
            # Buat dataset jika belum ada
            cfg_dir = E1_DIR / f"data_cfg{cfg}"
            if not (cfg_dir / "data.yaml").exists():
                print(f"Membangun dataset config {cfg}...")
                build_all(config=cfg, force=False)

            # Latih 3 seeds
            for s in seeds:
                print(f"\n>>> Melatih {model_name} | Config {cfg} | Seed {s}...")
                execute_run(model_name, s, cfg=cfg, device=args.device)

    # Agregasi ke Tabel 8
    print("\n>>> Mengagregasi Hasil ke Tabel 8...")
    cfg_labels = {
        "A": "Konfigurasi A (Tanpa Normalisasi Spasial & Spektral)",
        "B": "Konfigurasi B (Normalisasi Spasial Saja / ROI + Letterbox)",
        "C": "Konfigurasi C (Normalisasi Spasial + Spektral / Lengkap)",
    }

    rows = []
    for cfg in ["A", "B", "C"]:
        try:
            m = load_metrics_for_config(model_name, cfg, seeds)
            row = {
                "Konfigurasi": cfg_labels[cfg],
                "IoU Ekskreta": f"{np.mean(m['iou_exc']):.4f} ± {np.std(m['iou_exc'], ddof=1):.4f}",
                "DSC Ekskreta": f"{np.mean(m['dsc_exc']):.4f} ± {np.std(m['dsc_exc'], ddof=1):.4f}",
                "mIoU": f"{np.mean(m['miou']):.4f} ± {np.std(m['miou'], ddof=1):.4f}",
                "Presisi": f"{np.mean(m['precision']):.4f} ± {np.std(m['precision'], ddof=1):.4f}",
                "Recall": f"{np.mean(m['recall']):.4f} ± {np.std(m['recall'], ddof=1):.4f}",
            }
            rows.append(row)
        except Exception as e:
            print(f"[Peringatan] Gagal memuat data Config {cfg}: {e}")

    if rows:
        df_t8 = pd.DataFrame(rows)
        out_csv = E2_DIR / "tabel8.csv"
        out_md = E2_DIR / "tabel8.md"
        df_t8.to_csv(out_csv, index=False)
        out_md.write_text(df_t8.to_markdown(index=False), encoding="utf-8")

        print("\n================ TABEL 8: ABLASI NORMALISASI ================")
        print(df_t8.to_string(index=False))
        print(f"\nHasil Tabel 8 disimpan di {out_csv} dan {out_md}")
    else:
        print("[Error] Tidak ada data metrik yang berhasil dimuat untuk Tabel 8.")


if __name__ == "__main__":
    main()
