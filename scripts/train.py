"""Tahap 3: latih detektor ROI YOLOv8n (satu kelas 'excreta'), transfer learning dari bobot COCO.

    python -m scripts.train
    python -m scripts.train --imgsz 416 --epochs 150

Sesuai panduan:
  - hanya subset latih untuk training, subset validasi untuk seleksi model (best.pt);
  - augmentasi bawaan YOLOv8 dipertahankan (tidak ada override di bawah ini);
  - subset uji TIDAK disentuh di sini. Kalibrasi ambang dan evaluasi uji: scripts.calibrate_detector.
Bobot dan log disimpan di <folder data.yaml>/runs/<name>/.
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/detector/data.yaml")
    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="bawaan YOLOv8; cek distribusi ukuran citra di keluaran prepare_yolo",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--patience",
        type=int,
        default=20,
        help="early stopping pada metrik val (pilihan implementasi, bukan dari panduan)",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="catat nilai ini (panduan 10.2)"
    )
    ap.add_argument("--name", default="train")
    args = ap.parse_args()

    from ultralytics import YOLO  # impor di sini agar modul ringan diimpor tanpa torch

    data = Path(args.data).resolve()
    model = YOLO("yolov8n.pt")  # bobot pralatih COCO
    model.train(
        data=str(data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        project=str(
            data.parent / "runs"
        ),  # path absolut: hindari ultralytics menyisipkan runs/detect/
        name=args.name,
        plots=True,
    )
    
    if model.trainer is not None:
        best = Path(model.trainer.best)
        print(f"\nBobot terbaik: {best}")
        print(
            f"Lanjut: python -m scripts.calibrate_detector --weights {best} --imgsz {args.imgsz}"
        )


if __name__ == "__main__":
    main()
