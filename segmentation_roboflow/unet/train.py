"""Skrip Pelatihan U-Net MobileNetV2 untuk Segmentasi Ekskreta Berbasis Roboflow.
Mendukung 2 Jenis Pemrosesan (Mask-Based & BoundingBox-Based) serta 2 Varian Warna (Raw & Gray World).

Fitur & Kemampuan:
1. Jenis Pemrosesan:
   - mask_based: Menggunakan citra dan masker penuh (di-resize langsung ke 512x512).
   - bbox_based: Memotong citra dan masker pas sesuai bounding box anotasi Roboflow, lalu di-resize ke 512x512.
2. Varian Warna:
   - raw: Menggunakan citra asli (dataset/raw/).
   - gray_world: Menggunakan citra koreksi warna (dataset/rawGrayworld/).
3. Opsi Eksekusi Otomatis (4 Eksperimen Sekaligus):
   - bbox_based + raw
   - bbox_based + gray_world
   - mask_based + raw
   - mask_based + gray_world
4. Artefak Lengkap Mirip YOLOv8 (tersimpan di runs/<mode>/<variant>/):
   - weights/best.pth (bobot terbaik berdasarkan IoU validasi tertinggi)
   - weights/last.pth (bobot epoch terakhir)
   - args.yaml (parameter konfigurasi pelatihan)
   - results.csv (log per epoch: loss, IoU, Dice, LR)
   - results.png (grafik loss & IoU per epoch)
   - confusion_matrix.png & confusion_matrix_normalized.png (matriks kebingungan piksel)
   - val_batch0_labels.jpg & val_batch0_pred.jpg (visualisasi ground-truth vs prediksi)
   - comparison_summary.csv (tabel rekap gabungan seluruh eksperimen)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

# Mencegah pembuatan file __pycache__
sys.dont_write_bytecode = True


# -----------------------------------------------------------------------------
# Penentuan Root Proyek Otomatis
# -----------------------------------------------------------------------------
def find_project_root(current_path: Path) -> Path:
    for parent in [current_path.resolve()] + list(current_path.resolve().parents):
        if (parent / "dataset").exists():
            return parent
    return current_path.resolve().parents[2]


ROOT = find_project_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import albumentations as A
    import segmentation_models_pytorch as smp
except ImportError:
    sys.exit(
        "Pustaka pendukung belum terpasang!\n"
        "Jalankan: pip install segmentation-models-pytorch albumentations pyyaml"
    )

from utils.file_managements import load_json

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -----------------------------------------------------------------------------
# 1. Dataset Loader Universal (Mendukung Mask-Based & Bbox-Based)
# -----------------------------------------------------------------------------
class RoboflowExcretaDataset(Dataset):
    """Dataset pembaca pasangan citra dan masker biner dengan opsi cropping bboxes."""

    def __init__(
        self,
        mode: str,  # 'mask_based' atau 'bbox_based'
        variant: str,  # 'raw' atau 'gray_world'
        split: str,  # 'train', 'val', atau 'test'
        target_size: int = 512,
        is_train: bool = False,
    ):
        self.mode = mode
        self.variant = variant
        self.split = split
        self.target_size = target_size
        self.is_train = is_train

        # Tentukan direktori citra berdasarkan varian warna
        if variant == "raw":
            self.images_base_dir = ROOT / "dataset" / "raw"
        elif variant == "gray_world":
            self.images_base_dir = ROOT / "dataset" / "rawGrayworld" / split
        else:
            raise ValueError(f"Varian warna tidak valid: {variant}")

        self.masks_base_dir = ROOT / "dataset" / "rawMask" / split

        # Muat anotasi split JSON untuk daftar gambar & bounding box
        split_json_path = ROOT / "dataset" / "annotations" / f"{split}.json"
        if not split_json_path.exists():
            raise FileNotFoundError(f"File anotasi tidak ditemukan: {split_json_path}")

        split_data = load_json(split_json_path)
        self.images_info = split_data.get("images", [])

        # Kumpulkan bounding boxes gabungan per gambar untuk mode bbox_based
        self.image_bboxes = defaultdict(list)
        for ann in split_data.get("annotations", []):
            if "bbox" in ann:
                self.image_bboxes[ann["image_id"]].append(ann["bbox"])

        # Augmentasi Albumentations (Modul Hal. 17-18)
        if self.is_train:
            self.geom_transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                ]
            )
            # Augmentasi warna hanya pada citra
            self.color_transform = A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            )
        else:
            self.geom_transform = None
            self.color_transform = None

    def __len__(self) -> int:
        return len(self.images_info)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_info = self.images_info[idx]
        image_id = img_info["id"]
        filename = img_info["file_name"]
        stem = Path(filename).stem

        # Lokasi file citra
        if self.variant == "raw":
            img_path = self.images_base_dir / filename
        else:
            img_path = self.images_base_dir / f"{stem}.jpg"

        mask_path = self.masks_base_dir / f"{stem}.png"

        # Baca citra RGB & masker biner
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Gagal membaca citra: {img_path}")
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Gagal membaca masker: {mask_path}")

        orig_h, orig_w = image.shape[:2]

        # Logika Pemotongan Bounding Box (jika mode bbox_based)
        if self.mode == "bbox_based":
            bboxes = self.image_bboxes.get(image_id, [])
            if len(bboxes) > 0:
                # Ambil union (kotak terluar yang memuat seluruh ekskreta)
                x1 = int(min(b[0] for b in bboxes))
                y1 = int(min(b[1] for b in bboxes))
                x2 = int(max(b[0] + b[2] for b in bboxes))
                y2 = int(max(b[1] + b[3] for b in bboxes))

                # Batasi dalam batas gambar
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)

                if x2 > x1 and y2 > y1:
                    image = image[y1:y2, x1:x2]
                    mask = mask[y1:y2, x1:x2]

        # Resize seragam ke target_size (512x512)
        image = cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        # Terapkan augmentasi
        if self.geom_transform is not None:
            augmented = self.geom_transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if self.color_transform is not None:
            image = self.color_transform(image=image)["image"]

        # Normalisasi ImageNet
        image_norm = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).float()  # (3, H, W)

        # Normalisasi Masker ke biner [0.0, 1.0]
        mask_bin = (mask > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0).float()  # (1, H, W)

        return image_tensor, mask_tensor, stem


# -----------------------------------------------------------------------------
# 2. Fungsi Metrik & Matriks Kebingungan Piksel
# -----------------------------------------------------------------------------
def compute_metrics_and_cm(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> tuple[float, float, dict[str, int]]:
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        total_p = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
        dice = ((2.0 * intersection + 1e-6) / (total_p + 1e-6)).mean().item()

        # Hitung True Positive, False Positive, False Negative, True Negative
        tp = int((preds.bool() & targets.bool()).sum().item())
        fp = int((preds.bool() & ~targets.bool()).sum().item())
        fn = int((~preds.bool() & targets.bool()).sum().item())
        tn = int((~preds.bool() & ~targets.bool()).sum().item())

    return iou, dice, {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


# -----------------------------------------------------------------------------
# 3. Fungsi Pembangkit Artefak Visual Mirip YOLO (runs/train-2/)
# -----------------------------------------------------------------------------
def save_confusion_matrix_plots(cm_totals: dict[str, int], output_dir: Path) -> None:
    """Menyimpan matriks kebingungan piksel standar dan ternormalisasi."""
    tp, fp, fn, tn = cm_totals["tp"], cm_totals["fp"], cm_totals["fn"], cm_totals["tn"]
    cm = np.array([[tp, fn], [fp, tn]], dtype=np.float64)

    labels = ["Excreta", "Background"]

    # 1. Confusion Matrix Mentah
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Pixel-level Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(str(output_dir / "confusion_matrix.png"), dpi=200)
    plt.close()

    # 2. Confusion Matrix Ternormalisasi
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-6)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm_norm[i, j]:.3f}", ha="center", va="center", color="black" if cm_norm[i, j] < 0.5 else "white")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(str(output_dir / "confusion_matrix_normalized.png"), dpi=200)
    plt.close()


def save_val_batch_visualizations(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    max_images: int = 16,
) -> None:
    """Menyimpan visualisasi perbandingan batch validasi mirip val_batch0_pred.jpg."""
    model.eval()
    images_list, labels_list, preds_list = [], [], []

    with torch.no_grad():
        for images, masks, stems in val_loader:
            images_gpu = images.to(device)
            logits = model(images_gpu)
            preds = (torch.sigmoid(logits) > 0.5).float()

            for i in range(images.size(0)):
                # Denormalisasi citra
                img_np = images[i].permute(1, 2, 0).cpu().numpy()
                img_denorm = np.clip((img_np * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255).astype(np.uint8)

                gt_mask = (masks[i, 0].cpu().numpy() * 255).astype(np.uint8)
                pred_mask = (preds[i, 0].cpu().numpy() * 255).astype(np.uint8)

                images_list.append(img_denorm)
                labels_list.append(gt_mask)
                preds_list.append(pred_mask)

                if len(images_list) >= max_images:
                    break
            if len(images_list) >= max_images:
                break

    n = len(images_list)
    cols = 4
    rows = int(np.ceil(n / cols))

    # Simpan val_batch0_labels.jpg (Ground Truth Overlay)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        if i < n:
            overlay = images_list[i].copy()
            overlay[labels_list[i] > 127] = (0.5 * overlay[labels_list[i] > 127] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            axes[i].imshow(overlay)
            axes[i].set_title(f"Label (GT) {i+1}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(str(output_dir / "val_batch0_labels.jpg"), dpi=150)
    plt.close()

    # Simpan val_batch0_pred.jpg (Prediction Overlay)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        if i < n:
            overlay = images_list[i].copy()
            overlay[preds_list[i] > 127] = (0.5 * overlay[preds_list[i] > 127] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            axes[i].imshow(overlay)
            axes[i].set_title(f"Prediction {i+1}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(str(output_dir / "val_batch0_pred.jpg"), dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# 4. Fungsi Pelatihan untuk 1 Sesi Eksperimen
# -----------------------------------------------------------------------------
def train_experiment(
    mode: str,
    variant: str,
    output_root: Path,
    device: torch.device,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    imgsz: int = 512,
) -> dict[str, Any]:
    run_name = f"{mode}/{variant}"
    run_dir = output_root / mode / variant
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"MEMULAI PELATIHAN: {mode.upper()} - {variant.upper()}")
    print(f"Folder Output : {run_dir}")
    print(f"Perangkat     : {device}")
    print("=" * 80)

    # Simpan args.yaml
    args_dict = {
        "mode": mode,
        "variant": variant,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "imgsz": imgsz,
        "encoder": "mobilenet_v2",
        "loss": "BCEWithLogitsLoss + DiceLoss",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "device": str(device),
    }
    with open(run_dir / "args.yaml", "w") as f:
        yaml.dump(args_dict, f, default_flow_style=False)

    # Siapkan Dataset & DataLoader
    train_dataset = RoboflowExcretaDataset(mode, variant, "train", target_size=imgsz, is_train=True)
    val_dataset = RoboflowExcretaDataset(mode, variant, "val", target_size=imgsz, is_train=False)

    num_workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    print(f"Data Sampel: {len(train_dataset)} Train | {len(val_dataset)} Validation")

    # Inisialisasi Model U-Net MobileNetV2
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # CSV Logger
    results_csv = run_dir / "results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "lr"])

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": [], "val_dice": [], "lr": []}
    best_val_iou = 0.0
    best_epoch = 0
    patience_count = 0
    cm_totals_best = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = bce_loss_fn(logits, masks) + dice_loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_train_loss / len(train_dataset)

        # Validasi
        model.eval()
        running_val_loss = 0.0
        running_val_iou = 0.0
        running_val_dice = 0.0
        cm_epoch = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                logits = model(images)
                loss = bce_loss_fn(logits, masks) + dice_loss_fn(logits, masks)
                running_val_loss += loss.item() * images.size(0)

                b_iou, b_dice, b_cm = compute_metrics_and_cm(logits, masks)
                running_val_iou += b_iou * images.size(0)
                running_val_dice += b_dice * images.size(0)

                for k in cm_epoch:
                    cm_epoch[k] += b_cm[k]

        val_loss = running_val_loss / len(val_dataset)
        val_iou = running_val_iou / len(val_dataset)
        val_dice = running_val_dice / len(val_dataset)
        current_lr = scheduler.get_last_lr()[0]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)
        history["lr"].append(current_lr)

        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_iou:.4f}", f"{val_dice:.4f}", f"{current_lr:.6f}"])

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou*100:.2f}% | "
            f"Val Dice: {val_dice*100:.2f}%"
        )

        # Simpan bobot terbaik
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            patience_count = 0
            cm_totals_best = cm_epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_iou": val_iou,
                    "val_dice": val_dice,
                    "mode": mode,
                    "variant": variant,
                },
                weights_dir / "best.pth",
            )
            print(f"  -> ⭐ [BEST MODEL DISIMPAN] Val IoU: {best_val_iou*100:.2f}% (Epoch {epoch})")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n[EARLY STOPPING] Tidak ada peningkatan IoU selama {patience} epoch berturut-turut.")
                break

    # Simpan bobot terakhir
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_iou": val_iou,
            "val_dice": val_dice,
            "mode": mode,
            "variant": variant,
        },
        weights_dir / "last.pth",
    )

    # 1. Simpan results.png (Kurva Loss & Metrik)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="train/loss", color="royalblue")
    plt.plot(history["epoch"], history["val_loss"], label="val/loss", color="darkorange")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], [x * 100 for x in history["val_iou"]], label="metrics/IoU (%)", color="forestgreen")
    plt.plot(history["epoch"], [x * 100 for x in history["val_dice"]], label="metrics/Dice (%)", color="crimson")
    plt.axhline(y=85.0, color="gray", linestyle="--", label="Target >=85%")
    plt.title("Performance Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(run_dir / "results.png"), dpi=200)
    plt.close()

    # 2. Simpan Matriks Kebingungan Piksel
    save_confusion_matrix_plots(cm_totals_best, run_dir)

    # 3. Muat bobot terbaik untuk membuat visualisasi prediksi (val_batch0_pred.jpg)
    checkpoint = torch.load(weights_dir / "best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_val_batch_visualizations(model, val_loader, device, run_dir)

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"-> Pelatihan {run_name} Selesai dalam {elapsed_min:.2f} menit!")
    print(f"-> Seluruh artefak tersimpan di: {run_dir}")

    return {
        "mode": mode,
        "variant": variant,
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "best_val_dice": history["val_dice"][best_epoch - 1] if best_epoch > 0 else 0.0,
    }


# -----------------------------------------------------------------------------
# 5. Orkestrator Utama (Mendukung All 4 Runs Sekaligus)
# -----------------------------------------------------------------------------
def run_all(
    mode_arg: str = "all",
    variant_arg: str = "all",
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 12,
    imgsz: int = 512,
) -> None:
    output_root = ROOT / "segmentation_roboflow" / "unet" / "runs"
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modes = ["bbox_based", "mask_based"] if mode_arg == "all" else [mode_arg]
    variants = ["raw", "gray_world"] if variant_arg == "all" else [variant_arg]

    experiments_to_run = [(m, v) for m in modes for v in variants]

    print("=" * 80)
    print("ORKESTRATOR PELATIHAN SEGMENTASI ROBOFLOW (UNET MOBILENETV2)")
    print(f"Jumlah Sesi Eksperimen yang Dijalankan: {len(experiments_to_run)}")
    for m, v in experiments_to_run:
        print(f"  - Mode: {m:<15} | Varian: {v}")
    print("=" * 80)

    summary_list = []

    for m, v in experiments_to_run:
        res = train_experiment(
            mode=m,
            variant=v,
            output_root=output_root,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            imgsz=imgsz,
        )
        summary_list.append(res)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Rekap Ringkasan Akhir
    summary_csv = output_root / "comparison_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "variant", "best_epoch", "best_val_iou", "best_val_dice", "target_85_achieved"])
        for res in summary_list:
            achieved = "YA (>=85%)" if res["best_val_iou"] >= 0.85 else "TIDAK"
            writer.writerow([res["mode"], res["variant"], res["best_epoch"], f"{res['best_val_iou']:.4f}", f"{res['best_val_dice']:.4f}", achieved])

    print("\n" + "=" * 80)
    print("REKAPITULASI HASIL KESELURUHAN EKSPERIMEN")
    print("=" * 80)
    print(f"{'Mode':<15} | {'Varian':<12} | {'Best Epoch':<10} | {'Val IoU':<12} | {'Val Dice':<12} | {'Target >=85%'}")
    print("-" * 80)
    for res in summary_list:
        achieved = "TERCAPAI" if res["best_val_iou"] >= 0.85 else "BELUM"
        print(
            f"{res['mode']:<15} | {res['variant']:<12} | {res['best_epoch']:<10} | "
            f"{res['best_val_iou']*100:.2f}%{'':<6} | {res['best_val_dice']*100:.2f}%{'':<6} | {achieved}"
        )
    print("=" * 80)
    print(f"Tabel rekapitulasi tersimpan di: {summary_csv}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net MobileNetV2 on Roboflow Datasets")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "mask_based", "bbox_based"], help="Mode pemrosesan")
    parser.add_argument("--variant", type=str, default="all", choices=["all", "raw", "gray_world"], help="Varian warna")
    parser.add_argument("--epochs", type=int, default=60, help="Jumlah epoch maksimal")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=12, help="Patience early stopping")
    parser.add_argument("--imgsz", type=int, default=512, help="Ukuran resolusi target")
    args = parser.parse_args()

    run_all(
        mode_arg=args.mode,
        variant_arg=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        imgsz=args.imgsz,
    )
