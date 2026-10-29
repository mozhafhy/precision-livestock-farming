"""Skrip Pelatihan U-Net dengan Encoder MobileNetV2 untuk Segmentasi Ekskreta Ayam.
Sesuai Dokumen Bimbingan: Tahap 4, 5a, dan 6 (Model Ringan untuk Komputasi Edge).

Fitur Utama:
1. Arsitektur: U-Net (smp.Unet) dengan Backbone MobileNetV2 pralatih ImageNet.
2. Fungsi Kerugian Gabungan: BCEWithLogitsLoss + DiceLoss (smp.losses.DiceLoss).
3. Optimizer: AdamW (lr=1e-3, weight_decay=1e-4) + Penjadwal CosineAnnealingLR.
4. Augmentasi Data: Albumentations (Flip, Rotate, Brightness) tanpa membocorkan augmentasi warna ke masker.
5. Early Stopping: Berhenti bila Val IoU tidak meningkat selama 12 epoch berturut-turut (maks 60 epoch).
6. Multi-Varian Eksperimen (E3 & E4):
   - Opsi --variant raw (Eksperimen E3: tanpa normalisasi warna)
   - Opsi --variant gray_world (Eksperimen E4: dengan normalisasi Gray World)
   - Opsi --variant both (Jalankan keduanya secara berurutan dalam 1 kali run)
7. Output Artefak ke runs_unet/:
   - weights/best.pth (bobot terbaik berdasarkan IoU validasi tertinggi)
   - weights/last.pth (bobot epoch terakhir)
   - results.csv (log metrik per epoch)
   - training_curves.png (grafik loss dan IoU)
   - comparison_summary.csv (rekap hasil E3 vs E4)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Mencegah pembuatan file __pycache__
sys.dont_write_bytecode = True

# Menentukan root proyek secara dinamis (mencari folder induk yang memuat folder 'dataset')
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
        "Pustaka penting belum terpasang!\n"
        "Jalankan: pip install segmentation-models-pytorch albumentations"
    )

# Standar normalisasi ImageNet (sesuai modul hal. 14)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -----------------------------------------------------------------------------
# 1. Dataset & Pipeline Augmentasi (Fase 2: Data Pipeline PyTorch)
# -----------------------------------------------------------------------------
class ExcretaCropDataset(Dataset):
    """Dataset pembaca pasangan citra crop dan masker biner 512x512."""

    def __init__(self, images_dir: Path, masks_dir: Path, is_train: bool = False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.is_train = is_train

        self.image_paths = sorted(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"Tidak ada citra ditemukan di: {images_dir}")

        # Augmentasi geometris (diterapkan sama ke citra & masker)
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
            # Augmentasi warna/kecerahan (HANYA diterapkan ke citra, Modul Hal. 17)
            self.color_transform = A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            )
        else:
            self.geom_transform = None
            self.color_transform = None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"

        # Baca citra RGB
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Gagal membaca citra: {img_path}")
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Baca masker biner (grayscale)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Gagal membaca masker: {mask_path}")

        # Terapkan transformasi jika mode train
        if self.geom_transform is not None:
            augmented = self.geom_transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if self.color_transform is not None:
            image = self.color_transform(image=image)["image"]

        # Normalisasi Citra ke Standar ImageNet [0, 1] -> (x - mean) / std
        image_norm = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).float()  # (3, H, W)

        # Normalisasi Masker ke Biner [0.0, 1.0] tegas
        mask_bin = (mask > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0).float()  # (1, H, W)

        return image_tensor, mask_tensor


# -----------------------------------------------------------------------------
# 2. Fungsi Metrik Evaluasi Piksel (IoU & Dice Score)
# -----------------------------------------------------------------------------
def compute_batch_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> tuple[float, float]:
    """Menghitung rata-rata IoU (Jaccard) dan Dice Coefficient per citra dalam batch."""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        # Hitung irisan dan gabungan per elemen dalam batch (dim=(1, 2, 3))
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        total_pixels = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        # Epsilon 1e-6 untuk mencegah pembagian dengan nol
        iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
        dice = ((2.0 * intersection + 1e-6) / (total_pixels + 1e-6)).mean().item()

    return iou, dice


# -----------------------------------------------------------------------------
# 3. Prosedur Pelatihan 1 Varian (Raw atau Gray World)
# -----------------------------------------------------------------------------
def train_single_variant(
    variant: str,
    base_data_dir: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    num_workers: int = 2,
) -> dict[str, Any]:
    print("\n" + "=" * 75)
    print(f"MEMULAI PELATIHAN: U-NET + MOBILENETV2 (VARIAN: {variant.upper()})")
    print(f"Perangkat         : {device}")
    print(f"Batch Size        : {batch_size} | Learning Rate: {lr}")
    print(f"Maksimal Epoch    : {epochs} | Early Stopping Patience: {patience}")
    print("=" * 75)

    # 1. Siapkan folder tujuan
    weights_dir = output_dir / variant / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = output_dir / variant / "results.csv"
    plot_path = output_dir / variant / "training_curves.png"

    # 2. Siapkan DataLoader
    train_img_dir = base_data_dir / variant / "images" / "train"
    train_mask_dir = base_data_dir / variant / "masks" / "train"
    val_img_dir = base_data_dir / variant / "images" / "val"
    val_mask_dir = base_data_dir / variant / "masks" / "val"

    train_dataset = ExcretaCropDataset(train_img_dir, train_mask_dir, is_train=True)
    val_dataset = ExcretaCropDataset(val_img_dir, val_mask_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Jumlah sampel Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # 3. Inisialisasi Model U-Net MobileNetV2
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    # 4. Fungsi Loss: Gabungan BCE + Dice Loss (Modul Hal. 21)
    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = smp.losses.DiceLoss(mode="binary", from_logits=True)

    # 5. Optimizer AdamW & Penjadwal Cosine Annealing (Modul Hal. 22)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Inisialisasi Pencatat Riwayat
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
        "lr": [],
    }

    best_val_iou = 0.0
    best_epoch = 0
    patience_counter = 0

    # Buat header CSV
    with open(results_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "lr"])

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)

            loss = bce_criterion(logits, masks) + dice_criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)

        # ---------------------------------------------------------------------
        # Validation Loop
        # ---------------------------------------------------------------------
        model.eval()
        val_running_loss = 0.0
        val_running_iou = 0.0
        val_running_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                logits = model(images)
                loss = bce_criterion(logits, masks) + dice_criterion(logits, masks)

                val_running_loss += loss.item() * images.size(0)
                batch_iou, batch_dice = compute_batch_metrics(logits, masks)
                val_running_iou += batch_iou * images.size(0)
                val_running_dice += batch_dice * images.size(0)

        val_loss = val_running_loss / len(val_dataset)
        val_iou = val_running_iou / len(val_dataset)
        val_dice = val_running_dice / len(val_dataset)
        current_lr = scheduler.get_last_lr()[0]

        # Catat ke history dan CSV
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)
        history["lr"].append(current_lr)

        with open(results_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_iou:.4f}", f"{val_dice:.4f}", f"{current_lr:.6f}"])

        # Cetak kemajuan epoch
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou*100:.2f}% | "
            f"Val Dice: {val_dice*100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Cek Model Terbaik
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_iou": val_iou,
                    "val_dice": val_dice,
                    "variant": variant,
                },
                weights_dir / "best.pth",
            )
            print(f"  -> ⭐ [BEST MODEL DISIMPAN] Val IoU meningkat: {best_val_iou*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Tidak ada peningkatan IoU selama {patience} epoch berturut-turut.")
                break

    # Simpan bobot epoch terakhir
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_iou": val_iou,
            "val_dice": val_dice,
            "variant": variant,
        },
        weights_dir / "last.pth",
    )

    # 6. Gambar Grafik Pelatihan
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss", color="darkorange")
    plt.title(f"Kurva Kerugian (Loss) - {variant.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], [x * 100 for x in history["val_iou"]], label="Val IoU (%)", color="forestgreen")
    plt.plot(history["epoch"], [x * 100 for x in history["val_dice"]], label="Val Dice (%)", color="crimson")
    plt.axhline(y=85.0, color="gray", linestyle="--", label="Target Dosen (85%)")
    plt.title(f"Kurva Performa (IoU & Dice) - {variant.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("Skor (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=200)
    plt.close()

    print(f"-> Grafik kurva pelatihan disimpan di: {plot_path}")
    print(f"-> Bobot terbaik ({best_epoch}): {weights_dir / 'best.pth'} (IoU: {best_val_iou*100:.2f}%)")

    return {
        "variant": variant,
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "best_val_dice": history["val_dice"][best_epoch - 1] if best_epoch > 0 else 0.0,
    }


# -----------------------------------------------------------------------------
# 4. Fungsi Utama Pengendali (Orkestrator Single/Both Run)
# -----------------------------------------------------------------------------
def run_training_pipeline(
    variant: str = "both",
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 12,
) -> None:
    base_data_dir = ROOT / "dataset" / "cropped"
    output_dir = ROOT / "runs_unet"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2 if device.type == "cuda" else 0

    variants_to_train = ["raw", "gray_world"] if variant == "both" else [variant]

    summary_results = []

    for var in variants_to_train:
        # Jalankan pelatihan untuk varian yang dipilih
        res = train_single_variant(
            variant=var,
            base_data_dir=base_data_dir,
            output_dir=output_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            num_workers=num_workers,
        )
        summary_results.append(res)

        # Bersihkan cache VRAM GPU sebelum varian berikutnya dimulai
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Tampilkan dan simpan rekap hasil perbandingan
    print("\n" + "=" * 75)
    print("RINGKASAN HASIL AKHIR PELATIHAN U-NET MOBILENETV2 (E3 vs E4)")
    print("=" * 75)
    print(f"{'Eksperimen':<12} | {'Varian':<12} | {'Best Epoch':<10} | {'Val IoU':<12} | {'Val Dice':<12} | {'Target >=85%'}")
    print("-" * 75)

    summary_csv_path = output_dir / "comparison_summary.csv"
    with open(summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "variant", "best_epoch", "best_val_iou", "best_val_dice", "target_achieved"])

        for res in summary_results:
            exp_code = "E3" if res["variant"] == "raw" else "E4"
            iou_pct = f"{res['best_val_iou']*100:.2f}%"
            dice_pct = f"{res['best_val_dice']*100:.2f}%"
            achieved = "TERCAPAI (Lolos)" if res["best_val_iou"] >= 0.85 else "Belum Tercapai"

            print(f"{exp_code:<12} | {res['variant']:<12} | {res['best_epoch']:<10} | {iou_pct:<12} | {dice_pct:<12} | {achieved}")
            writer.writerow([exp_code, res["variant"], res["best_epoch"], f"{res['best_val_iou']:.4f}", f"{res['best_val_dice']:.4f}", achieved])

    print("=" * 75)
    print(f"Rekap ringkasan tersimpan di: {summary_csv_path}")
    print(f"Seluruh artefak tersimpan di: {output_dir}")
    print("=" * 75)


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pelatihan U-Net MobileNetV2 Ekskreta Ayam")
    parser.add_argument(
        "--variant",
        type=str,
        default="both",
        choices=["raw", "gray_world", "both"],
        help="Varian data masukan yang dilatih: 'raw', 'gray_world', atau 'both' (default: both)",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Maksimal jumlah epoch (default: 60)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size pelatihan (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate awal (default: 0.001)")
    parser.add_argument("--patience", type=int, default=12, help="Patience early stopping (default: 12)")
    args = parser.parse_args()

    run_training_pipeline(
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
