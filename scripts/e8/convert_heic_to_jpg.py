"""Script Konversi Citra HEIC ke JPG (Bisa dijalankan di Lokal maupun Google Drive / Colab).

Fitur:
- Otomatis mendeteksi folder input (lokal maupun Google Drive / Google Colab).
- Menangani rotasi EXIF dari kamera HP (agar orientasi foto tidak terbalik/miring).
- Menyimpan hasil JPG ke folder JPG IMG dengan kualitas tinggi (JPEG quality=95).
- Terdapat progress bar konversi.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def ensure_dependencies():
    """Memastikan pustaka pillow-heif dan tqdm terpasang."""
    try:
        import pillow_heif  # noqa: F401
    except ImportError:
        print("[Info] Menginstal pustaka pillow-heif...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pillow-heif", "tqdm"])


ensure_dependencies()

from PIL import Image, ImageOps  # noqa: E402
import pillow_heif  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

# Daftarkan format HEIF ke PIL
pillow_heif.register_heif_opener()


def resolve_paths(custom_input: str | None = None, custom_output: str | None = None) -> tuple[Path, Path]:
    """Mendeteksi lokasi folder input dan output baik di lokal maupun Google Drive."""
    script_dir = Path(__file__).resolve().parent

    # 1. Tentukan folder output (default: folder JPG IMG)
    if custom_output:
        out_dir = Path(custom_output).resolve()
    else:
        out_dir = Path("D:/Richard/Semester 5/Dataset Mitra Banyuwangi/JPG IMG")
        if not out_dir.exists():
            out_dir = Path("dataset/raw/banyuwangi")

    # 2. Tentukan folder input
    if custom_input:
        in_dir = Path(custom_input).resolve()
        return in_dir, out_dir

    candidate_inputs = [
        Path("D:/Richard/Semester 5/Dataset Mitra Banyuwangi"),
        Path("/content/drive/MyDrive/Dataset Mitra Banyuwangi"),
        Path("/content/drive/MyDrive/precision-livestock-farming/dataset/banyuwangi"),
        Path("/content/drive/MyDrive/precision-livestock-farming/dataset/raw/banyuwangi"),
        script_dir.parent,
        Path.cwd(),
    ]

    for cand in candidate_inputs:
        if cand.exists():
            heic_count = len(list(cand.glob("*.heic")) + list(cand.glob("*.HEIC")))
            if heic_count > 0:
                return cand.resolve(), out_dir

    return Path("D:/Richard/Semester 5/Dataset Mitra Banyuwangi"), out_dir


def convert_all(input_dir: Path, output_dir: Path, quality: int = 95):
    print("=" * 65)
    print("KONVERSI CITRA HEIC KE JPG (PETERNAKAN MITRA BANYUWANGI)")
    print("=" * 65)
    print(f"Folder Sumber (HEIC) : {input_dir}")
    print(f"Folder Tujuan (JPG)  : {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    heic_files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".heic"]
    )

    if not heic_files:
        print(f"\n[Peringatan] Tidak ditemukan file .HEIC di folder: {input_dir}")
        print("Silakan pastikan file HEIC berada di folder tersebut.")
        return

    print(f"\nTotal file HEIC yang akan dikonversi: {len(heic_files)} citra\n")

    success_count = 0
    fail_count = 0

    pbar = tqdm(heic_files, desc="Mengonversi", unit="citra")
    for file_p in pbar:
        out_filename = f"{file_p.stem}.jpg"
        out_p = output_dir / out_filename

        try:
            img = Image.open(file_p)
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(out_p, "JPEG", quality=quality)
            success_count += 1
        except Exception as e:
            tqdm.write(f"[Gagal] {file_p.name}: {e}")
            fail_count += 1

    print("\n" + "=" * 65)
    print(f"SELESAI: {success_count} citra berhasil dikonversi ke JPG!")
    if fail_count > 0:
        print(f"[Perhatian] {fail_count} citra gagal dikonversi.")
    print(f"Seluruh file JPG tersimpan di: {output_dir}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Konversi HEIC ke JPG")
    parser.add_argument("--input", "-i", default=None, help="Folder sumber file HEIC")
    parser.add_argument("--output", "-o", default=None, help="Folder tujuan penyimpanan file JPG")
    parser.add_argument("--quality", "-q", type=int, default=95, help="Kualitas JPEG (default: 95)")
    args = parser.parse_args()

    in_dir, out_dir = resolve_paths(args.input, args.output)
    convert_all(in_dir, out_dir, quality=args.quality)


if __name__ == "__main__":
    main()
