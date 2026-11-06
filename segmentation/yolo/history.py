"""Riwayat transformasi per sampel: dari citra mentah sampai masker tersegmentasi.

Setiap sampel dataset punya SampleRecord yang menyimpan parameter transformasi
geometri (crop box, ukuran asal, ukuran keluaran). Karena transformasinya hanya
crop + resize, riwayat ini bisa dibalik (restore_mask) dan divisualisasikan (trace).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import rasterize, read_label_polygons

MANIFEST = "manifest.jsonl"


@dataclass(frozen=True)
class SampleRecord:
    split: str
    stem: str
    image_id: int
    variant: str
    mode: str
    raw_image: str  # citra mentah asli (relatif terhadap root)
    source_image: str  # citra varian yang dipakai (raw / gray_world)
    orig_size: tuple  # (lebar, tinggi) citra asli
    crop_box: tuple  # (x, y, w, h) di koordinat citra asli
    crop_fallback: bool  # True: bbox_based tapi tidak ada bbox valid -> citra penuh
    out_size: int
    n_polygons: int
    image: str  # citra keluaran (input model)
    label: str  # label YOLO-seg keluaran

    @property
    def key(self) -> str:
        return f"{self.split}/{self.stem}"

    @classmethod
    def from_dict(cls, data: dict) -> SampleRecord:
        data = {
            **data,
            "orig_size": tuple(data["orig_size"]),
            "crop_box": tuple(data["crop_box"]),
        }
        return cls(**data)


# --- manifest -----------------------------------------------------------
def write_manifest(path: Path, records: list[SampleRecord]) -> None:
    lines = (json.dumps(asdict(record)) for record in records)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_manifest(dataset_dir: Path) -> dict[str, SampleRecord]:
    path = Path(dataset_dir) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"{path} tidak ada. Bangun dataset lebih dulu.")
    with open(path, encoding="utf-8") as file:
        records = [
            SampleRecord.from_dict(json.loads(line)) for line in file if line.strip()
        ]
    return {record.key: record for record in records}


def get_record(manifest: dict[str, SampleRecord], key: str) -> SampleRecord:
    if key not in manifest:
        raise KeyError(f"Sampel {key!r} tidak ada. Contoh key: {list(manifest)[:3]}")
    return manifest[key]


# --- penelusuran ---------------------------------------------------------
def describe(record: SampleRecord) -> list[str]:
    """Riwayat sampel dalam bentuk teks, urut dari citra mentah ke keluaran."""
    orig_w, orig_h = record.orig_size
    x, y, w, h = record.crop_box
    size = record.out_size
    if record.mode == "mask_based":
        crop_note = "seluruh citra"
    elif record.crop_fallback:
        crop_note = "FALLBACK: tidak ada bbox valid, seluruh citra"
    else:
        crop_note = "union bbox anotasi"
    return [
        f"[{record.key}]",
        f"1. Citra mentah  : {record.raw_image} ({orig_w}x{orig_h})",
        f"2. Varian warna  : {record.variant} -> {record.source_image}",
        f"3. Crop          : x={x} y={y} w={w} h={h} ({crop_note})",
        f"4. Resize        : {w}x{h} -> {size}x{size} (skala x{size / w:.2f}, y{size / h:.2f})",
        f"5. Keluaran      : {record.image} + {record.label} ({record.n_polygons} polygon)",
        f"6. Pemetaan balik: masker {size}x{size} -> {orig_w}x{orig_h} via restore_mask()",
    ]


def restore_mask(mask: np.ndarray, record: SampleRecord) -> np.ndarray:
    """Petakan masker ruang-input-model kembali ke ruang citra asli (inversi crop + resize)."""
    x, y, w, h = record.crop_box
    orig_w, orig_h = record.orig_size
    canvas = np.zeros((orig_h, orig_w), np.uint8)
    canvas[y : y + h, x : x + w] = cv2.resize(
        mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    )
    return canvas


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Gagal membaca citra: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _overlay(
    image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.5
) -> np.ndarray:
    out = image.copy()
    selected = mask > 0
    out[selected] = ((1 - alpha) * out[selected] + alpha * np.array(color)).astype(
        np.uint8
    )
    return out


def trace(
    cfg,
    record: SampleRecord,
    pred_mask: np.ndarray | None = None,
    save_to: Path | None = None,
):
    """Visualisasi satu baris: mentah -> varian -> input model (+GT) [-> prediksi -> dipetakan balik]."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    size = record.out_size
    raw = _read_rgb(cfg.root / record.raw_image)
    variant_image = _read_rgb(cfg.root / record.source_image)
    model_input = _read_rgb(cfg.root / record.image)
    gt = rasterize(read_label_polygons(cfg.root / record.label, size), size, size)

    # (judul, citra, masker overlay, gambar kotak crop?)
    panels = [
        ("1. Mentah + kotak crop", raw, None, True),
        (f"2. Varian: {record.variant}", variant_image, None, True),
        (f"3. Input model {size}px + GT", model_input, gt, False),
    ]
    if pred_mask is not None:
        panels += [
            ("4. Prediksi (ruang input)", model_input, pred_mask, False),
            ("5. Prediksi dipetakan balik", raw, restore_mask(pred_mask, record), True),
        ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.3))
    x, y, w, h = record.crop_box
    for ax, (title, image, mask, show_box) in zip(np.atleast_1d(axes), panels):
        ax.imshow(image if mask is None else _overlay(image, mask))
        if show_box:
            ax.add_patch(
                Rectangle((x, y), w, h, fill=False, edgecolor="yellow", linewidth=2)
            )
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(record.key, fontsize=10)
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=120)
    return fig
