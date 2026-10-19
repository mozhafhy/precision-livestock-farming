"""Detektor ROI ekskreta berbasis Otsu: pipeline, ground truth, metrik, dan grid search.

Pipeline (setiap tahap dapat diambil sebagai citra untuk visualisasi):
    01_input -> 02_norm -> 03_channel -> 04_blur -> 05_otsu -> 06_morphology -> 07_roi

Satu kontur akhir diperlakukan sebagai satu deteksi (Panduan, Bagian 9.2), sehingga
metrik dapat dihitung identik dengan detektor YOLO.

Eksperimen (B1, B2, grid search) dijalankan dari notebook, bukan dari modul ini.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

sys.dont_write_bytecode = True


def get_project_root(path: Path, marker: str = ".git") -> Path:
    current = path.resolve()
    # Naik terus ke atas sampai menemukan marker atau mencapai root sistem
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    return current


ROOT = get_project_root(Path(__file__))
RAW_DIR = ROOT / "dataset" / "raw"
COCO_JSON = ROOT / "dataset" / "annotations" / "master_cleaned.json"
SPLIT_CSV = ROOT / "dataset" / "split.csv"

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2) dalam piksel

IOU_MATCH_THRESHOLD = 0.5
CROP_COVERAGE_THRESHOLD = 0.95  # GT dianggap "termuat utuh" bila >= 95% luasnya ada di dalam kotak (toleransi bbox-vs-mask)
OTSU_FOREGROUND_MAX_FRACTION = (
    0.5  # polaritas auto: objek diasumsikan menempati area lebih kecil
)


# ----------------------------------------------------------------------------
# Parameter
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class OtsuParams:
    norm_mode: str = "none"  # "none" | "gray_world"
    color_space: str = "gray"  # "gray" | "lab" | "hsv"
    channel: int = 0  # lab: 0=L,1=a,2=b; hsv: 0=H,1=S,2=V; diabaikan pada gray
    blur_ksize: int = 5  # ganjil; 0 = tanpa blur
    polarity: str = "auto"  # "auto" | "dark" | "bright"
    open_ksize: int = 0  # ganjil; 0 = tanpa opening
    close_ksize: int = 5  # ganjil; 0 = tanpa closing
    min_area_frac: float = 0.002  # luas kontur minimum (fraksi luas citra)
    max_area_frac: float = 0.95  # kontur sebesar citra dianggap latar
    padding_frac: float = 0.0  # margin kotak (fraksi sisi kotak)

    @classmethod
    def from_json(cls, path: Path) -> OtsuParams:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))["params"])


# B1: Otsu + Gray World versi awal (Panduan 9.1). Nilai ini dibekukan.
PRESET_B1 = OtsuParams(
    norm_mode="gray_world", color_space="gray", blur_ksize=5, close_ksize=5
)


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float  # luas kontur / luas citra; Otsu tidak punya confidence, ini proksi untuk AP


@dataclass
class DetectionResult:
    detections: list[Detection]
    threshold: float
    stages: dict[str, np.ndarray]


@dataclass(frozen=True)
class Sample:
    image_id: str
    image: np.ndarray
    gt_boxes: list[Box]


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def _make_odd(ksize: int) -> int:
    return ksize if ksize <= 0 or ksize % 2 == 1 else ksize + 1


def normalize_color(img_bgr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return img_bgr
    if mode == "gray_world":
        channel_means = img_bgr.reshape(-1, 3).mean(axis=0)
        gains = channel_means.mean() / (channel_means + 1e-6)
        return np.clip(img_bgr * gains, 0, 255).astype(np.uint8)
    raise ValueError(f"norm_mode tidak dikenal: {mode}")


def extract_channel(img_bgr: np.ndarray, color_space: str, channel: int) -> np.ndarray:
    if color_space == "gray":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if color_space == "lab":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., channel]
    if color_space == "hsv":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., channel]
    raise ValueError(f"color_space tidak dikenal: {color_space}")


def gaussian_blur(channel: np.ndarray, ksize: int) -> np.ndarray:
    ksize = _make_odd(ksize)
    return cv2.GaussianBlur(channel, (ksize, ksize), 0) if ksize > 0 else channel


def binarize_otsu(channel: np.ndarray, polarity: str) -> tuple[float, np.ndarray]:
    """Otsu dengan objek = putih (255)."""
    invert_flag = cv2.THRESH_BINARY_INV if polarity == "dark" else cv2.THRESH_BINARY
    threshold, binary = cv2.threshold(channel, 0, 255, invert_flag + cv2.THRESH_OTSU)
    if polarity == "auto" and (binary > 0).mean() > OTSU_FOREGROUND_MAX_FRACTION:
        binary = 255 - binary
    return float(threshold), binary


def _morphology(binary: np.ndarray, operation: int, ksize: int) -> np.ndarray:
    ksize = _make_odd(ksize)
    if ksize <= 0:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(binary, operation, kernel)


def clean_mask(binary: np.ndarray, open_ksize: int, close_ksize: int) -> np.ndarray:
    opened = _morphology(binary, cv2.MORPH_OPEN, open_ksize)
    return _morphology(opened, cv2.MORPH_CLOSE, close_ksize)


def _padded_box(
    x: int, y: int, bw: int, bh: int, pad_frac: float, shape: tuple[int, int]
) -> Box:
    height, width = shape
    pad_x, pad_y = bw * pad_frac, bh * pad_frac
    return (
        max(0.0, x - pad_x),
        max(0.0, y - pad_y),
        min(float(width), x + bw + pad_x),
        min(float(height), y + bh + pad_y),
    )


def contours_to_detections(mask: np.ndarray, p: OtsuParams) -> list[Detection]:
    height, width = mask.shape[:2]
    image_area = height * width
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (p.min_area_frac * image_area <= area <= p.max_area_frac * image_area):
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        detections.append(
            Detection(
                _padded_box(x, y, bw, bh, p.padding_frac, (height, width)),
                area / image_area,
            )
        )
    return detections


def _draw_boxes(
    img_bgr: np.ndarray, boxes: Iterable[Box], color: tuple[int, int, int]
) -> np.ndarray:
    canvas = img_bgr.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return canvas


def detect(
    img_bgr: np.ndarray, p: OtsuParams, return_stages: bool = False
) -> DetectionResult:
    normalized = normalize_color(img_bgr, p.norm_mode)
    channel = extract_channel(normalized, p.color_space, p.channel)
    blurred = gaussian_blur(channel, p.blur_ksize)
    threshold, binary = binarize_otsu(blurred, p.polarity)
    mask = clean_mask(binary, p.open_ksize, p.close_ksize)
    detections = contours_to_detections(mask, p)

    stages: dict[str, np.ndarray] = {}
    if return_stages:
        stages = {
            "01_input": img_bgr,
            "02_norm": normalized,
            "03_channel": channel,
            "04_blur": blurred,
            "05_otsu": binary,
            "06_morphology": mask,
            "07_roi": _draw_boxes(img_bgr, [d.box for d in detections], (0, 255, 0)),
        }
    return DetectionResult(detections, threshold, stages)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_ground_truth() -> dict[str, list[Box]]:
    """{image_id: [kotak per instans]} dari anotasi COCO. Citra tanpa anotasi tidak muncul."""
    coco = json.loads(COCO_JSON.read_text(encoding="utf-8"))
    stems = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    boxes: dict[str, list[Box]] = {}
    for ann in coco["annotations"]:
        x, y, bw, bh = ann["bbox"]
        boxes.setdefault(stems[ann["image_id"]], []).append((x, y, x + bw, y + bh))
    return boxes


def load_split_ids(split: str) -> list[str]:
    with open(SPLIT_CSV, encoding="utf-8") as f:
        return [row["image_id"] for row in csv.DictReader(f) if row["split"] == split]


def load_samples(split: str) -> list[Sample]:
    """Muat citra mentah + GT. Citra tanpa anotasi dipertahankan sebagai sampel negatif."""
    gt = load_ground_truth()
    samples = []
    for image_id in load_split_ids(split):
        image = cv2.imread(str(RAW_DIR / f"{image_id}.jpg"))
        if image is None:
            raise FileNotFoundError(f"Citra tidak terbaca: {image_id}")
        samples.append(Sample(image_id, image, gt.get(image_id, [])))
    return samples


# ----------------------------------------------------------------------------
# Metrik
# ----------------------------------------------------------------------------
def _intersection_area(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _area(box: Box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def iou(a: Box, b: Box) -> float:
    inter = _intersection_area(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def coverage(pred: Box, gt: Box) -> float:
    """Porsi luas GT yang berada di dalam kotak prediksi."""
    return _intersection_area(pred, gt) / _area(gt) if _area(gt) > 0 else 0.0


@dataclass(frozen=True)
class ImageOutcome:
    tp_flags: list[bool]  # per deteksi (urut skor menurun)
    scores: list[float]
    matched_ious: list[float]
    n_gt: int
    crops_ok: int  # jumlah GT yang termuat utuh pada suatu deteksi
    roi_frac: float  # total luas kotak prediksi / luas citra


def match_image(
    detections: list[Detection], gt_boxes: list[Box], image_area: float
) -> ImageOutcome:
    ranked = sorted(detections, key=lambda d: d.score, reverse=True)
    unmatched = set(range(len(gt_boxes)))
    tp_flags, matched_ious = [], []
    for det in ranked:
        best_iou, best_gt = 0.0, None
        for gi in unmatched:
            value = iou(det.box, gt_boxes[gi])
            if value > best_iou:
                best_iou, best_gt = value, gi
        is_tp = best_gt is not None and best_iou >= IOU_MATCH_THRESHOLD
        if is_tp:
            unmatched.discard(best_gt)
            matched_ious.append(best_iou)
        tp_flags.append(is_tp)
    crops_ok = sum(
        any(coverage(d.box, g) >= CROP_COVERAGE_THRESHOLD for d in ranked)
        for g in gt_boxes
    )
    roi_frac = sum(_area(d.box) for d in ranked) / image_area
    return ImageOutcome(
        tp_flags,
        [d.score for d in ranked],
        matched_ious,
        len(gt_boxes),
        crops_ok,
        roi_frac,
    )


def average_precision(scores: list[float], tp_flags: list[bool], n_gt: int) -> float:
    """AP@0.5 (interpolasi seluruh titik) memakai skor proksi."""
    if n_gt == 0 or not scores:
        return 0.0
    order = np.argsort(scores)[::-1]
    tp = np.cumsum(np.array(tp_flags)[order])
    fp = np.cumsum(~np.array(tp_flags)[order])
    recall = np.concatenate(([0.0], tp / n_gt, [1.0]))
    precision = np.concatenate(([0.0], tp / (tp + fp), [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall) * precision[1:]))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(outcomes: list[ImageOutcome]) -> dict[str, float]:
    tp = sum(sum(o.tp_flags) for o in outcomes)
    fp = sum(len(o.tp_flags) - sum(o.tp_flags) for o in outcomes)
    n_gt = sum(o.n_gt for o in outcomes)
    precision, recall = _safe_ratio(tp, tp + fp), _safe_ratio(tp, n_gt)
    ious = [v for o in outcomes for v in o.matched_ious]
    all_scores = [s for o in outcomes for s in o.scores]
    all_flags = [f for o in outcomes for f in o.tp_flags]
    n_images = len(outcomes)
    n_detections = [len(o.tp_flags) for o in outcomes]
    return {
        "n_images": n_images,
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2 * precision * recall, precision + recall),
        "map50": average_precision(all_scores, all_flags, n_gt),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "crop_success_rate": _safe_ratio(sum(o.crops_ok for o in outcomes), n_gt),
        "count_accuracy": _safe_ratio(
            sum(n == o.n_gt for n, o in zip(n_detections, outcomes)), n_images
        ),
        "no_detection_rate": _safe_ratio(sum(n == 0 for n in n_detections), n_images),
        "mean_roi_frac": float(np.mean([o.roi_frac for o in outcomes]))
        if outcomes
        else 0.0,
    }


def compute_outcomes(p: OtsuParams, samples: list[Sample]) -> list[ImageOutcome]:
    outcomes = []
    for s in samples:
        height, width = s.image.shape[:2]
        outcomes.append(
            match_image(detect(s.image, p).detections, s.gt_boxes, height * width)
        )
    return outcomes


def evaluate(p: OtsuParams, samples: list[Sample]) -> dict[str, float]:
    return summarize(compute_outcomes(p, samples))


def per_image_report(p: OtsuParams, samples: list[Sample]) -> list[dict]:
    """IoU terbaik per citra, untuk menelusuri kasus terburuk."""
    rows = []
    for s in samples:
        dets = detect(s.image, p).detections
        best = max((iou(d.box, g) for d in dets for g in s.gt_boxes), default=0.0)
        rows.append(
            {
                "image_id": s.image_id,
                "n_pred": len(dets),
                "n_gt": len(s.gt_boxes),
                "best_iou": best,
            }
        )
    return sorted(rows, key=lambda r: r["best_iou"])


# ----------------------------------------------------------------------------
# Grid search & visualisasi
# ----------------------------------------------------------------------------
def grid_search(
    samples: list[Sample], grid: dict[str, list], base: OtsuParams | None = None
) -> list[tuple[OtsuParams, dict[str, float]]]:
    """Sapu seluruh kombinasi `grid`; urut menurun menurut (f1, mean_iou). Gunakan HANYA subset val."""
    if base is None:
        base = OtsuParams()
    
    keys = list(grid)
    results = []
    for values in itertools.product(*grid.values()):
        params = replace(base, **dict(zip(keys, values)))
        results.append((params, evaluate(params, samples)))
    results.sort(key=lambda r: (r[1]["f1"], r[1]["mean_iou"]), reverse=True)
    return results


def select_best(
    results: list[tuple[OtsuParams, dict[str, float]]], min_crop_success: float = 0.0
) -> tuple[OtsuParams, dict[str, float]]:
    """Konfigurasi terbaik (urutan grid_search) dengan crop_success_rate >= ambang.

    Tetapkan ambang dari subset val SEBELUM test dibuka; F1 saja mengabaikan keutuhan potongan.
    """
    for params, metrics in results:
        if metrics["crop_success_rate"] >= min_crop_success:
            return params, metrics
    raise ValueError(
        f"Tidak ada konfigurasi dengan crop_success_rate >= {min_crop_success}"
    )


# ----------------------------------------------------------------------------
# Bootstrap per citra (interval kepercayaan 95%)
# ----------------------------------------------------------------------------
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
CI_PERCENTILES = (2.5, 97.5)


def _bootstrap_summaries(
    outcome_sets: list[list[ImageOutcome]],
) -> list[list[dict[str, float]]]:
    """Resampel indeks citra yang SAMA untuk semua konfigurasi, sehingga perbandingannya berpasangan."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_images = len(outcome_sets[0])
    draws: list[list[dict[str, float]]] = [[] for _ in outcome_sets]
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, n_images, n_images)
        for draw, outcomes in zip(draws, outcome_sets):
            draw.append(summarize([outcomes[i] for i in indices]))
    return draws


def _metric_names(summary: dict[str, float]) -> list[str]:
    return [name for name in summary if name != "n_images"]


def bootstrap_ci(
    p: OtsuParams, samples: list[Sample]
) -> dict[str, tuple[float, float]]:
    """{metrik: (batas bawah, batas atas)} interval 95%."""
    (draws,) = _bootstrap_summaries([compute_outcomes(p, samples)])
    return {
        name: tuple(np.percentile([d[name] for d in draws], CI_PERCENTILES))
        for name in _metric_names(draws[0])
    }


def bootstrap_paired_difference(
    p_a: OtsuParams, p_b: OtsuParams, samples: list[Sample]
) -> dict[str, tuple[float, float, float]]:
    """{metrik: (selisih b - a, batas bawah, batas atas)}. Selisih bermakna bila interval tidak memuat 0."""
    draws_a, draws_b = _bootstrap_summaries(
        [compute_outcomes(p_a, samples), compute_outcomes(p_b, samples)]
    )
    result = {}
    for name in _metric_names(draws_a[0]):
        diffs = [b[name] - a[name] for a, b in zip(draws_a, draws_b)]
        low, high = np.percentile(diffs, CI_PERCENTILES)
        result[name] = (float(np.mean(diffs)), float(low), float(high))
    return result


def freeze_params(path: Path, p: OtsuParams, metrics: dict[str, float]) -> None:
    """Bekukan konfigurasi (mis. B2) beserta metrik val-nya."""
    payload = {"params": asdict(p), "val_metrics": metrics}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _label_tile(img: np.ndarray, title: str) -> np.ndarray:
    tile = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(
        tile,
        title,
        (4, 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def make_montage(sample: Sample, p: OtsuParams, columns: int = 4) -> np.ndarray:
    """Grid seluruh tahap; kotak GT (biru) digambar pada tahap 07_roi."""
    stages = detect(sample.image, p, return_stages=True).stages
    stages["07_roi"] = _draw_boxes(stages["07_roi"], sample.gt_boxes, (255, 0, 0))
    tiles = [_label_tile(img, name) for name, img in stages.items()]
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack(
        [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    )
