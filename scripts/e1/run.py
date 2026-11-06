"""E1: latih -> pilih epoch (+ ambang YOLO) di validasi -> prediksi uji satu kali -> metrik per citra.

Pemakaian:
    python run_e1.py --model unet-mobilenetv2 --seed 0
    python run_e1.py --all            # 3 model x seed di protocol.yaml
Prasyarat: python build_datasets.py --config C
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ...utils.file_managements import load_img
from .build_datasets import data_dir
from .common import (
    COCO_JSON,
    CONFIGS,
    E1,
    MODELS,
    PROTOCOL_YAML,
    ROI_MARGIN,
    git_info,
    input_size,
    load_protocol,
    run_id,
    set_seed,
    sha256_of,
)
from .metrics import METRIC_COLS, image_metrics, summarize
from .preprocessing import CocoIndex, Transform, invert_to_original, load_rgb

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Nama kunci mengikuti Ultralytics; U-Net memakai augmentasi yang setara (lihat augment()).
AUG_KEYS = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}
UNSUPPORTED_FOR_UNET = ("shear", "perspective", "mosaic", "mixup", "copy_paste")


# =========================================================================== data
@dataclass
class EvalItem:
    stem: str
    file_name: str
    x: np.ndarray  # citra ternormalisasi S x S RGB uint8
    tf: Transform
    gt: np.ndarray  # masker GT pada resolusi asli (bool)


def load_items(split: str, cfg: str) -> list[EvalItem]:
    d = data_dir(cfg)
    meta = json.loads((d / "meta" / f"{split}.json").read_text())
    coco = CocoIndex(COCO_JSON)
    items = []
    for stem, m in meta.items():
        gt = coco.union_mask(coco.image(m["file_name"]))
        items.append(
            EvalItem(
                stem,
                m["file_name"],
                load_rgb(d / "images" / split / f"{stem}.png"),
                Transform.from_dict(m["transform"]),
                gt,
            )
        )
    return items


def norm_aug(proto: dict) -> dict:
    raw = proto.get("augmentation", {}) or {}
    unknown = set(raw) - set(AUG_KEYS)
    if unknown:
        raise ValueError(
            f"Kunci augmentasi tidak dikenal: {unknown}. Petakan ke {sorted(AUG_KEYS)}."
        )
    a = {**AUG_KEYS, **raw}
    return a


def augment(x: np.ndarray, y: np.ndarray, a: dict):
    """Augmentasi U-Net setara YOLO: HSV, flip, rotasi/skala/translasi (bingkai 114 / 0)."""
    s = x.shape[0]
    if any(a[k] for k in ("hsv_h", "hsv_s", "hsv_v")):
        r = np.random.uniform(-1, 1, 3) * [a["hsv_h"], a["hsv_s"], a["hsv_v"]] + 1
        h, sat, v = cv2.split(cv2.cvtColor(x, cv2.COLOR_RGB2HSV))
        i = np.arange(256, dtype=r.dtype)
        lut_h = ((i * r[0]) % 180).astype(np.uint8)
        lut_s = np.clip(i * r[1], 0, 255).astype(np.uint8)
        lut_v = np.clip(i * r[2], 0, 255).astype(np.uint8)
        x = cv2.cvtColor(
            cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(v, lut_v))),
            cv2.COLOR_HSV2RGB,
        )
    if np.random.rand() < a["fliplr"]:
        x, y = x[:, ::-1], y[:, ::-1]
    if np.random.rand() < a["flipud"]:
        x, y = x[::-1], y[::-1]
    if a["degrees"] or a["scale"] or a["translate"]:
        m = cv2.getRotationMatrix2D(
            (s / 2, s / 2),
            np.random.uniform(-a["degrees"], a["degrees"]),
            np.random.uniform(1 - a["scale"], 1 + a["scale"]),
        )
        m[:, 2] += np.random.uniform(-a["translate"], a["translate"], 2) * s
        x = cv2.warpAffine(
            np.ascontiguousarray(x),
            m,
            (s, s),
            flags=cv2.INTER_LINEAR,
            borderValue=(114, 114, 114),
        )
        y = cv2.warpAffine(
            np.ascontiguousarray(y), m, (s, s), flags=cv2.INTER_NEAREST, borderValue=0
        )
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(
        ((x.astype(np.float32) / 255 - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1)
    ).float()


class SegDataset(Dataset):
    def __init__(self, split, cfg, aug, train):
        d = data_dir(cfg)
        self.stems = list(json.loads((d / "meta" / f"{split}.json").read_text()))
        self.img_dir, self.mask_dir = d / "images" / split, d / "masks" / split
        self.aug, self.train = aug, train

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        s = self.stems[i]
        x = load_rgb(self.img_dir / f"{s}.png")
        y = (load_img((self.mask_dir / f"{s}.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(
            np.uint8
        )
        if self.train:
            x, y = augment(x, y, self.aug)
        return to_tensor(x), torch.from_numpy(y).float().unsqueeze(0)


def _worker_init(_):
    np.random.seed(torch.initial_seed() % 2**32)


# =========================================================================== evaluasi umum
def score(items: list[EvalItem], masks: list[np.ndarray]) -> pd.DataFrame:
    rows = [
        {"image": it.stem, "file_name": it.file_name, **image_metrics(m, it.gt)}
        for it, m in zip(items, masks)
    ]
    return pd.DataFrame(rows)


# =========================================================================== U-Net
@torch.no_grad()
def unet_probs(model, items, device, bs=32) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(items), bs):
        x = torch.stack([to_tensor(it.x) for it in items[i : i + bs]]).to(device)
        out.append(torch.sigmoid(model(x)).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def unet_masks(probs, items, thr):
    # peta probabilitas dibalik ke resolusi asli dulu, lalu diberi ambang
    return [invert_to_original(p, it.tf) >= thr for p, it in zip(probs, items)]


def make_optimizer(name: str, params, lr: float):
    n = name.lower()
    if n == "adam":
        return torch.optim.Adam(params, lr=lr)
    if n == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if n == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.937, nesterov=True)
    raise ValueError(f"Optimizer tidak dikenal: {name}")


def fit_unet(
    run_dir: Path, seed: int, proto: dict, aug: dict, cfg: str, val_items, device
):
    import segmentation_models_pytorch as smp

    for k in UNSUPPORTED_FOR_UNET:
        if aug[k]:
            raise NotImplementedError(
                f"Augmentasi '{k}' belum didukung di jalur U-Net; protokol harus sama untuk semua model"
            )
    t, thr = proto["training"], proto["inference"]["unet_threshold"]
    ds = SegDataset("train", cfg, aug, train=True)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(
        ds,
        batch_size=t["batch_size"],
        shuffle=True,
        num_workers=4,
        generator=g,
        worker_init_fn=_worker_init,
        drop_last=(len(ds) % t["batch_size"] == 1),
    )

    model = smp.Unet(
        "mobilenet_v2", encoder_weights="imagenet", in_channels=3, classes=1
    ).to(device)
    opt = make_optimizer(t["optimizer"], model.parameters(), t["learning_rate"])
    bce, dice = nn.BCEWithLogitsLoss(), smp.losses.DiceLoss("binary", from_logits=True)

    best, best_ep, bad, log = -1.0, 0, 0, []
    for ep in range(1, t["max_epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = bce(logits, y) + dice(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        df = score(
            val_items, unet_masks(unet_probs(model, val_items, device), val_items, thr)
        )
        row = {
            "epoch": ep,
            "train_loss": tot / n,
            **{f"val_{c}": df[c].mean() for c in METRIC_COLS},
        }
        log.append(row)
        pd.DataFrame(log).to_csv(run_dir / "train_log.csv", index=False)
        if row["val_miou"] > best:
            best, best_ep, bad = row["val_miou"], ep, 0
            torch.save(model.state_dict(), run_dir / "best_val.pt")
        else:
            bad += 1
            if bad >= t["patience"]:
                break
        print(
            f"[unet s{seed}] ep{ep} loss={row['train_loss']:.4f} val_mIoU={row['val_miou']:.4f}"
        )

    model.load_state_dict(torch.load(run_dir / "best_val.pt", map_location=device))
    info = {"best_epoch": best_ep, "val_miou": best, "unet_threshold": thr}
    predictor = lambda items: unet_masks(unet_probs(model, items, device), items, thr)
    return info, predictor


# =========================================================================== YOLOv8-Seg
def yolo_instances(model, items, conf_floor, size, device, bs=32):
    """Prediksi sekali pada ambang terendah; ambang lebih tinggi cukup memfilter skor
    (ekuivalen dengan prediksi ulang karena NMS mengurutkan berdasarkan skor)."""
    out = []
    for i in range(0, len(items), bs):
        batch = [cv2.cvtColor(it.x, cv2.COLOR_RGB2BGR) for it in items[i : i + bs]]
        for r in model.predict(
            batch, imgsz=size, conf=conf_floor, device=device, verbose=False
        ):
            if r.masks is None or len(r.masks) == 0:
                out.append((np.zeros((0, size, size), bool), np.zeros(0)))
                continue
            m = r.masks.data.cpu().numpy() > 0.5
            if m.shape[1:] != (size, size):
                m = np.stack(
                    [
                        cv2.resize(
                            k.astype(np.uint8),
                            (size, size),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        for k in m
                    ]
                ).astype(bool)
            out.append((m, r.boxes.conf.cpu().numpy()))
    return out


def yolo_union(inst, tf: Transform, thr: float) -> np.ndarray:
    """Aturan 4: gabungkan semua masker instans dengan skor >= ambang menjadi satu masker biner."""
    m, s = inst
    keep = s >= thr
    if not keep.any():
        return np.zeros((tf.orig_h, tf.orig_w), bool)
    return invert_to_original(m[keep].any(0), tf) >= 0.5


def fit_yolo(
    run_dir: Path,
    model_name: str,
    seed: int,
    proto: dict,
    aug: dict,
    cfg: str,
    val_items,
    size,
    device,
):
    from ultralytics import YOLO

    t, grid = proto["training"], list(proto["inference"]["yolo_conf_grid"])
    opt_map = {"adam": "Adam", "adamw": "AdamW", "sgd": "SGD"}
    train_dir = run_dir / "yolo_train"
    YOLO(f"{model_name}.pt").train(
        data=str(data_dir(cfg) / "data.yaml"),
        epochs=t["max_epochs"],
        patience=t["patience"],
        imgsz=size,
        batch=t["batch_size"],
        optimizer=opt_map[t["optimizer"].lower()],
        lr0=t["learning_rate"],
        seed=seed,
        deterministic=True,
        project=str(run_dir.resolve()),
        name="yolo_train",
        exist_ok=True,
        save_period=1,
        pretrained=True,
        plots=False,
        close_mosaic=0,
        device=device,
        **aug,
    )

    ckpts = sorted(
        (int(re.search(r"epoch(\d+)", p.stem).group(1)), p)
        for p in (train_dir / "weights").glob("epoch*.pt")
    )
    if not ckpts:
        raise RuntimeError("Tidak ada checkpoint per-epoch (save_period=1)")

    sweep, log = [], []
    for ep, path in ckpts:
        inst = yolo_instances(YOLO(str(path)), val_items, min(grid), size, device)
        rows = []
        for thr in grid:
            df = score(
                val_items, [yolo_union(i, it.tf, thr) for i, it in zip(inst, val_items)]
            )
            rows.append(
                {
                    "epoch": ep,
                    "conf": thr,
                    **{f"val_{c}": df[c].mean() for c in METRIC_COLS},
                }
            )
        sweep += rows
        best_row = max(
            rows, key=lambda r: r["val_iou_exc"]
        )  # ambang: maks. IoU ekskreta validasi
        log.append({**best_row, "ckpt": path.name})
        pd.DataFrame(log).to_csv(run_dir / "train_log.csv", index=False)
    pd.DataFrame(sweep).to_csv(run_dir / "val_conf_sweep.csv", index=False)

    best = max(
        log, key=lambda r: r["val_miou"]
    )  # epoch: maks. mIoU validasi (pada ambang terbaiknya)
    best_path = train_dir / "weights" / best["ckpt"]
    shutil.copy(best_path, run_dir / "best_val.pt")
    shutil.copy(train_dir / "results.csv", run_dir / "ultralytics_results.csv")
    shutil.rmtree(train_dir / "weights")  # hemat disk

    model, conf = YOLO(str(run_dir / "best_val.pt")), best["conf"]
    info = {
        "best_epoch": best["epoch"],
        "val_miou": best["val_miou"],
        "val_iou_exc": best["val_iou_exc"],
        "yolo_conf": conf,
    }
    predictor = lambda items: [
        yolo_union(i, it.tf, conf)
        for i, it in zip(yolo_instances(model, items, conf, size, device), items)
    ]
    return info, predictor


# =========================================================================== satu run
def log_run(row: dict) -> None:
    p = E1 / "run_log.csv"
    pd.DataFrame([row]).to_csv(p, mode="a", header=not p.exists(), index=False)


def execute_run(
    model_name: str, seed: int, cfg: str = "C", device: str | None = None
) -> None:
    rid = run_id(model_name, cfg, seed)
    run_dir = E1 / rid
    if (run_dir / "metrics_summary.json").exists():
        print(f"{rid}: sudah selesai, dilewati")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    proto, git, started = (
        load_protocol(),
        git_info(),
        datetime.now().isoformat(timespec="seconds"),
    )
    size, aug = input_size(proto), norm_aug(proto)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": rid,
                "model": model_name,
                "seed": seed,
                "config": cfg,
                "preprocessing": CONFIGS[cfg],
                "roi_margin": ROI_MARGIN,
                "device": device,
                "git": git,
                "protocol_sha256": sha256_of(PROTOCOL_YAML),
                "protocol": proto,
                "started_at": started,
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )

    t0, status = time.time(), "ok"
    try:
        set_seed(seed)
        val_items = load_items("val", cfg)
        if model_name == "unet-mobilenetv2":
            info, predictor = fit_unet(
                run_dir, seed, proto, aug, cfg, val_items, device
            )
        else:
            info, predictor = fit_yolo(
                run_dir, model_name, seed, proto, aug, cfg, val_items, size, device
            )
        train_seconds = time.time() - t0

        # ---- uji: tepat satu kali per run final ----
        lock = run_dir / "TEST_DONE"
        if lock.exists():
            raise RuntimeError(f"{rid}: test set sudah pernah dipakai (aturan 1)")
        lock.write_text(datetime.now().isoformat(timespec="seconds"))
        test_items = load_items("test", cfg)
        masks = predictor(test_items)

        mask_dir = run_dir / "pred_masks"
        mask_dir.mkdir(exist_ok=True)
        for it, m in zip(test_items, masks):
            cv2.imwrite(str(mask_dir / f"{it.stem}.png"), m.astype(np.uint8) * 255)

        df = score(test_items, masks)
        df.insert(0, "model", model_name)
        df.insert(0, "seed", seed)
        df.insert(0, "run_id", rid)
        df.to_csv(run_dir / "metrics_per_image.csv", index=False)
        (run_dir / "metrics_summary.json").write_text(
            json.dumps(
                {
                    "run_id": rid,
                    "model": model_name,
                    "seed": seed,
                    "config": cfg,
                    "git": git,
                    "selection": info,
                    "test": summarize(df),
                    "n_test": len(df),
                    "n_no_prediction": int(df["no_pred"].sum()),
                    "train_seconds": round(train_seconds, 1),
                },
                indent=2,
            )
        )
        print(f"{rid}: IoU ekskreta={df['iou_exc'].mean():.4f}")
    except Exception:
        status = "failed"
        traceback.print_exc()
        raise
    finally:
        log_run(
            {
                "run_id": rid,
                "seed": seed,
                "git_commit": git["commit"],
                "git_dirty": git["dirty"],
                "started_at": started,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "status": status,
            }
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--config", default="C", choices=list(CONFIGS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.all:
        for m in MODELS:
            for s in load_protocol()["seeds"]:
                execute_run(m, s, a.config, a.device)
    else:
        if a.model is None or a.seed is None:
            ap.error("--model dan --seed wajib kecuali --all")
        execute_run(a.model, a.seed, a.config, a.device)
