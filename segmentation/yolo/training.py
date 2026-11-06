"""Training dan validasi YOLOv8-seg (Ultralytics) dengan log per tahapan."""

from __future__ import annotations

from pathlib import Path

from .config import Config, run_name
from .logger import get_logger, stage

NAN = float("nan")


def stage_callbacks(log) -> dict:
    """Callback Ultralytics yang menulis tahapan training ke logger kita."""

    def on_pretrain_routine_start(trainer):
        log.info("Menyiapkan model dan dataloader")

    def on_train_start(trainer):
        log.info(
            "Training dimulai | epochs=%d batch=%s imgsz=%s device=%s",
            trainer.epochs,
            trainer.args.batch,
            trainer.args.imgsz,
            trainer.device,
        )

    def on_fit_epoch_end(trainer):
        # Callback ini juga terpanggil sekali saat evaluasi final best.pt (epoch > total).
        is_final_eval = trainer.epoch + 1 > trainer.epochs
        label = (
            "Evaluasi final (best.pt)"
            if is_final_eval
            else f"Epoch {trainer.epoch + 1}/{trainer.epochs}"
        )
        metrics = trainer.metrics
        log.info(
            "%s | mask mAP50=%.4f mAP50-95=%.4f | val seg_loss=%.4f",
            label,
            metrics.get("metrics/mAP50(M)", NAN),
            metrics.get("metrics/mAP50-95(M)", NAN),
            metrics.get("val/seg_loss", NAN),
        )

    def on_train_end(trainer):
        log.info("Training selesai | bobot terbaik: %s", trainer.best)

    return {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }


def data_yaml_path(cfg: Config, variant: str, mode: str) -> Path:
    path = cfg.dataset_dir(variant, mode) / "data.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ada. Jalankan build untuk {variant}/{mode} lebih dulu."
        )
    return path


def train(
    cfg: Config,
    variant: str,
    mode: str,
    *,
    weights: str = "yolov8n-seg.pt",
    epochs: int = 100,
    batch: int = 16,
    patience: int = 30,
    seed: int = 42,
    device=None,
    name: str | None = None,
    log=None,
    **overrides,
) -> Path:
    """Latih satu kombinasi. Mengembalikan path best.pt.

    name: nama folder run (default '<variant>_<mode>'). Gunakan nama lain untuk uji cepat
    agar run penuh tidak tertimpa. **overrides diteruskan ke model.train().
    """
    from ultralytics import YOLO

    log = log or get_logger()
    name = name or run_name(variant, mode)
    data = data_yaml_path(cfg, variant, mode)
    if (cfg.runs_dir / name).exists():
        log.warning("Folder run %s sudah ada dan akan ditimpa", cfg.runs_dir / name)

    with stage(log, f"Training {name}"):
        model = YOLO(weights)
        for event, callback in stage_callbacks(log).items():
            model.add_callback(event, callback)
        model.train(
            data=str(data),
            imgsz=cfg.size,
            epochs=epochs,
            batch=batch,
            patience=patience,
            seed=seed,
            device=device,
            project=str(cfg.runs_dir),
            name=name,
            exist_ok=True,
            **overrides,
        )
    return Path(model.trainer.best)


def validate(
    cfg: Config, weights: Path, variant: str, mode: str, split: str = "test", log=None
) -> dict:
    """Metrik bawaan Ultralytics (mAP instance) pada satu split."""
    from ultralytics import YOLO

    log = log or get_logger()
    name = f"{run_name(variant, mode)}_eval_{split}"
    with stage(log, f"Validasi {name}"):
        metrics = YOLO(str(weights)).val(
            data=str(data_yaml_path(cfg, variant, mode)),
            split=split,
            imgsz=cfg.size,
            project=str(cfg.runs_dir),
            name=name,
            exist_ok=True,
            plots=False,
            verbose=False,
        )
    return {
        "mask_map50": float(metrics.seg.map50),
        "mask_map50_95": float(metrics.seg.map),
        "box_map50": float(metrics.box.map50),
    }
