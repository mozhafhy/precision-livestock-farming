"""Proses training yang sebenarnya. Jangan dijalankan langsung: pakai `python -m scripts.train`,
yang menjalankan modul ini sebagai child process dan mengurus log konsol, telemetri GPU, dan laporan crash.

Catatan yang ditulis modul ini ke <log-dir>:
    events.jsonl   awal/akhir training, lingkungan torch, metrik+loss+memori tiap epoch, exception beserta traceback
    batches.jsonl  SATU baris per batch SEBELUM batch diproses: epoch, nomor langkah, nama file citra, memori CUDA.
                   Baris terakhir = tersangka utama saat crash (error CUDA bersifat asinkron: cek beberapa baris terakhir).
    fault.log      dump traceback Python dari faulthandler bila proses mati mendadak (segfault/access violation).
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO


class TrainerLike(Protocol):
    """Subset atribut BaseTrainer ultralytics yang dipakai di sini (menghindari ketergantungan tipe pada ultralytics)."""

    epoch: int
    epochs: int
    batch_size: int
    save_dir: Path
    train_loader: Any
    test_loader: Any
    preprocess_batch: Callable[[Any], Any]
    tloss: Any
    metrics: Mapping[str, Any]
    lr: Mapping[str, Any]
    best_fitness: Any
    best: Any

    def label_loss_items(
        self, loss_items: Any = None, prefix: str = "train"
    ) -> Mapping[str, Any]: ...


TrainerCallback = Callable[[TrainerLike], None]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Training detektor ROI YOLOv8n (Tahap 3) dengan logging."
    )
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
        help="early stopping pada metrik val (pilihan implementasi)",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="catat nilai ini (panduan 10.2)"
    )
    ap.add_argument("--name", default="train")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="jumlah worker dataloader; 0 untuk mengisolasi masalah dataloader",
    )
    ap.add_argument("--device", default=None, help="mis. 0 atau cpu")
    ap.add_argument(
        "--no-amp",
        action="store_true",
        help="matikan mixed precision (untuk mengisolasi sumber error)",
    )
    ap.add_argument(
        "--resume",
        default=None,
        help="path last.pt dari run yang terhenti; parameter training lain diabaikan",
    )
    ap.add_argument(
        "--debug-cuda",
        action="store_true",
        help="CUDA_LAUNCH_BLOCKING=1: traceback menunjuk operasi yang benar-benar gagal (jauh lebih lambat)",
    )
    ap.add_argument("--log-root", default=None, help="default: <folder data.yaml>/logs")
    ap.add_argument(
        "--gpu-interval", type=float, default=10.0, help="detik antar sampel nvidia-smi"
    )
    ap.add_argument(
        "--stall-minutes",
        type=float,
        default=10.0,
        help="peringatan bila tak ada batch baru selama N menit",
    )
    ap.add_argument("--log-dir", default=None, help="diisi otomatis oleh scripts.train")
    ap.add_argument(
        "--worker-module", default="scripts.train_worker", help=argparse.SUPPRESS
    )
    return ap


class JsonlLog:
    """Satu objek JSON per baris, di-flush tiap tulis agar selamat dari proses yang mati mendadak."""

    def __init__(self, path: Path) -> None:
        with open(path, "a", encoding="utf-8", buffering=1) as f:
            self.f: TextIO = f

    def write(self, **kw: Any) -> None:
        rec: dict[str, Any] = {
            "t": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ts": round(time.time(), 3),
            **kw,
        }
        self.f.write(json.dumps(rec, default=str) + "\n")
        self.f.flush()


def cuda_mem_mb() -> dict[str, float] | None:
    """Memori CUDA milik proses ini (tanpa sinkronisasi). None bila tak tersedia/gagal."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        mb = 1024**2
        return {
            "alloc_mb": round(torch.cuda.memory_allocated() / mb, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / mb, 1),
            "peak_mb": round(torch.cuda.max_memory_allocated() / mb, 1),
        }
    except Exception:
        return None


def _floats(d: Mapping[Any, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in dict(d).items():
        try:
            out[str(k)] = round(float(v), 6)
        except (TypeError, ValueError):
            pass
    return out


@dataclass
class _State:
    step: int = 0
    t0: float = field(default_factory=time.time)


def make_callbacks(events: JsonlLog, batches: JsonlLog) -> dict[str, TrainerCallback]:
    state = _State()

    def on_train_start(trainer: TrainerLike) -> None:
        info: dict[str, int] = {}
        try:
            info["n_train_images"] = len(trainer.train_loader.dataset)
            info["n_val_images"] = len(trainer.test_loader.dataset)
        except Exception:
            pass
        events.write(
            type="train_start",
            save_dir=str(trainer.save_dir),
            epochs=trainer.epochs,
            batch_size=trainer.batch_size,
            **info,
            **(cuda_mem_mb() or {}),
        )

        orig = trainer.preprocess_batch

        def logged_preprocess(batch: Any) -> Any:
            # ditulis SEBELUM batch menyentuh GPU; untuk mosaic, im_file = citra utama tiap sampel
            state.step += 1
            files = batch.get("im_file") if isinstance(batch, dict) else None
            batches.write(
                type="batch",
                epoch=trainer.epoch + 1,
                step=state.step,
                files=[str(f).replace("\\", "/").rsplit("/", 1)[-1] for f in files]
                if files
                else None,
                **(cuda_mem_mb() or {}),
            )
            return orig(batch)

        trainer.preprocess_batch = logged_preprocess

    def on_fit_epoch_end(trainer: TrainerLike) -> None:
        try:
            loss = _floats(trainer.label_loss_items(trainer.tloss))
        except Exception:
            loss = {}
        events.write(
            type="epoch_end",
            epoch=trainer.epoch + 1,
            of=trainer.epochs,
            loss=loss,
            nan_loss=any(math.isnan(v) or math.isinf(v) for v in loss.values()),
            metrics=_floats(getattr(trainer, "metrics", {}) or {}),
            lr=_floats(getattr(trainer, "lr", {}) or {}),
            best_fitness=_floats({"f": getattr(trainer, "best_fitness", None)}).get(
                "f"
            ),
            elapsed_min=round((time.time() - state.t0) / 60, 2),
            **(cuda_mem_mb() or {}),
        )

    def on_train_end(trainer: TrainerLike) -> None:
        events.write(
            type="train_end",
            best=str(getattr(trainer, "best", "")),
            **(cuda_mem_mb() or {}),
        )

    return {
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }


def main() -> None:
    args = build_parser().parse_args()
    if not args.log_dir:
        sys.exit("Jalankan lewat: python -m scripts.train")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "fault.log", "a", encoding="utf-8") as f:
        fault_file: TextIO = f  # tetap terbuka selama proses hidup

    faulthandler.enable(file=fault_file, all_threads=True)
    events = JsonlLog(log_dir / "events.jsonl")
    batches = JsonlLog(log_dir / "batches.jsonl")
    events.write(
        type="worker_start",
        pid=os.getpid(),
        argv=sys.argv,
        cuda_launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING"),
    )

    try:
        import torch
        import torchvision
        import ultralytics
        from ultralytics import YOLO

        env: dict[str, Any] = {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "ultralytics": ultralytics.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "python": sys.version.split()[0],
        }
        if torch.cuda.is_available():
            env["devices"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "capability": ".".join(
                        map(str, torch.cuda.get_device_capability(i))
                    ),
                    "total_mb": round(
                        torch.cuda.get_device_properties(i).total_memory / 1024**2
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
        events.write(type="torch_env", **env)
        (log_dir / "torch_config.txt").write_text(
            torch.__config__.show(), encoding="utf-8"
        )

        kwargs: dict[str, Any]
        if args.resume:
            model = YOLO(args.resume)
            kwargs = {"resume": True}
        else:
            data = Path(args.data).resolve()
            model = YOLO("yolov8n.pt")  # bobot pralatih COCO
            kwargs = {
                "data": str(data),
                "imgsz": args.imgsz,
                "epochs": args.epochs,
                "batch": args.batch,
                "patience": args.patience,
                "seed": args.seed,
                "deterministic": True,
                "project": str(
                    data.parent / "runs"
                ),  # path absolut: hindari ultralytics menyisipkan runs/detect/
                "name": args.name,
                "plots": True,
                "amp": not args.no_amp,
            }
            if args.workers is not None:
                kwargs["workers"] = args.workers
            if args.device:
                kwargs["device"] = args.device
        events.write(type="train_kwargs", **{k: str(v) for k, v in kwargs.items()})

        for event, fn in make_callbacks(events, batches).items():
            model.add_callback(event, fn)
        model.train(**kwargs)

        best = Path(model.trainer.best)
        print(f"\nBobot terbaik: {best}")
        print(
            f"Lanjut: python -m scripts.calibrate_detector --weights {best} --imgsz {args.imgsz}"
        )
    except BaseException as e:  # termasuk KeyboardInterrupt; catat lalu lempar ulang agar exit code tetap non-nol
        events.write(
            type="exception",
            error=repr(e),
            traceback=traceback.format_exc(),
            **(cuda_mem_mb() or {}),
        )
        raise
    events.write(type="done")


if __name__ == "__main__":
    main()
