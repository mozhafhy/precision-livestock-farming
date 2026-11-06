"""CLI.

Contoh (jalankan dari folder proyek yang berisi dataset/ dan yolo_seg/):
    python -m yolo_seg.run build --variant all --mode all
    python -m yolo_seg.run train --variant raw --mode mask_based --epochs 100
    python -m yolo_seg.run eval  --variant all --mode all --split test
    python -m yolo_seg.run trace --variant gray_world --mode bbox_based \
        --key test/KGL_0002 --predict --out trace.png
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

from .config import MODES, VARIANTS, Config
from .dataset_builder import build_dataset
from .history import describe, get_record, load_manifest, trace
from .logger import get_logger


def selected(value: str, allowed: tuple) -> tuple:
    return allowed if value == "all" else (value,)


def combos(args) -> itertools.product:
    return itertools.product(
        selected(args.variant, VARIANTS), selected(args.mode, MODES)
    )


# --- perintah -------------------------------------------------------------
def cmd_build(cfg, args, log) -> None:
    for variant, mode in combos(args):
        build_dataset(cfg, variant, mode, log=log)


def cmd_train(cfg, args, log) -> None:
    from .training import train

    for variant, mode in combos(args):
        train(
            cfg,
            variant,
            mode,
            weights=args.weights,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
            device=args.device,
            log=log,
        )


def cmd_eval(cfg, args, log) -> None:
    from .evaluation import evaluate_semantic
    from .training import validate

    rows = []
    for variant, mode in combos(args):
        weights = cfg.best_weights(variant, mode)
        row = {
            "variant": variant,
            "mode": mode,
            "split": args.split,
            **validate(cfg, weights, variant, mode, args.split, log=log),
            **evaluate_semantic(cfg, weights, variant, mode, args.split, log=log),
        }
        log.info("Hasil: %s", row)
        rows.append(row)

    out = cfg.runs_dir / f"eval_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log.info("Ringkasan disimpan: %s", out)


def cmd_trace(cfg, args, log) -> None:
    manifest = load_manifest(cfg.dataset_dir(args.variant, args.mode))
    record = get_record(manifest, args.key)

    pred_mask = None
    if args.predict:
        from ultralytics import YOLO

        from .evaluation import predict_mask

        pred_mask = predict_mask(
            YOLO(str(cfg.best_weights(args.variant, args.mode))), cfg, record
        )

    trace(cfg, record, pred_mask=pred_mask, save_to=args.out)
    print("\n".join(describe(record)))
    if args.out:
        log.info("Trace disimpan: %s", args.out)


COMMANDS = {
    "build": cmd_build,
    "train": cmd_train,
    "eval": cmd_eval,
    "trace": cmd_trace,
}


# --- argumen --------------------------------------------------------------
def add_combo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", default="all", choices=("all", *VARIANTS))
    parser.add_argument("--mode", default="all", choices=("all", *MODES))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="yolo_seg", description="Pipeline YOLOv8-seg")
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="folder proyek (berisi dataset/)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add_combo_args(sub.add_parser("build", help="COCO -> dataset YOLO-seg + manifest"))

    p_train = sub.add_parser("train", help="latih model")
    add_combo_args(p_train)
    p_train.add_argument("--weights", default="yolov8n-seg.pt")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--patience", type=int, default=30)
    p_train.add_argument("--device", default=None)

    p_eval = sub.add_parser(
        "eval", help="mAP Ultralytics + IoU/Dice semantik di ruang citra asli"
    )
    add_combo_args(p_eval)
    p_eval.add_argument("--split", default="test", choices=("val", "test"))

    p_trace = sub.add_parser("trace", help="visualisasi riwayat satu sampel")
    p_trace.add_argument("--variant", default="raw", choices=VARIANTS)
    p_trace.add_argument("--mode", default="mask_based", choices=MODES)
    p_trace.add_argument("--key", required=True, help="mis. test/KGL_0002")
    p_trace.add_argument(
        "--predict", action="store_true", help="sertakan prediksi model"
    )
    p_trace.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = Config(root=args.root)
    log = get_logger(
        log_file=cfg.logs_dir / f"{args.command}_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    COMMANDS[args.command](cfg, args, log)


if __name__ == "__main__":
    main()
