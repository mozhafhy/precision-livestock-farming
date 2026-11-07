"""Convert non-JPEG images (PNG, WebP, HEIC, BMP, TIFF, ...) into JPEG.

Usage:
  python -m scripts.e8.convert <subdir> --out OUT_DIR [--src SRC] [--recursive] [--overwrite]
  python -m scripts.e8.convert --just FILENAME --out OUT_DIR [<subdir>] [--src SRC] [--recursive] [--overwrite]

Arguments:
  subdir        Subdirectory inside --src to process.
  --src PATH    Source root. Default: tmp/res
  --just NAME   Convert only the file with this exact name (case-insensitive,
                no glob, no path). Searched in --src (or --src/<subdir>).
  --recursive   Also search subdirectories. Alias: --recurs
  --overwrite   Overwrite existing output files. Default: skip them.
  --out PATH    Output directory (required). Created if missing.

Examples:
  # All images directly inside tmp/res/photos
  python convert.py photos --out out

  # Same, including nested folders (relative structure is preserved in out/)
  python convert.py photos --recursive --out out

  # One file, searched anywhere under tmp/res; output is flat: out/IMG_0001.jpg
  python convert.py --just IMG_0001.heic --recursive --out out

  # Another source root, re-converting files that already exist in out/
  python convert.py shots --src ~/Pictures --overwrite --out out

Behavior:
  - JPEG inputs (.jpg .jpeg .jpe .jfif .jif, or JPEG content under another
    extension) are skipped, never re-encoded.
  - Transparency is composited onto white. EXIF orientation is applied.
    The ICC profile is kept when it is RGB. Other EXIF data (date, GPS) is dropped.
  - Animated images: only the first frame is converted (a warning is printed).
  - Same-name sources in one folder (a.png + a.webp) become a_png.jpg and
    a_webp.jpg, so nothing overwrites anything.
  - Hidden files/folders (including macOS "._*") are ignored.
  - HEIC/HEIF needs `pip install pillow-heif`; without it those files are
    reported as ignored.
  - Output is written to a .part file first, then renamed, so a crash never
    leaves a half-written JPEG.

Exit codes:
  0  finished with no failures (skips do not count as failures)
  1  directory or --just file not found, or at least one conversion failed
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False


DEFAULT_SRC = Path("tmp/res")
JPEG_QUALITY = 95

JPEG_EXTENSIONS = frozenset({".jpg", ".jpeg", ".jpe", ".jfif", ".jif"})

# Registered by Pillow but save-only, stubs, or dependent on external binaries.
NON_RASTER_EXTENSIONS = frozenset(
    {".pdf", ".ps", ".eps", ".bufr", ".grib", ".h5", ".hdf"}
)

CONVERTIBLE_EXTENSIONS = (
    frozenset(Image.registered_extensions()) - JPEG_EXTENSIONS - NON_RASTER_EXTENSIONS
)

ALPHA_MODES = frozenset({"RGBA", "LA", "PA", "RGBa", "La"})


@dataclass
class Outcome:
    status: str  # ok | exists | jpeg | failed
    multiframe: bool = False
    in_bytes: int = 0
    out_bytes: int = 0


def log(message: str, *, err: bool = False) -> None:
    """Print without corrupting an active tqdm bar."""
    stream = sys.stderr if err else sys.stdout
    if tqdm is not None:
        tqdm.write(message, file=stream)
    else:
        print(message, file=stream)


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def scan(directory: Path, recursive: bool) -> tuple[list[Path], Counter]:
    """Return (sorted convertible images, counter of ignored extensions)."""
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    images: list[Path] = []
    ignored: Counter = Counter()

    for path in iterator:
        relative_parts = path.relative_to(directory).parts
        if any(part.startswith(".") for part in relative_parts):
            continue  # hidden files/dirs, including macOS "._*" resource forks
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in CONVERTIBLE_EXTENSIONS:
            images.append(path)
        elif suffix not in JPEG_EXTENSIONS:
            ignored[suffix or "(no ext)"] += 1

    return sorted(images), ignored


def plan_outputs(
    images: list[Path],
    root: Path,
    out_dir: Path,
) -> dict[Path, Path]:
    """Map each source to a unique output path, preserving relative structure.

    Sources sharing (directory, stem) get an extension suffix (a_png.jpg),
    so the result does not depend on iteration order.
    """
    groups: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for src in images:
        relative = src.relative_to(root)
        groups[(relative.parent, relative.stem.casefold())].append(src)

    plan: dict[Path, Path] = {}
    used: set[str] = set()

    for (parent, _), members in groups.items():
        for src in members:
            stem = src.stem
            if len(members) > 1:
                stem = f"{stem}_{src.suffix.lower().lstrip('.')}"

            dst = out_dir / parent / f"{stem}.jpg"
            counter = 2
            while str(dst).casefold() in used:
                dst = out_dir / parent / f"{stem}_{counter}.jpg"
                counter += 1

            used.add(str(dst).casefold())
            plan[src] = dst

    return plan


def to_rgb(image: Image.Image) -> Image.Image:
    """Flatten any supported mode into 8-bit RGB (alpha composited on white)."""
    if image.mode in ("I", "I;16"):
        _, high = image.getextrema()
        if isinstance(high, float) and high > 65535:
            raise ValueError("integer image exceeds 16-bit range")
        if isinstance(high, float) and high > 255:
            image = image.point(lambda v: v * (1 / 256))
        image = image.convert("L")

    if image.mode in ALPHA_MODES or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background

    return image.convert("RGB")


def convert_image(src: Path, dst: Path, overwrite: bool) -> Outcome:
    """Convert one image and report what happened."""
    if dst.exists() and not overwrite:
        return Outcome("exists")

    tmp = dst.with_name(dst.name + ".part")
    multiframe = False

    try:
        with Image.open(src) as opened:
            if opened.format == "JPEG":
                return Outcome(
                    "jpeg"
                )  # mislabeled JPEG; re-encoding would only lose quality

            multiframe = getattr(opened, "n_frames", 1) > 1 and opened.format != "MPO"
            if multiframe:
                log(f"[WARN] {src}: multi-frame image, using first frame only")

            # Keep the profile only if it matches the RGB output.
            icc = opened.info.get("icc_profile")
            keep_icc = bool(icc) and len(icc) >= 20 and icc[16:20] == b"RGB "

            image = to_rgb(ImageOps.exif_transpose(opened))

        save_kwargs: dict = {
            "format": "JPEG",
            "quality": JPEG_QUALITY,
            "optimize": True,
        }
        if keep_icc:
            save_kwargs["icc_profile"] = icc

        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(tmp, **save_kwargs)
        tmp.replace(dst)

        return Outcome(
            "ok",
            multiframe=multiframe,
            in_bytes=src.stat().st_size,
            out_bytes=dst.stat().st_size,
        )

    except Exception as error:  # Pillow plugins raise SyntaxError, struct.error, etc.
        log(f"[ERROR] {src}: {type(error).__name__}: {error}", err=True)
        return Outcome("failed")

    finally:
        tmp.unlink(missing_ok=True)


def print_summary(
    counts: Counter,
    failed: list[Path],
    multiframe: int,
    in_bytes: int,
    out_bytes: int,
    elapsed: float,
    total: int,
    interrupted: bool,
) -> None:
    processed = sum(counts.values())
    skipped = counts["exists"] + counts["jpeg"]

    print()
    print("=" * 48)
    print(
        "Summary"
        + (f" (INTERRUPTED: {processed}/{total} processed)" if interrupted else "")
    )
    print("=" * 48)

    converted_line = f"  Converted : {counts['ok']}"
    if counts["ok"] and in_bytes:
        change = 1 - out_bytes / in_bytes
        direction = "smaller" if change >= 0 else "larger"
        converted_line += (
            f"  ({human_size(in_bytes)} -> {human_size(out_bytes)}, "
            f"{abs(change):.0%} {direction})"
        )
    print(converted_line)

    print(
        f"  Skipped   : {skipped}"
        f"  ({counts['exists']} output exists, {counts['jpeg']} already JPEG)"
    )
    print(f"  Failed    : {counts['failed']}")

    if multiframe:
        print(f"  Multi-frame, first frame only: {multiframe}")

    print(f"  Elapsed   : {elapsed:.1f} s")

    if failed:
        print("\nFailed files:")
        for path in failed:
            print(f"  {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert non-JPEG images into JPEG.")

    parser.add_argument(
        "subdir",
        nargs="?",
        help="Subdirectory inside --src to process.",
    )

    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help=f"Source root (default: {DEFAULT_SRC}).",
    )

    parser.add_argument(
        "--just",
        metavar="FILENAME",
        help="Convert only the file with this exact name (case-insensitive).",
    )

    parser.add_argument(
        "--recursive",
        "--recurs",
        dest="recursive",
        action="store_true",
        help="Search subdirectories too.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip them).",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print one line per file even when the progress bar is shown.",
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory.",
    )

    args = parser.parse_args()

    if not args.subdir and not args.just:
        parser.error("Specify <subdir> or --just.")

    if args.just and Path(args.just).name != args.just:
        parser.error("--just takes a file name, not a path.")

    return args


def main() -> int:
    args = parse_args()

    base = (args.src / args.subdir if args.subdir else args.src).resolve()
    out_dir = args.out.resolve()

    if not base.is_dir():
        print(f"[ERROR] Directory not found: {base}", file=sys.stderr)
        return 1

    images, ignored = scan(base, args.recursive)

    if ignored:
        summary = ", ".join(f"{ext}({n})" for ext, n in sorted(ignored.items()))
        print(f"[INFO] Ignored (unsupported/non-image): {summary}")

    if not HEIF_AVAILABLE and {".heic", ".heif"} & ignored.keys():
        print(
            "[WARN] HEIC/HEIF files were ignored: install pillow-heif to convert them.",
            file=sys.stderr,
        )

    plan_root = base

    if args.just:
        wanted = args.just.casefold()
        matches = [p for p in images if p.name.casefold() == wanted]

        if not matches:
            print(
                f"[ERROR] No convertible image named {args.just!r} in {base}"
                + ("" if args.recursive else " (try --recursive)"),
                file=sys.stderr,
            )
            return 1

        if len(matches) > 1:
            print(
                f"[WARN] {len(matches)} matches for {args.just!r}; using {matches[0]}"
            )

        images = matches[:1]
        plan_root = images[0].parent  # flat output for a single file

    if not images:
        print(f"[INFO] No convertible images found in {base}")
        return 0

    plan = plan_outputs(images, plan_root, out_dir)

    counts: Counter = Counter()
    failed: list[Path] = []
    multiframe = in_bytes = out_bytes = 0
    interrupted = False

    # disable=None: bar is suppressed automatically when stderr is not a TTY.
    bar = tqdm(images, unit="img", dynamic_ncols=True, disable=None) if tqdm else None
    bar_active = bar is not None and not bar.disable
    show_lines = args.verbose or not bar_active

    started = time.perf_counter()

    try:
        for src in bar if bar is not None else images:
            dst = plan[src]

            if bar is not None and not bar.disable:
                bar.set_postfix_str(src.name[:40], refresh=False)

            outcome = convert_image(src, dst, args.overwrite)
            counts[outcome.status] += 1

            if outcome.status == "ok":
                in_bytes += outcome.in_bytes
                out_bytes += outcome.out_bytes
                multiframe += outcome.multiframe
                if show_lines:
                    log(f"[OK] {src} -> {dst}")
            elif outcome.status == "failed":
                failed.append(src)
            elif show_lines:
                if outcome.status == "exists":
                    log(f"[SKIP] output exists: {dst}")
                else:
                    log(f"[SKIP] already JPEG: {src}")

    except KeyboardInterrupt:
        interrupted = True

    finally:
        if bar is not None:
            bar.close()

    print_summary(
        counts,
        failed,
        multiframe,
        in_bytes,
        out_bytes,
        time.perf_counter() - started,
        len(images),
        interrupted,
    )

    if interrupted:
        return 130
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
