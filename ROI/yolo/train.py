"""Tahap 3: latih detektor ROI YOLOv8n dengan pencatatan lengkap.

    python -m ROI.yolo.train
    python -m ROI.yolo.train --imgsz 416 --epochs 150
    python -m ROI.yolo.train --debug-cuda                 # setelah "illegal memory access": lokasi error yang akurat
    python -m ROI.yolo.train --resume dataset/detector/runs/train/weights/last.pt

Skrip ini adalah supervisor: training dijalankan sebagai child process (ROI.yolo.train_worker) sehingga
supervisor tetap hidup dan bisa mencatat apa pun yang terjadi, termasuk saat child mati mendadak
tanpa traceback Python (segfault / access violation / driver CUDA).

Setiap run menghasilkan folder <folder data.yaml>/logs/<waktu>_<name>/ berisi:
    console.log         seluruh keluaran konsol (stdout+stderr) bertimestamp; progress bar tqdm diringkas
    events.jsonl        metrik/loss/memori tiap epoch, lingkungan torch, exception + traceback
    batches.jsonl       satu baris per batch sebelum diproses (nama file citra + memori CUDA)
    gpu.csv             sampel nvidia-smi tiap --gpu-interval detik (util, VRAM, suhu, daya, clock)
    fault.log           dump traceback Python saat crash fatal
    environment.txt     versi paket, driver GPU, git commit, jumlah citra per subset, argumen
    git_diff.patch      perubahan kode yang belum di-commit saat run dimulai
    run.json            status (running/ok/crashed/interrupted), kode keluar + artinya, durasi
    CRASH_REPORT.txt    ringkasan + petunjuk, hanya ada bila run tidak berakhir normal
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from io import BufferedReader
from pathlib import Path
from typing import Any, Literal, NoReturn, TextIO, cast

from ROI.yolo.train_worker import build_parser

Status = Literal["ok", "crashed", "interrupted"]

ANSI: re.Pattern[bytes] = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")

WIN_EXIT: dict[int, str] = {
    0xC0000005: "ACCESS_VIOLATION (segfault: akses memori ilegal di sisi native/driver)",
    0xC0000409: "STACK_BUFFER_OVERRUN / fast-fail (abort native, mis. CUDA/driver)",
    0xC0000374: "HEAP_CORRUPTION",
    0xC00000FD: "STACK_OVERFLOW",
    0xC000013A: "CTRL_C_EXIT (diinterupsi dengan Ctrl+C)",
    0xC0000135: "DLL_NOT_FOUND (DLL tidak ditemukan; cek instalasi torch/CUDA)",
    0xC0000142: "DLL_INIT_FAILED",
}

# (pola pada console.log/events, petunjuk)
HINTS: list[tuple[bytes, str]] = [
    (
        b"illegal memory access",
        (
            "Error CUDA bersifat ASINKRON: baris di traceback bisa salah. Jalankan ulang dengan --debug-cuda agar traceback menunjuk "
            "operasi yang benar-benar gagal. Periksa batches.jsonl (beberapa batch terakhir), gpu.csv (suhu, VRAM, clock), pastikan "
            "torch dan torchvision berasal dari build CUDA yang sama, lalu bisect dengan --no-amp dan --workers 0."
        ),
    ),
    (
        b"out of memory",
        (
            "VRAM habis. Turunkan --batch atau --imgsz. Bandingkan mem_used di gpu.csv dan alloc_mb/reserved_mb di batches.jsonl: "
            "naik terus antar epoch = kebocoran; melonjak di satu batch = batch itu terlalu berat."
        ),
    ),
    (
        b"device-side assert",
        "Assert di sisi GPU, biasanya indeks kelas/label di luar rentang. Periksa label hasil prepare_yolo; pakai --debug-cuda untuk lokasi.",
    ),
    (
        b"DataLoader worker",
        "Worker dataloader mati. Coba --workers 0 untuk memastikan masalahnya di dataloader (umum di Windows) dan bukan di GPU.",
    ),
    (
        b"No package metadata was found",
        "Paket pendukung hilang atau instalasi rusak (mis. torchvision). Perbaiki instalasi di env, bukan skrip.",
    ),
    (
        b"UnicodeEncodeError",
        "Masalah encoding keluaran konsol; worker dijalankan dengan PYTHONUTF8=1, laporkan jika masih muncul.",
    ),
]

GPU_FIELDS: str = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate"


# ------------------------------------------------------------------ pencatatan konsol
class ConsoleTee:
    """Menulis keluaran child ke console.log: bertimestamp, tanpa kode ANSI, progress bar tqdm diringkas
    (satu baris final per bar + cuplikan berkala selama bar berjalan, supaya posisi terakhir sebelum crash tercatat)."""

    def __init__(self, path: Path, progress_every: float = 15.0) -> None:
        self.f: TextIO = open(path, "w", encoding="utf-8", buffering=1)
        self.buf: bytes = b""
        self.tail: collections.deque[str] = collections.deque(maxlen=200)
        self.lock = threading.Lock()
        self.last_progress: float = 0.0
        self.progress_every: float = progress_every

    @staticmethod
    def _clean(raw: bytes) -> str:
        raw = ANSI.sub(b"", raw)
        segs = [s for s in raw.split(b"\r") if s.strip()]
        return (segs[-1] if segs else b"").decode("utf-8", "replace").rstrip()

    def emit(self, text: str, tag: str = "") -> None:
        if not text:
            return
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {tag}{text}"
        with self.lock:
            self.f.write(line + "\n")
            self.f.flush()
            self.tail.append(line)

    def feed(self, chunk: bytes) -> None:
        self.buf += chunk
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            self.emit(self._clean(line))
        now = time.time()
        if self.buf and now - self.last_progress >= self.progress_every:
            self.emit(self._clean(self.buf), "[progress] ")
            self.last_progress = now

    def finish(self) -> None:
        if self.buf:
            self.emit(self._clean(self.buf), "[sisa buffer] ")
            self.buf = b""


def pump(proc: subprocess.Popen[bytes], tee: ConsoleTee) -> None:
    """Teruskan keluaran child ke konsol apa adanya, sambil mencatatnya lewat ConsoleTee."""
    stream = cast(BufferedReader, proc.stdout)
    out: Any = getattr(sys.stdout, "buffer", None)
    while True:
        chunk = stream.read1(65536)
        if not chunk:
            break
        try:
            if out is not None:
                out.write(chunk)
                out.flush()
            else:
                sys.stdout.write(chunk.decode("utf-8", "replace"))
                sys.stdout.flush()
        except Exception:
            pass
        tee.feed(chunk)


# ------------------------------------------------------------------ telemetri GPU + watchdog
def monitor(
    log_dir: Path,
    stop: threading.Event,
    interval: float,
    stall_minutes: float,
    tee: ConsoleTee,
) -> None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        tee.emit("nvidia-smi tidak ditemukan; telemetri GPU dilewati", "[monitor] ")
    batches = log_dir / "batches.jsonl"
    warned_at = 0.0
    with open(log_dir / "gpu.csv", "w", encoding="utf-8", buffering=1) as f:
        f.write(
            "time,gpu_util_pct,mem_used_mib,mem_total_mib,temp_c,power_w,sm_clock_mhz,mem_clock_mhz,pstate\n"
        )
        while not stop.wait(interval):
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            if exe:
                try:
                    r = subprocess.run(
                        [
                            exe,
                            f"--query-gpu={GPU_FIELDS}",
                            "--format=csv,noheader,nounits",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    f.writelines(
                        now + "," + ",".join(x.strip() for x in ln.split(",")) + "\n"
                        for ln in r.stdout.strip().splitlines()
                    )
                    if r.returncode != 0:
                        f.write(
                            f"{now},ERROR,nvidia-smi exit {r.returncode}: {r.stderr.strip()[:200]!r}\n"
                        )
                except Exception as e:  # nvidia-smi yang macet/gagal juga informasi (GPU hang, driver reset)
                    f.write(f"{now},ERROR,{e!r}\n")
            if stall_minutes > 0 and batches.exists():
                idle = time.time() - batches.stat().st_mtime
                if (
                    idle > stall_minutes * 60
                    and time.time() - warned_at > stall_minutes * 60
                ):
                    msg = (
                        f"tidak ada batch baru selama {idle / 60:.1f} menit (GPU/dataloader hang?). "
                        f"Lihat gpu.csv; proses TIDAK dihentikan otomatis."
                    )
                    tee.emit(msg, "[watchdog] ")
                    print(f"\n[watchdog] {msg}", flush=True)
                    warned_at = time.time()


# ------------------------------------------------------------------ lingkungan
def sh(cmd: list[str], timeout: int = 60) -> str:
    try:
        r = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
        out = (r.stdout + r.stderr).strip()
        return out if r.returncode == 0 else f"<gagal rc={r.returncode}: {out[:200]}>"
    except Exception as e:
        return f"<gagal: {e!r}>"


def capture_environment(
    log_dir: Path, args: argparse.Namespace, data_path: Path
) -> None:
    lines: list[str] = [
        f"waktu mulai : {datetime.now().isoformat(timespec='seconds')}",
        f"perintah    : {' '.join(sys.argv)}",
        f"argumen     : {json.dumps(vars(args), default=str)}",
        f"platform    : {platform.platform()}",
        f"python      : {sys.version.split()[0]} ({sys.executable})",
        f"cwd         : {os.getcwd()}",
        f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX')}",
        "",
    ]
    lines += [
        "== git ==",
        sh(["git", "rev-parse", "HEAD"]),
        sh(["git", "branch", "--show-current"]),
        sh(["git", "status", "--short"]),
        "",
    ]
    diff = sh(["git", "diff"])
    if diff and not diff.startswith("<gagal"):
        (log_dir / "git_diff.patch").write_text(diff[:2_000_000], encoding="utf-8")
    lines.append("== dataset ==")
    try:
        import yaml

        cfg: dict[str, Any] = yaml.safe_load(Path(data_path).read_text())
        root = Path(cfg["path"])
        for s in ("train", "val", "test"):
            imgs = root / cfg[s]
            lbls = root / cfg[s].replace("images", "labels", 1)
            n_img = sum(1 for _ in imgs.iterdir()) if imgs.exists() else 0
            n_lbl = (
                sum(1 for p in lbls.glob("*.txt") if p.stat().st_size > 0)
                if lbls.exists()
                else 0
            )
            lines.append(
                f"{s}: {n_img} citra, {n_lbl} dengan label, {n_img - n_lbl} tanpa label (hard negative)"
            )
    except Exception as e:
        lines.append(f"<gagal membaca dataset: {e!r}>")
    lines += [
        "",
        "== nvidia-smi ==",
        sh(["nvidia-smi"]) if shutil.which("nvidia-smi") else "tidak tersedia",
        "",
        "== pip freeze ==",
        sh([sys.executable, "-m", "pip", "freeze"], 120),
        "",
    ]
    if shutil.which("conda"):
        lines += ["== conda list --export ==", sh(["conda", "list", "--export"], 120)]
    (log_dir / "environment.txt").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ pasca-run
def decode_exit(rc: int | None) -> str:
    if rc is None:
        return "tidak diketahui"
    if rc == 0:
        return "0 (normal)"
    if os.name == "nt":
        u = rc & 0xFFFFFFFF
        if u in WIN_EXIT:
            return f"0x{u:08X} {WIN_EXIT[u]}"
        if u >= 0x80000000:
            return f"0x{u:08X} (kode NTSTATUS tidak dikenal)"
    elif rc < 0:
        import signal

        try:
            return f"sinyal {-rc} {signal.Signals(-rc).name}"
        except ValueError:
            return f"sinyal {-rc}"
    return f"{rc} (umumnya exception Python yang tidak tertangani; lihat traceback di console.log)"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if path.exists():
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def build_report(log_dir: Path, status: Status, rc: int | None, tee: ConsoleTee) -> str:
    events = read_jsonl(log_dir / "events.jsonl")
    batches = read_jsonl(log_dir / "batches.jsonl")
    console = (log_dir / "console.log").read_bytes()
    fault_path = log_dir / "fault.log"
    fault = (
        fault_path.read_text(encoding="utf-8", errors="replace").strip()
        if fault_path.exists()
        else ""
    )
    exc = [e for e in events if e.get("type") == "exception"]
    epochs = [e for e in events if e.get("type") == "epoch_end"]
    start: dict[str, Any] = next(
        (e for e in events if e.get("type") == "train_start"), {}
    )

    report: list[str] = [
        f"STATUS    : {status}",
        f"KODE KELUAR: {decode_exit(rc)}",
        (
            f"UJUNG TRAINING: epoch selesai terakhir = {epochs[-1]['epoch'] if epochs else 'belum ada'}"
            f" dari {start.get('epochs', '?')}"
        ),
        f"folder run ultralytics: {start.get('save_dir', '(belum terbentuk)')}",
        "",
    ]

    has_tb = b"Traceback (most recent call last)" in console
    if rc not in (0, None) and not has_tb and not exc:
        report.append(
            "Tidak ada traceback Python: proses mati dari sisi native (driver/CUDA/memori). Lihat fault.log dan gpu.csv."
        )
    if fault:
        report += ["", "--- fault.log (faulthandler) ---", fault[-3000:]]
    if exc:
        report += [
            "",
            "--- exception tercatat worker ---",
            exc[-1].get("traceback", exc[-1].get("error", ""))[-3000:],
        ]

    blob = console + b"\n".join(
        e.get("traceback", "").encode("utf-8", "replace") for e in exc
    )
    hints = [h for pat, h in HINTS if pat.lower() in blob.lower()]
    if hints:
        report += ["", "--- petunjuk ---"] + [f"* {h}" for h in hints]

    if epochs:
        e = epochs[-1]
        report += [
            "",
            f"--- epoch terakhir ({e['epoch']}) ---",
            f"loss={e.get('loss')}",
            f"metrik={e.get('metrics')}",
            f"nan_loss={e.get('nan_loss')} alloc_mb={e.get('alloc_mb')} peak_mb={e.get('peak_mb')}",
        ]
    if batches:
        report += ["", "--- batch terakhir sebelum berhenti (terbaru di bawah) ---"]
        for b in batches[-5:]:
            report.append(
                f"{b['t']} epoch={b['epoch']} step={b['step']} alloc={b.get('alloc_mb')}MB files={b.get('files')}"
            )
    gpu = log_dir / "gpu.csv"
    if gpu.exists():
        all_rows = gpu.read_text().splitlines()[1:]
        rows: list[list[str]] = [
            ln.split(",") for ln in all_rows if ln and "ERROR" not in ln
        ]
        errs = [ln for ln in all_rows if "ERROR" in ln]
        if rows:

            def col(i: int) -> float | None:
                vals: list[float] = []
                for r in rows:
                    try:
                        vals.append(float(r[i]))
                    except (ValueError, IndexError):
                        pass
                return max(vals) if vals else None

            report += [
                "",
                "--- GPU ---",
                f"sampel terakhir: {','.join(rows[-1])}",
                f"maks: suhu={col(4)}C VRAM={col(2)}MiB daya={col(5)}W util={col(1)}%",
            ]
        if errs:
            report.append(
                f"{len(errs)} sampel nvidia-smi gagal, terakhir: {errs[-1]}  (GPU/driver mungkin hang atau reset)"
            )
    report += ["", "--- 40 baris konsol terakhir ---"] + list(tee.tail)[-40:]
    return "\n".join(report)


def main() -> NoReturn:
    args = build_parser().parse_args()
    data = Path(args.data).resolve()
    root = Path(args.log_root) if args.log_root else data.parent / "logs"
    log_dir = root / f"{datetime.now():%Y%m%d-%H%M%S}_{args.name}"
    log_dir.mkdir(parents=True, exist_ok=False)
    (root / "LATEST.txt").write_text(str(log_dir), encoding="utf-8")

    run: dict[str, Any] = {
        "status": "running",
        "started": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "note": "status 'running' yang tak berubah = supervisor ikut mati (mis. mati listrik, BSOD, jendela ditutup)",
    }
    (log_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    capture_environment(log_dir, args, data)
    if data.exists():
        shutil.copy2(data, log_dir / "data.yaml")

    env = os.environ.copy()
    env.update(
        PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8"
    )  # emoji ultralytics vs cp1252 saat di-pipe
    if args.debug_cuda:
        env.update(CUDA_LAUNCH_BLOCKING="1", TORCH_SHOW_CPP_STACKTRACES="1")
    cmd = [
        sys.executable,
        "-u",
        "-m",
        args.worker_module,
        *sys.argv[1:],
        "--log-dir",
        str(log_dir),
    ]

    tee = ConsoleTee(log_dir / "console.log")
    tee.emit(" ".join(cmd), "[cmd] ")
    print(f"[log] {log_dir}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    stop = threading.Event()
    reader = threading.Thread(target=pump, args=(proc, tee), daemon=True)
    mon = threading.Thread(
        target=monitor,
        args=(log_dir, stop, args.gpu_interval, args.stall_minutes, tee),
        daemon=True,
    )
    reader.start()
    mon.start()

    interrupted = False
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        interrupted = True
        try:
            rc = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
    reader.join(timeout=10)
    tee.finish()
    stop.set()
    mon.join(timeout=20)

    status: Status = "interrupted" if interrupted else ("ok" if rc == 0 else "crashed")
    run.update(
        status=status,
        ended=datetime.now().isoformat(timespec="seconds"),
        duration_min=round((time.time() - t0) / 60, 2),
        exit_code=rc,
        exit_meaning=decode_exit(rc),
    )
    run.pop("note", None)
    (log_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    if status != "ok":
        report = build_report(log_dir, status, rc, tee)
        (log_dir / "CRASH_REPORT.txt").write_text(report, encoding="utf-8")
        print(
            f"\n{'=' * 70}\nRUN {status.upper()} — ringkasan (lengkap: {log_dir / 'CRASH_REPORT.txt'})\n{'=' * 70}"
        )
        print("\n".join(report.splitlines()[:25]))
        print("\nUntuk melanjutkan: --resume <folder run ultralytics>/weights/last.pt")
    else:
        print(f"\n[log] selesai normal, {run['duration_min']} menit -> {log_dir}")
    sys.exit(rc if rc else 0)


if __name__ == "__main__":
    main()
