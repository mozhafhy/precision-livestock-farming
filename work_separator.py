import shutil
import sys
from collections import defaultdict
from pathlib import Path

from utils import get_file_number, reset_author_dir


def main() -> None:
    target_dir = Path("./dataset/original-dataset/Train")
    works_dir = Path("./dataset/works")
    mapping = {
        "kalist": "ncd",
        "richard": "pcrsalmo",
        "zhafif": "cocci",
    }

    N_SAMPLES = 50
    HEALTHY_CHUNK = 50

    # Reset dulu — sekali di awal, untuk SEMUA author, sebelum copy apa pun.
    for author in mapping:
        reset_author_dir(works_dir / author)

    sorted_files = sorted(
        [f for f in target_dir.iterdir() if f.is_file()], key=get_file_number
    )

    counts = defaultdict(int)
    for file_path in sorted_files:
        for author, prefix in mapping.items():
            if file_path.name.startswith(prefix):
                if counts[author] < N_SAMPLES:
                    shutil.copy2(file_path, works_dir / author / file_path.name)
                    counts[author] += 1
                break

    for author, prefix in mapping.items():
        if counts[author] < N_SAMPLES:
            print(
                f"PERINGATAN: {author} ({prefix}) cuma {counts[author]}/{N_SAMPLES}.",
                file=sys.stderr,
            )

    healthy_files = [f for f in sorted_files if f.name.startswith("healthy")]
    required = HEALTHY_CHUNK * len(mapping)
    if len(healthy_files) < required:
        print(
            f"PERINGATAN: healthy cuma {len(healthy_files)}, butuh {required}.",
            file=sys.stderr,
        )

    for i, author in enumerate(mapping):
        chunk = healthy_files[i * HEALTHY_CHUNK : (i + 1) * HEALTHY_CHUNK]
        for file_path in chunk:
            shutil.copy2(file_path, works_dir / author / file_path.name)


if __name__ == "__main__":
    main()
