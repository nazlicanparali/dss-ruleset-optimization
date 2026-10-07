#!/usr/bin/env python3
"""Downloads application_train.csv (Home Credit Default Risk) into data/.

Needs a Kaggle API token (~/.kaggle/kaggle.json) and you have to accept the
competition rules on the Kaggle page once. Optional: without the file the
project uses synthetic data.

    pip install kaggle
    python scripts/download_kaggle_data.py
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
COMPETITION = "home-credit-default-risk"
TARGET_FILE = "application_train.csv"


def main() -> None:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("The 'kaggle' package isn't installed. Run: pip install kaggle", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)
    dest = DATA_DIR / TARGET_FILE
    if dest.exists():
        print(f"{dest} already exists")
        return

    print(f"Downloading '{TARGET_FILE}' from the '{COMPETITION}' Kaggle competition...")
    result = subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-f",
            TARGET_FILE,
            "-p",
            str(DATA_DIR),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        print("download failed (check kaggle.json and that the competition rules are accepted)",
              file=sys.stderr)
        sys.exit(1)

    zip_path = DATA_DIR / f"{TARGET_FILE}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()

    print(f"Done: {dest}")


if __name__ == "__main__":
    main()
