"""Download the IEEE-CIS Fraud Detection dataset from Kaggle into data/raw/.

Prerequisites (one-time, manual — cannot be automated):
1. Create a Kaggle API token: https://www.kaggle.com/settings -> API -> "Create New Token".
   This downloads kaggle.json.
2. Place kaggle.json at C:\\Users\\<you>\\.kaggle\\kaggle.json
3. Accept the competition rules at
   https://www.kaggle.com/c/ieee-fraud-detection/rules
   (Kaggle blocks API downloads for competitions you haven't joined.)

Usage:
    .venv/Scripts/python.exe src/data/download_data.py
"""

import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "ieee-fraud-detection"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

EXPECTED_FILES = {
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
}


def already_downloaded() -> bool:
    return all((RAW_DIR / f).exists() for f in EXPECTED_FILES)


def main() -> None:
    if already_downloaded():
        print(f"All expected files already present in {RAW_DIR}. Nothing to do.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle  # noqa: F401  (import validates kaggle.json is configured)
    except OSError as exc:
        sys.exit(
            "Kaggle API credentials not found or invalid.\n"
            "Create a token at https://www.kaggle.com/settings -> API -> 'Create New Token'\n"
            "and place kaggle.json at C:\\Users\\<you>\\.kaggle\\kaggle.json\n"
            f"Original error: {exc}"
        )

    print(f"Downloading competition data for '{COMPETITION}' ...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-p",
            str(RAW_DIR),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(
            "Kaggle download failed. If this mentions 403 Forbidden, you likely "
            "haven't accepted the competition rules yet:\n"
            f"https://www.kaggle.com/c/{COMPETITION}/rules\n\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    print(result.stdout)

    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    missing = EXPECTED_FILES - {p.name for p in RAW_DIR.glob("*.csv")}
    if missing:
        sys.exit(f"Download/extract finished but files are missing: {missing}")

    print(f"Done. Files ready in {RAW_DIR}")


if __name__ == "__main__":
    main()
