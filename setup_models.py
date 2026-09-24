"""Downloads and verifies the Vosk model plus silero-vad, into a top-level models/
directory. Idempotent: skips anything already present and valid. Exits non-zero if
anything required is missing/corrupt after running.

Usage:
    python setup_models.py
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import requests

MODELS_DIR = Path("models")

VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"
VOSK_MODEL_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in resp.iter_content(chunk_size=1 << 20):
                fh.write(block)
    tmp.rename(dest)


def setup_vosk() -> None:
    dest_dir = MODELS_DIR / "vosk"
    if dest_dir.is_dir() and any(dest_dir.iterdir()):
        print(f"[vosk] already present at {dest_dir}, skipping download")
        return

    print(f"[vosk] downloading {VOSK_MODEL_URL} ...")
    zip_path = MODELS_DIR / f"{VOSK_MODEL_NAME}.zip"
    _download(VOSK_MODEL_URL, zip_path)

    print("[vosk] extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(MODELS_DIR)
    extracted = MODELS_DIR / VOSK_MODEL_NAME
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    extracted.rename(dest_dir)
    zip_path.unlink()

    if not any(dest_dir.iterdir()):
        raise RuntimeError(f"vosk model extraction to {dest_dir} produced an empty directory")
    print(f"[vosk] OK -> {dest_dir}")


def setup_silero_vad() -> None:
    from silero_vad import load_silero_vad

    print("[silero-vad] loading (downloads/verifies its bundled model if needed)...")
    load_silero_vad()
    print("[silero-vad] OK")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    failures = []

    for name, fn in (("silero-vad", setup_silero_vad), ("vosk", setup_vosk)):
        try:
            fn()
        except Exception as exc:
            failures.append((name, exc))

    print()
    if failures:
        print("FAILED:")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        sys.exit(1)
    print("All models downloaded/verified successfully.")


if __name__ == "__main__":
    main()
