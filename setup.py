from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / "venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
MODELS_DIR = ROOT / "models"
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"
VOSK_MODEL_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"


def step(text: str) -> None:
    print(f"\n== {text}", flush=True)


def run(*args: object) -> None:
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def in_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()



def bootstrap(argv: list) -> None:
    if sys.version_info < (3, 11):
        sys.exit(f"Signify needs Python 3.11 or newer (this is {sys.version.split()[0]}).")
    if not VENV_PYTHON.exists():
        step("Making the virtual environment (venv/)")
        venv.EnvBuilder(with_pip=True).create(VENV)
    step("Installing the packages (the first time takes a few minutes)")
    upgrade = [VENV_PYTHON, "-m", "pip", "install", "--upgrade", "--quiet", "pip"]
    if subprocess.run([str(a) for a in upgrade + ["--use-feature=truststore"]], cwd=ROOT).returncode:
        run(*upgrade)
    run(VENV_PYTHON, "-m", "pip", "install", "--quiet", "-r", ROOT / "requirements.txt")
    sys.exit(subprocess.run([str(VENV_PYTHON), __file__, *argv], cwd=ROOT).returncode)



def download(url: str, dest: Path) -> None:
    import requests
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in resp.iter_content(chunk_size=1 << 20):
                fh.write(block)
    tmp.replace(dest)


def setup_speech_models() -> None:
    step("Speech models")
    dest = MODELS_DIR / "vosk"
    if dest.is_dir() and any(dest.iterdir()):
        print(f"vosk: already in {dest.relative_to(ROOT)}")
    else:
        MODELS_DIR.mkdir(exist_ok=True)
        print(f"vosk: downloading {VOSK_MODEL_URL}")
        zip_path = MODELS_DIR / f"{VOSK_MODEL_NAME}.zip"
        download(VOSK_MODEL_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(MODELS_DIR)
        if dest.exists():
            shutil.rmtree(dest)
        (MODELS_DIR / VOSK_MODEL_NAME).rename(dest)
        zip_path.unlink()
        print(f"vosk: OK -> {dest.relative_to(ROOT)}")
    from silero_vad import load_silero_vad
    load_silero_vad()
    print("silero voice detection: OK")


def build_pages() -> None:
    step("Building the sign pages")
    run(sys.executable, "-m", "gloss_to_asl.library", "--web", "web")


def rebuild_signs() -> None:
    step("Downloading the WLASL videos and tracking the hands (hours; stop and re-run any time)")
    run(sys.executable, "-m", "gloss_to_asl.setup_clips", "--max-glosses", "0", "--max-per-gloss", "3",
        "--insecure-host", "aslsignbank.haskins.yale.edu")
    step("Checking the signs against ASL-LEX")
    run(sys.executable, "-m", "gloss_to_asl.signcheck", "--apply")
    build_pages()


def main() -> None:
    ap = argparse.ArgumentParser(description="Set up Signify: venv, packages, speech models, sign pages, then run it")
    ap.add_argument("--no-run", action="store_true", help="set up but do not start the app")
    ap.add_argument("--rebuild-signs", action="store_true",
                    help="download the WLASL videos and track every sign again (only to add or fix signs)")
    args = ap.parse_args()
    if not in_venv():
        bootstrap(sys.argv[1:])

    if args.rebuild_signs:
        rebuild_signs()
        return
    setup_speech_models()
    build_pages()
    run_cmd = r"venv\Scripts\python main.py" if os.name == "nt" else "venv/bin/python main.py"
    step(f"Done. To start Signify later: {run_cmd}")
    if not args.no_run:
        run(sys.executable, ROOT / "main.py")


if __name__ == "__main__":
    main()
