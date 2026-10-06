from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

CLIPS_DIR = Path("clips")  # what the app builds: motion, index, checks
DATASETS_DIR = Path("datasets")  # what was downloaded: WLASL videos, ASL-LEX (licences: not redistributable)
WLASL_DIR = DATASETS_DIR / "wlasl"
INDEX_FILENAME = "index.json"
INDEX_VERSION = 1


class ClipIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class Clip:
    gloss: str
    path: str
    clip_id: str
    signer_id: Optional[int] = None
    quality: float = 0.0  # share of frames with a tracked hand; path is a motion .npz
    faults: int = 0  # failed sign-form checks, 10 per serious one (signcheck.py --apply); fewer ranks first


def normalize_gloss(gloss: str) -> str:
    return gloss.strip().upper()


class ClipIndex:
    """gloss -> clips lookup. Qt-free; paths are absolute once loaded."""

    def __init__(self, clips: Iterable[Clip], letters: Optional[Dict[str, Clip]] = None) -> None:
        self._by_gloss: Dict[str, List[Clip]] = {}
        for clip in clips:
            self._by_gloss.setdefault(normalize_gloss(clip.gloss), []).append(clip)

        signer_coverage: Dict[int, int] = {}
        for group in self._by_gloss.values():
            for signer in {c.signer_id for c in group if c.signer_id is not None}:
                signer_coverage[signer] = signer_coverage.get(signer, 0) + 1

        def rank(clip: Clip):
            coverage = signer_coverage.get(clip.signer_id, 0) if clip.signer_id is not None else 0
            return (clip.faults, -round(clip.quality, 1), -coverage, clip.clip_id)

        for group in self._by_gloss.values():
            group.sort(key=rank)

        self._letters = {k.upper(): v for k, v in (letters or {}).items()}

    @property
    def glosses(self) -> List[str]:
        return sorted(self._by_gloss)

    def lookup(self, gloss: str) -> List[Clip]:
        """All clips for a gloss, best first (deterministic)."""
        return list(self._by_gloss.get(normalize_gloss(gloss), ()))

    def best(self, gloss: str) -> Optional[Clip]:
        found = self._by_gloss.get(normalize_gloss(gloss))
        return found[0] if found else None

    def letter(self, char: str) -> Optional[Clip]:
        """Fingerspelling clip: dedicated letter clips first, else a one-letter WLASL gloss."""
        key = char.upper()
        return self._letters.get(key) or self.best(key)

    @classmethod
    def load(cls, path: Union[Path, str] = CLIPS_DIR / INDEX_FILENAME) -> "ClipIndex":
        path = Path(path)
        if path.is_dir():
            path = path / INDEX_FILENAME
        if not path.is_file():
            raise ClipIndexError(f"clip index not found at {path} -- run setup_clips.py first")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ClipIndexError(f"could not read clip index {path}: {exc}") from exc
        if data.get("version") != INDEX_VERSION:
            raise ClipIndexError(f"unsupported clip index version in {path}")

        base = path.parent

        def to_clip(entry: dict) -> Clip:
            entry = dict(entry)
            entry["path"] = str((base / entry["path"]).resolve())
            return Clip(**entry)

        clips = [to_clip(e) for e in data.get("clips", [])]
        letters = {k: to_clip(v) for k, v in data.get("letters", {}).items()}
        return cls(clips, letters)

    @staticmethod
    def write(
        path: Union[Path, str],
        clips: Iterable[Clip],
        letters: Optional[Dict[str, Clip]] = None,
    ) -> None:
        """Write an index; clip paths must live under the index's directory (stored relative)."""
        path = Path(path)
        base = path.parent.resolve()

        def to_entry(clip: Clip) -> dict:
            entry = asdict(clip)
            entry["path"] = Path(clip.path).resolve().relative_to(base).as_posix()
            return entry

        payload = {
            "version": INDEX_VERSION,
            "clips": [to_entry(c) for c in clips],
            "letters": {k: to_entry(v) for k, v in (letters or {}).items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
