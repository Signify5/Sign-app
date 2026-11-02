"""Enumerate local WLASL/Signbank videos and write signer-disjoint manifests.

Outputs (under this directory):
  lexical_manifest.json  -- lexical gloss clips with train/val/test
  letters_manifest.json  -- single-letter fingerspell classes (separate)
  split_report.md        -- human-readable counts and coverage flags
  split_report.json      -- same numbers as JSON

Does not copy or commit videos. Deterministic split with a fixed seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

WLASL_JSON = ROOT / "datasets" / "wlasl" / "WLASL_v0.3.json"
WLASL_DIR = ROOT / "datasets" / "wlasl"
SIGNBANK_DIR = ROOT / "datasets" / "signbank_videos"
CLIPS_INDEX = ROOT / "clips" / "index.json"
EXPANSION_MANIFEST = ROOT / "datasets" / "expansion" / "new_words_manifest.csv"

SEED = 42
# Signer fractions after a seeded shuffle of signer ids.
TEST_SIGNER_FRAC = 0.15
VAL_SIGNER_FRAC = 0.15
# Classes with fewer than this many lexical clips are flagged "too few".
MIN_CLIPS = 5

LETTER_GLOSSES = set("abcdefghijklmnopqrstuvwxyz")


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _load_clips_index() -> Dict[str, dict]:
    """Map clip_id -> index row for motion/pose metadata."""
    if not CLIPS_INDEX.is_file():
        return {}
    data = json.loads(CLIPS_INDEX.read_text(encoding="utf-8"))
    return {c["clip_id"]: c for c in data.get("clips", [])}


def _pose_path(gloss: str, clip_id: str) -> Optional[str]:
    p = ROOT / "clips" / "pose" / gloss / f"{clip_id}.npy"
    return _rel(p) if p.is_file() else None


def _motion_path_from_index(row: Optional[dict]) -> Optional[str]:
    if not row:
        return None
    p = ROOT / "clips" / row["path"]
    return _rel(p) if p.is_file() else None


def enumerate_wlasl(clips_by_id: Dict[str, dict]) -> List[dict]:
    """All local WLASL mp4s listed in WLASL_v0.3.json."""
    data = json.loads(WLASL_JSON.read_text(encoding="utf-8"))
    rows: List[dict] = []
    for entry in data:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            video_id = str(inst["video_id"])
            video = WLASL_DIR / gloss / f"{video_id}.mp4"
            if not video.is_file():
                continue
            clip_id = f"wlasl-{video_id}"
            idx = clips_by_id.get(clip_id)
            rows.append({
                "clip_id": clip_id,
                "video_path": _rel(video),
                "gloss": gloss,
                "source": "wlasl",
                "wlasl_source": inst.get("source"),
                "signer_id": inst.get("signer_id"),
                "wlasl_official_split": inst.get("split"),
                "motion_path": _motion_path_from_index(idx),
                "pose_path": _pose_path(gloss, clip_id),
            })
    return rows


def enumerate_signbank(clips_by_id: Dict[str, dict]) -> List[dict]:
    """Signbank clips from clips/index.json whose mp4 exists. signer_id is null; train-only."""
    gloss_by_cid: Dict[str, str] = {}
    if EXPANSION_MANIFEST.is_file():
        with EXPANSION_MANIFEST.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                gloss_by_cid[r["signbank_cve_id"]] = r["word"].lower()

    rows: List[dict] = []
    for clip_id, idx in clips_by_id.items():
        if not clip_id.startswith("signbank-"):
            continue
        cid = clip_id.split("-", 1)[1]
        video = SIGNBANK_DIR / f"{cid}.mp4"
        if not video.is_file():
            continue
        gloss = idx.get("gloss") or gloss_by_cid.get(cid)
        if not gloss:
            continue
        rows.append({
            "clip_id": clip_id,
            "video_path": _rel(video),
            "gloss": gloss,
            "source": "signbank",
            "wlasl_source": None,
            "signer_id": None,
            "wlasl_official_split": None,
            "motion_path": _motion_path_from_index(idx),
            "pose_path": _pose_path(gloss, clip_id),
        })
    return rows


def assign_signer_splits(
    signer_ids: Sequence[int],
    seed: int,
    test_frac: float,
    val_frac: float,
) -> Dict[int, str]:
    """Map each signer_id to train/val/test. Deterministic given seed."""
    ids = sorted(set(signer_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * test_frac))) if n else 0
    n_val = max(1, int(round(n * val_frac))) if n else 0
    if n_test + n_val >= n and n > 2:
        # Leave at least one train signer when possible.
        overflow = n_test + n_val - (n - 1)
        # Prefer shrinking val first.
        shrink_val = min(overflow, n_val - 1) if n_val > 1 else 0
        n_val -= shrink_val
        overflow -= shrink_val
        if overflow > 0 and n_test > 1:
            n_test -= overflow
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test:n_test + n_val])
    out: Dict[int, str] = {}
    for s in ids:
        if s in test_ids:
            out[s] = "test"
        elif s in val_ids:
            out[s] = "val"
        else:
            out[s] = "train"
    return out


def apply_splits(rows: Iterable[dict], signer_split: Dict[int, str]) -> List[dict]:
    """Assign split. Unknown/null signer_id (Signbank) -> train only."""
    out: List[dict] = []
    for r in rows:
        row = dict(r)
        sid = row.get("signer_id")
        if sid is None:
            row["split"] = "train"
        else:
            row["split"] = signer_split[int(sid)]
        out.append(row)
    return out


def partition_lexical_letters(rows: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    """Single-letter WLASL glosses go to the letters manifest; everything else lexical."""
    lexical: List[dict] = []
    letters: List[dict] = []
    for r in rows:
        if r["source"] == "wlasl" and r["gloss"] in LETTER_GLOSSES and len(r["gloss"]) == 1:
            letters.append(r)
        else:
            lexical.append(r)
    return lexical, letters


def _clips_per_class(rows: Sequence[dict]) -> Dict[str, int]:
    return dict(Counter(r["gloss"] for r in rows))


def _signer_counts(rows: Sequence[dict]) -> Dict[str, int]:
    """Distinct signers per split (null signer counted as one 'null' bucket for train)."""
    by_split: Dict[str, Set[Any]] = defaultdict(set)
    for r in rows:
        sid = r.get("signer_id")
        key = "null" if sid is None else int(sid)
        by_split[r["split"]].add(key)
    return {sp: len(ids) for sp, ids in sorted(by_split.items())}


def build_report(
    lexical: Sequence[dict],
    letters: Sequence[dict],
    signer_split: Dict[int, str],
    seed: int,
    min_clips: int,
) -> Dict[str, Any]:
    per_class = _clips_per_class(lexical)
    too_few = sorted(g for g, n in per_class.items() if n < min_clips)

    by_split = Counter(r["split"] for r in lexical)
    by_source = Counter(r["source"] for r in lexical)

    test_classes = {r["gloss"] for r in lexical if r["split"] == "test"}
    all_classes = set(per_class)
    no_held_out = sorted(all_classes - test_classes)

    letter_per_class = _clips_per_class(letters)
    letter_by_split = Counter(r["split"] for r in letters)

    split_signer_sets: Dict[str, Set[int]] = defaultdict(set)
    for sid, sp in signer_split.items():
        split_signer_sets[sp].add(sid)

    report: Dict[str, Any] = {
        "seed": seed,
        "test_signer_frac": TEST_SIGNER_FRAC,
        "val_signer_frac": VAL_SIGNER_FRAC,
        "min_clips": min_clips,
        "lexical": {
            "num_clips": len(lexical),
            "num_classes": len(per_class),
            "clips_per_split": dict(by_split),
            "clips_per_source": dict(by_source),
            "signer_counts_per_split": {
                sp: len(ids) for sp, ids in sorted(split_signer_sets.items())
            },
            "signbank_train_clips": sum(
                1 for r in lexical if r["source"] == "signbank"
            ),
            "clips_per_class": dict(sorted(per_class.items())),
            "classes_too_few_clips": too_few,
            "num_classes_too_few": len(too_few),
            "classes_no_held_out_signer_coverage": no_held_out,
            "num_classes_no_held_out": len(no_held_out),
            "wlasl_videos_present": sum(1 for r in lexical if r["source"] == "wlasl"),
        },
        "letters": {
            "num_clips": len(letters),
            "num_classes": len(letter_per_class),
            "classes_present": sorted(letter_per_class),
            "classes_missing_a_to_z": sorted(LETTER_GLOSSES - set(letter_per_class)),
            "clips_per_class": dict(sorted(letter_per_class.items())),
            "clips_per_split": dict(letter_by_split),
            "signer_counts_per_split": _signer_counts(letters),
        },
        "signer_split": {
            "train": sorted(split_signer_sets.get("train", [])),
            "val": sorted(split_signer_sets.get("val", [])),
            "test": sorted(split_signer_sets.get("test", [])),
        },
        "notes": [
            "Split is signer-disjoint over WLASL signer_id values.",
            "Signbank clips have signer_id null and are train-only.",
            "Single-letter WLASL glosses are excluded from the lexical manifest "
            "and listed only in letters_manifest.json.",
            "Only clips whose local video file exists are indexed.",
            f"Held-out test signers are the 95% gate set (seed={seed}).",
        ],
    }
    return report


def render_report_md(report: Dict[str, Any]) -> str:
    lex = report["lexical"]
    let = report["letters"]
    lines = [
        "# Sign-to-speech dataset split report",
        "",
        f"Seed: {report['seed']}. "
        f"Test signer frac: {report['test_signer_frac']}. "
        f"Val signer frac: {report['val_signer_frac']}. "
        f"Too-few threshold: < {report['min_clips']} clips.",
        "",
        "## Lexical",
        "",
        f"- Classes: {lex['num_classes']}",
        f"- Clips: {lex['num_clips']}",
        f"- Clips per split: {lex['clips_per_split']}",
        f"- Clips per source: {lex['clips_per_source']}",
        f"- Distinct WLASL signers per split: {lex['signer_counts_per_split']}",
        f"- Signbank train-only clips: {lex['signbank_train_clips']}",
        f"- Classes with too few clips (< {report['min_clips']}): "
        f"{lex['num_classes_too_few']}",
        f"- Classes with no held-out (test) signer coverage: "
        f"{lex['num_classes_no_held_out']}",
        "",
        "### Classes with too few clips",
        "",
    ]
    max_show = 80
    if lex["classes_too_few_clips"]:
        shown = lex["classes_too_few_clips"]
        for g in shown[:max_show]:
            lines.append(f"- {g}: {lex['clips_per_class'][g]}")
        if len(shown) > max_show:
            lines.append(
                f"- ... and {len(shown) - max_show} more "
                f"(full list in split_report.json)"
            )
    else:
        lines.append("- (none)")
    lines += [
        "",
        "### Classes with no held-out signer coverage",
        "",
        "These classes have zero clips in the test split under the signer-disjoint "
        "assignment (cannot contribute to the held-out 95% gate as written).",
        "",
    ]
    if lex["classes_no_held_out_signer_coverage"]:
        shown = lex["classes_no_held_out_signer_coverage"]
        for g in shown[:max_show]:
            lines.append(f"- {g} ({lex['clips_per_class'][g]} clips)")
        if len(shown) > max_show:
            lines.append(
                f"- ... and {len(shown) - max_show} more "
                f"(full list in split_report.json)"
            )
    else:
        lines.append("- (none)")

    lines += [
        "",
        "## Fingerspell letters (separate manifest)",
        "",
        f"- Letter classes present: {let['num_classes']} "
        f"({', '.join(let['classes_present'])})",
        f"- Missing A-Z: {', '.join(let['classes_missing_a_to_z']) or '(none)'}",
        f"- Clips: {let['num_clips']}",
        f"- Clips per split: {let['clips_per_split']}",
        f"- Signers per split: {let['signer_counts_per_split']}",
        f"- Clips per class: {let['clips_per_class']}",
        "",
        "## Notes",
        "",
    ]
    for n in report["notes"]:
        lines.append(f"- {n}")
    lines.append("")
    return "\n".join(lines)


def write_manifest(path: Path, rows: Sequence[dict], kind: str) -> None:
    payload = {
        "version": 1,
        "kind": kind,
        "seed": SEED,
        "num_clips": len(rows),
        "clips": sorted(rows, key=lambda r: (r["gloss"], r["clip_id"])),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-clips", type=int, default=MIN_CLIPS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    clips_by_id = _load_clips_index()
    wlasl_rows = enumerate_wlasl(clips_by_id)
    signbank_rows = enumerate_signbank(clips_by_id)
    all_rows = wlasl_rows + signbank_rows

    # Signer pool from every local WLASL video (letters included) so the same
    # signer never appears in two splits across lexical and letter manifests.
    signer_ids = sorted({
        int(r["signer_id"]) for r in all_rows if r.get("signer_id") is not None
    })
    signer_split = assign_signer_splits(
        signer_ids, args.seed, TEST_SIGNER_FRAC, VAL_SIGNER_FRAC
    )
    all_split = apply_splits(all_rows, signer_split)
    lexical, letters = partition_lexical_letters(all_split)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(out_dir / "lexical_manifest.json", lexical, "lexical")
    write_manifest(out_dir / "letters_manifest.json", letters, "letters")

    report = build_report(lexical, letters, signer_split, args.seed, args.min_clips)
    (out_dir / "split_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "split_report.md").write_text(
        render_report_md(report), encoding="utf-8"
    )

    print(
        f"lexical: {len(lexical)} clips, {report['lexical']['num_classes']} classes; "
        f"letters: {len(letters)} clips, {report['letters']['num_classes']} classes; "
        f"signers train/val/test="
        f"{report['lexical']['signer_counts_per_split']}"
    )
    print(f"wrote under {out_dir.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
