"""Emit / abstain decision for lexical classifier top-k scores.

Contract is documented in SAFETY.md. Reimplement identically in JavaScript and
check against golden/decision_vectors.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

DecisionAction = Literal["emit", "abstain"]

DEFAULT_CONFIDENCE_THRESHOLD = 0.55
# Used when a risky pair/group has no severity-specific margin.
DEFAULT_MARGIN_THRESHOLD = 0.15
DEFAULT_SEVERITY_MARGINS: dict[str, float] = {
    "critical": 0.25,
    "high": 0.20,
    "medium": 0.15,
}

SAFETY_DIR = Path(__file__).resolve().parent
DEFAULT_RISKY_PAIRS_PATH = SAFETY_DIR / "risky_pairs.json"


@dataclass(frozen=True)
class ScoredLabel:
    label: str
    score: float


@dataclass(frozen=True)
class RiskyHit:
    a: str
    b: str
    severity: str
    category: str
    source: str  # "pair" | "group"
    group_id: str | None = None


@dataclass(frozen=True)
class Decision:
    action: DecisionAction
    label: str | None
    guess: str | None
    confidence: float
    margin: float | None
    reason: str
    severity: str | None = None
    risky_pair: tuple[str, str] | None = None
    category: str | None = None


def _as_scored(items: Sequence[ScoredLabel | Mapping[str, Any] | Sequence[Any]]) -> list[ScoredLabel]:
    out: list[ScoredLabel] = []
    for item in items:
        if isinstance(item, ScoredLabel):
            out.append(item)
            continue
        if isinstance(item, Mapping):
            label = item.get("label", item.get("gloss"))
            score = item.get("score", item.get("prob", item.get("confidence")))
            if label is None or score is None:
                raise ValueError(f"scored item missing label/score: {item!r}")
            out.append(ScoredLabel(str(label), float(score)))
            continue
        if isinstance(item, Sequence) and len(item) >= 2:
            out.append(ScoredLabel(str(item[0]), float(item[1])))
            continue
        raise ValueError(f"unsupported scored item: {item!r}")
    return out


class RiskyLexicon:
    """Undirected risky confusions from risky_pairs.json."""

    def __init__(
        self,
        pairs: Iterable[Mapping[str, Any]],
        groups: Iterable[Mapping[str, Any]] | None = None,
        severity_margins: Mapping[str, float] | None = None,
    ) -> None:
        self.severity_margins: dict[str, float] = dict(
            severity_margins if severity_margins is not None else DEFAULT_SEVERITY_MARGINS
        )
        # frozenset({a,b}) -> RiskyHit template (labels filled at lookup)
        self._pair_index: dict[frozenset[str], dict[str, Any]] = {}
        for p in pairs:
            a = str(p["a"])
            b = str(p["b"])
            if a == b:
                continue
            key = frozenset((a, b))
            # First declaration wins (stable draft order).
            if key in self._pair_index:
                continue
            self._pair_index[key] = {
                "severity": str(p.get("severity", "medium")),
                "category": str(p.get("category", "")),
                "source": "pair",
                "group_id": None,
            }

        self._groups: list[dict[str, Any]] = []
        for g in groups or []:
            members = [str(m) for m in g.get("members", [])]
            member_set = frozenset(members)
            self._groups.append(
                {
                    "id": str(g.get("id", "")),
                    "severity": str(g.get("severity", "medium")),
                    "category": str(g.get("category", "")),
                    "members": member_set,
                }
            )

    @classmethod
    def from_json_dict(cls, doc: Mapping[str, Any]) -> "RiskyLexicon":
        margins = doc.get("default_margins")
        return cls(
            pairs=list(doc.get("pairs") or []),
            groups=list(doc.get("groups") or []),
            severity_margins=margins if isinstance(margins, dict) else None,
        )

    @classmethod
    def load(cls, path: Path | str | None = None) -> "RiskyLexicon":
        p = Path(path) if path is not None else DEFAULT_RISKY_PAIRS_PATH
        doc = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_json_dict(doc)

    def lookup(self, a: str, b: str) -> RiskyHit | None:
        if a == b:
            return None
        key = frozenset((a, b))
        meta = self._pair_index.get(key)
        if meta is not None:
            return RiskyHit(
                a=a,
                b=b,
                severity=meta["severity"],
                category=meta["category"],
                source="pair",
                group_id=None,
            )
        for g in self._groups:
            members: frozenset[str] = g["members"]
            if a in members and b in members:
                return RiskyHit(
                    a=a,
                    b=b,
                    severity=g["severity"],
                    category=g["category"],
                    source="group",
                    group_id=g["id"],
                )
        return None

    def margin_for(self, severity: str, fallback: float = DEFAULT_MARGIN_THRESHOLD) -> float:
        if severity in self.severity_margins:
            return float(self.severity_margins[severity])
        return float(fallback)


def decide(
    topk: Sequence[ScoredLabel | Mapping[str, Any] | Sequence[Any]],
    lexicon: RiskyLexicon,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    margin_threshold: float | None = None,
) -> Decision:
    """Return emit or abstain for a top-k score list.

    Exact decision order (must match JS):

    1. Sort scores by score descending, stable for ties (original order).
    2. If empty -> abstain, reason ``empty``.
    3. If top1.score < confidence_threshold -> abstain, reason ``low_confidence``.
       Guess is top1.label; label is null (not emitted).
    4. If at least two scores and lexicon.lookup(top1, top2) is a hit:
       margin = top1.score - top2.score.
       required = lexicon.margin_for(severity) if margin_threshold is None
                  else margin_threshold.
       If margin < required -> abstain, reason ``risky_confusion``.
       Guess is top1.label; label is null (never emit either side as a commit).
    5. Else emit top1, reason ``ok``.

    Scores are assumed to be calibrated probabilities or comparable confidences
    in roughly [0, 1]; the logic itself only compares thresholds and margins.
    """
    scored = _as_scored(topk)
    # Stable sort: Python's sort is stable, so equal scores keep input order.
    scored.sort(key=lambda s: s.score, reverse=True)

    if not scored:
        return Decision(
            action="abstain",
            label=None,
            guess=None,
            confidence=0.0,
            margin=None,
            reason="empty",
        )

    top1 = scored[0]
    top2 = scored[1] if len(scored) > 1 else None
    margin = (top1.score - top2.score) if top2 is not None else None

    if top1.score < confidence_threshold:
        return Decision(
            action="abstain",
            label=None,
            guess=top1.label,
            confidence=top1.score,
            margin=margin,
            reason="low_confidence",
        )

    if top2 is not None:
        hit = lexicon.lookup(top1.label, top2.label)
        if hit is not None:
            assert margin is not None
            required = (
                float(margin_threshold)
                if margin_threshold is not None
                else lexicon.margin_for(hit.severity)
            )
            if margin < required:
                ordered = tuple(sorted((top1.label, top2.label)))
                return Decision(
                    action="abstain",
                    label=None,
                    guess=top1.label,
                    confidence=top1.score,
                    margin=margin,
                    reason="risky_confusion",
                    severity=hit.severity,
                    risky_pair=(ordered[0], ordered[1]),
                    category=hit.category,
                )

    return Decision(
        action="emit",
        label=top1.label,
        guess=top1.label,
        confidence=top1.score,
        margin=margin,
        reason="ok",
    )


def decision_to_dict(d: Decision) -> dict[str, Any]:
    return {
        "action": d.action,
        "label": d.label,
        "guess": d.guess,
        "confidence": d.confidence,
        "margin": d.margin,
        "reason": d.reason,
        "severity": d.severity,
        "risky_pair": list(d.risky_pair) if d.risky_pair is not None else None,
        "category": d.category,
    }
