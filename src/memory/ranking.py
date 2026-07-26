"""Memory ranking by usefulness (similarity + performance + recency + confidence).

Extracted verbatim from Notebook Cell 21
("NEW CELL — MEMORY RETRIEVAL: RANKING & CONFIDENCE").
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd


def _performance_score(metrics: Optional[dict[str, Any]]) -> float:
    """Normalize whatever primary metric is present to a 0-1 'goodness' score."""
    if not metrics:
        return 0.0
    if "f1" in metrics:
        return float(metrics["f1"])
    if "r2" in metrics:
        return float(max(0.0, min(1.0, metrics["r2"])))
    return 0.0


def _recency_score(created_at_iso: Optional[str], half_life_days: float = 30.0) -> float:
    """Exponential decay — a run from today scores ~1.0, one from 30 days
    ago scores ~0.5, one from 90 days ago scores ~0.125.

    Uses pandas' timestamp parser rather than ``datetime.fromisoformat``
    directly — it's more tolerant of the exact ISO format stored across
    SQLite writes and avoids any ambiguity with the ``datetime`` name in
    scopes where it might be shadowed.
    """
    if not created_at_iso:
        return 0.5  # unknown timestamp — neutral, not penalized
    created = pd.Timestamp(created_at_iso)
    if created.tzinfo is not None:
        created = created.tz_localize(None)
    now = pd.Timestamp(datetime.utcnow())
    age_days = (now - created).total_seconds() / 86400.0
    return 0.5 ** (max(age_days, 0.0) / half_life_days)


def _confidence_score(critic_notes: Optional[dict[str, Any]]) -> float:
    """How much to trust this memory's outcome. A Critic-approved run is
    high confidence; a flagged/revised run is lower but not zero — it's
    still informative as a 'what to avoid' signal (see build_planner_context)."""
    if not critic_notes:
        return 0.5  # unknown — neutral
    return 0.9 if critic_notes.get("recommendation") == "approve" else 0.3


def rank_memories(
    candidates: list[dict[str, Any]],
    weights: tuple[float, float, float, float] = (0.4, 0.3, 0.15, 0.15),
) -> list[dict[str, Any]]:
    """Re-order retrieve_memories() output by usefulness, not just similarity.

    usefulness = w_sim*similarity + w_perf*performance + w_recency*recency + w_conf*confidence

    Any candidate missing a required field is skipped and logged rather than
    aborting ranking for the remaining candidates (Task 6, architecture audit).
    """
    w_sim, w_perf, w_recency, w_conf = weights
    enriched: list[dict[str, Any]] = []
    for c in candidates:
        try:
            performance = _performance_score(c.get("metrics"))
            recency = _recency_score(c["created_at"])
            confidence = _confidence_score(c.get("critic_notes"))
            usefulness = (
                w_sim * c["similarity"] + w_perf * performance
                + w_recency * recency + w_conf * confidence
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[rank_memories] Skipping malformed memory candidate "
                  f"run_id={c.get('run_id', 'unknown')}: {exc}")
            continue
        approved = bool(c.get("critic_notes") and c["critic_notes"].get("recommendation") == "approve")
        reason_parts = [f"similarity={c['similarity']:.2f}"]
        if c.get("metrics"):
            reason_parts.append(f"performance={performance:.2f}")
        reason_parts.append(f"recency={recency:.2f}")
        reason_parts.append("Critic-approved" if approved else "not Critic-approved")

        enriched.append({
            **c,
            "confidence": round(confidence, 3),
            "quality": round(performance, 3),
            "quality_label": "success" if approved else "failure",
            "usefulness": round(usefulness, 4),
            "reason": "; ".join(reason_parts),
        })
    enriched.sort(key=lambda m: m["usefulness"], reverse=True)
    return enriched