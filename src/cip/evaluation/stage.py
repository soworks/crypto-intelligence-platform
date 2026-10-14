"""How far a stored symbol can be reproduced without a later stage.

A policy rejection is complete. An evidence block stays an evidence block.
A symbol that passes the lane still needs the next stage. This module does not
fetch data and does not change eligibility thresholds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from cip.domain.policy import UniverseHypotheses, load_policy
from cip.evaluation.classification import SymbolClassification, to_facts
from cip.evaluation.eligibility import _exclusions, _lane, assess
from cip.evaluation.lane_fields import LaneFields, overlay_lane

Kind = Literal["policy_rejected", "evidence_blocked"]


@dataclass(frozen=True)
class Settlement:
    """The terminal stage already stored for one symbol."""

    kind: Kind
    reason_codes: tuple[str, ...]


def settle(
    item: SymbolClassification,
    lane: LaneFields | None,
    universe: UniverseHypotheses,
) -> Settlement | None:
    """The stored terminal decision, or none when a later stage is still open."""
    facts = to_facts(item, quote_asset="USDT", status="TRADING")
    excluded = tuple(_exclusions(facts, universe))
    chosen, reasons = _lane(facts, universe)
    if excluded or chosen is None:
        paired = tuple(reasons) if chosen is None else ()
        return _settlement(excluded + paired)
    if lane is None:
        return None
    overlaid = overlay_lane(
        facts,
        lane=lane.lane,
        circulating_ratio=lane.circulating_ratio,
        history_days=lane.history_days,
        market_cap_rank=lane.market_cap_rank,
        fdv_to_market_cap=lane.fdv_to_market_cap,
        unlock=lane.unlock_state,
    )
    decision = assess(overlaid, universe)
    if decision.eligible:
        return None
    return _settlement(decision.reason_codes)


def needs_lane(item: SymbolClassification, universe: UniverseHypotheses) -> bool:
    """True when the lane file is the next required observation."""
    facts = to_facts(item, quote_asset="USDT", status="TRADING")
    if _exclusions(facts, universe):
        return False
    chosen, _reasons = _lane(facts, universe)
    return chosen is not None


def load_universe() -> UniverseHypotheses:
    """The workload policy. A checkout uses the repo file."""
    configured = os.environ.get("POLICY_PATH")
    path = (
        Path(configured)
        if configured
        else Path(__file__).resolve().parents[3] / "policies" / "investment-policy.yaml"
    )
    return load_policy(path).policy.hypotheses.universe


def settlements(
    root: Path,
    session: date,
    symbols: tuple[str, ...],
    *,
    sealed: bool = False,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Terminal stages that do not need a later capture. Open symbols are omitted."""
    universe = load_universe()
    found: dict[str, tuple[str, tuple[str, ...]]] = {}
    for symbol in symbols:
        loaded = load_symbol(root, session, symbol, sealed=sealed)
        if loaded is None:
            continue
        item, lane = loaded
        settled = settle(item, lane, universe)
        if settled is not None:
            found[symbol] = (settled.kind, settled.reason_codes)
    return found


def load_symbol(
    root: Path,
    session: date,
    symbol: str,
    *,
    sealed: bool = False,
) -> tuple[SymbolClassification, LaneFields | None] | None:
    """The stored classification and lane file. Missing classification is incomplete."""
    from cip.evaluation.lane_capture import _load_classification, _load_lane

    path = _classification_path(root, session, symbol, sealed=sealed)
    if not path.is_file():
        return None
    item = _load_classification(path)
    if item.symbol != symbol:
        from cip.domain.errors import EvaluationError

        raise EvaluationError("capture is unusable")
    lane_path = _lane_path(root, session, symbol, sealed=sealed)
    lane = _load_lane(lane_path) if lane_path.is_file() else None
    if lane is not None and lane.symbol != symbol:
        from cip.domain.errors import EvaluationError

        raise EvaluationError("capture is unusable")
    return item, lane


def _settlement(codes: tuple[str, ...]) -> Settlement:
    kind: Kind = (
        "policy_rejected"
        if any(not code.startswith("missing_") for code in codes)
        else "evidence_blocked"
    )
    return Settlement(kind, codes)


def _classification_path(root: Path, session: date, symbol: str, *, sealed: bool) -> Path:
    return _stage_dir(root, session, "classification", sealed=sealed) / f"symbol={symbol}.json"


def _lane_path(root: Path, session: date, symbol: str, *, sealed: bool) -> Path:
    return _stage_dir(root, session, "lane", sealed=sealed) / f"symbol={symbol}.json"


def _stage_dir(root: Path, session: date, label: str, *, sealed: bool) -> Path:
    if sealed:
        return root / "sessions" / f"date={session.isoformat()}" / label
    return root / "captures" / f"session={session.isoformat()}" / label
