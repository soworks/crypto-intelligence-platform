"""Closed-session readiness. A missing input stays missing."""

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from cip.domain.errors import EvaluationError
from cip.evaluation.scan import ScanCandidate, UniverseSnapshot
from cip.history.bars import DailyBar
from cip.recorders.observation import Observation


class InputPresence(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    NOT_PRODUCED = "not_produced"
    LOOKAHEAD = "lookahead"


def session_close(session: date) -> datetime:
    """00:00 UTC on the day after the bar open date."""
    if type(session) is not date:
        raise EvaluationError("session is a date")
    return datetime.combine(session + timedelta(days=1), time.min, tzinfo=UTC)


class SessionReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    blocks: tuple[str, ...]
    decision_notes: tuple[str, ...]
    score_weights: Literal["absent", "present"]

    @model_validator(mode="after")
    def _ready_has_no_blocks(self) -> "SessionReadiness":
        if any(item == "" for item in self.blocks + self.decision_notes):
            raise EvaluationError("readiness reasons are non-empty")
        if self.ready and self.blocks:
            raise EvaluationError("a ready session has no blocks")
        return self


_BTC = "BTCUSDT"
_REGIME = ("btc_dominance", "stablecoin_supply")


def assess_session(
    session: date,
    as_of: datetime,
    *,
    snapshot: UniverseSnapshot | None,
    bars: Mapping[str, Sequence[DailyBar]],
    bar_months_present: Mapping[str, bool],
    candidates: Mapping[str, ScanCandidate | None],
    candidate_files_present: Mapping[str, bool],
    observations: Sequence[Observation],
    regime_failures: frozenset[str],
    weights_present: bool,
    bar_absences: Mapping[str, bool] | None = None,
    settled: Mapping[str, tuple[str, tuple[str, ...]]] | None = None,
) -> SessionReadiness:
    """Say whether this closed session can be evaluated. Do not evaluate it."""
    close = session_close(session)
    _require_utc(as_of, "as_of")
    blocks: set[str] = set()
    notes: set[str] = set()
    if as_of < close:
        blocks.add("session_not_closed")
    symbols = _snapshot_symbols(snapshot, session, close, blocks)
    finished = {} if settled is None else settled
    _bars(
        symbols,
        session,
        bars,
        bar_months_present,
        {} if bar_absences is None else bar_absences,
        blocks,
        notes,
        finished,
    )
    _candidates(symbols, close, candidates, candidate_files_present, blocks, notes, finished)
    _regime(session, close, observations, regime_failures, blocks)
    if not weights_present:
        notes.add("score_weights_not_frozen")
    return SessionReadiness(
        ready=not blocks,
        blocks=tuple(sorted(blocks)),
        decision_notes=tuple(sorted(notes)),
        score_weights="present" if weights_present else "absent",
    )


def _require_utc(moment: datetime, label: str) -> None:
    if moment.tzinfo is None or moment.utcoffset() != timedelta(0):
        raise EvaluationError(f"{label} must be timezone-aware UTC")


def _snapshot_symbols(
    snapshot: UniverseSnapshot | None,
    session: date,
    close: datetime,
    blocks: set[str],
) -> tuple[str, ...]:
    if snapshot is None:
        blocks.add("snapshot_not_produced")
        return ()
    _require_utc(snapshot.observed_at, "observed_at")
    if snapshot.session != session:
        blocks.add("snapshot_session_mismatch")
        return ()
    if snapshot.observed_at > close:
        blocks.add("snapshot_lookahead")
    if len(snapshot.symbols) != len(set(snapshot.symbols)):
        blocks.add("snapshot_repeats_symbol")
        return ()
    return snapshot.symbols


def _bars(
    symbols: tuple[str, ...],
    session: date,
    bars: Mapping[str, Sequence[DailyBar]],
    months_present: Mapping[str, bool],
    absences: Mapping[str, bool],
    blocks: set[str],
    notes: set[str],
    settled: Mapping[str, tuple[str, tuple[str, ...]]],
) -> None:
    if _BTC not in settled:
        _one_symbol_bars(_BTC, session, bars, months_present, absences, blocks, notes, btc=True)
    for symbol in symbols:
        if symbol == _BTC or symbol in settled:
            continue
        _one_symbol_bars(symbol, session, bars, months_present, absences, blocks, notes, btc=False)


def _one_symbol_bars(
    symbol: str,
    session: date,
    bars: Mapping[str, Sequence[DailyBar]],
    months_present: Mapping[str, bool],
    absences: Mapping[str, bool],
    blocks: set[str],
    notes: set[str],
    *,
    btc: bool,
) -> None:
    series = bars.get(symbol, ())
    if any(bar.symbol != symbol for bar in series):
        blocks.add(f"bar_symbol_mismatch:{symbol}")
    dates = [bar.open_date for bar in series]
    if len(dates) != len(set(dates)):
        blocks.add(f"duplicate_bar_date:{symbol}")
    if any(bar.open_date > session for bar in series):
        blocks.add(f"bars_lookahead:{symbol}")
    if not months_present.get(symbol, False):
        if not btc and absences.get(symbol, False):
            notes.add(f"daily_bar_absent:{symbol}")
            return
        blocks.add("btc_bars_not_produced" if btc else f"bars_not_produced:{symbol}")
        return
    if any(bar.open_date == session for bar in series):
        return
    if btc:
        blocks.add("btc_session_bar_absent")
    else:
        notes.add(f"daily_bar_absent:{symbol}")


def _candidates(
    symbols: tuple[str, ...],
    close: datetime,
    candidates: Mapping[str, ScanCandidate | None],
    files_present: Mapping[str, bool],
    blocks: set[str],
    notes: set[str],
    settled: Mapping[str, tuple[str, tuple[str, ...]]],
) -> None:
    for symbol in symbols:
        recorded = settled.get(symbol)
        if recorded is not None:
            for code in recorded[1]:
                label = "evidence_blocked" if code.startswith("missing_") else "policy_rejected"
                notes.add(f"{label}:{symbol}:{code}")
        if recorded is not None and not files_present.get(symbol, False):
            continue
        if not files_present.get(symbol, False):
            blocks.add(f"candidate_not_produced:{symbol}")
            continue
        packet = candidates.get(symbol)
        if packet is None:
            notes.add(f"missing_candidate:{symbol}")
            continue
        if packet.facts.symbol != symbol:
            blocks.add(f"candidate_symbol_mismatch:{symbol}")
        if packet.market.as_of > close:
            blocks.add(f"candidate_lookahead:{symbol}")
        for cap, stamp in (
            (packet.coingecko.market_cap_usd, packet.coingecko.source_timestamp),
            (packet.cmc.market_cap_usd, packet.cmc.source_timestamp),
        ):
            if cap is not None and stamp is None:
                blocks.add(f"undated_fundamental:{symbol}")
            elif stamp is not None and stamp > close:
                blocks.add(f"candidate_lookahead:{symbol}")


def _regime(
    session: date,
    close: datetime,
    observations: Sequence[Observation],
    failures: frozenset[str],
    blocks: set[str],
) -> None:
    for series in _REGIME:
        if series in failures:
            blocks.add(f"regime_failed:{series}")
            continue
        matched = [
            item
            for item in observations
            if item.series == series and item.observed_at.astimezone(UTC).date() == session
        ]
        later = [
            item
            for item in observations
            if item.series == series and item.observed_at.astimezone(UTC).date() > session
        ]
        source_after_close = any(
            item.source_timestamp is not None and item.source_timestamp > close for item in matched
        )
        if later or source_after_close:
            blocks.add(f"regime_lookahead:{series}")
        elif not matched:
            blocks.add(f"regime_not_produced:{series}")
