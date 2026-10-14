"""Turn closed-session captures into session inputs and freeze a ready session.

The daily scan is not called. A universe, packet, or regime value retrieved
after the close is not stored. A completed daily bar may be retrieved after
the close, because its period ends at that instant. A later bar is not stored.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from cip.domain.errors import EvaluationError
from cip.evaluation.inputs import (
    AbsentInput,
    load_session,
    write_absence,
    write_candidate,
    write_universe,
)
from cip.evaluation.scan import ScanCandidate, UniverseSnapshot
from cip.evaluation.session import SessionReadiness, assess_session, session_close
from cip.history.bars import DailyBar
from cip.recorders.observation import CollectionFailure, Observation

TRANSFORMATION = "session-input/v1"
BLOCKED_THROUGH = date(2026, 10, 5)
_SYMBOL = re.compile(r"^[A-Z0-9]{1,20}$")
_REGIME = ("btc_dominance", "stablecoin_supply")
_CORRUPT_EXACT = frozenset(
    {"snapshot_lookahead", "snapshot_session_mismatch", "snapshot_repeats_symbol"}
)
_CORRUPT_PREFIX = (
    "bars_lookahead:",
    "bar_symbol_mismatch:",
    "duplicate_bar_date:",
    "candidate_lookahead:",
    "candidate_symbol_mismatch:",
    "undated_fundamental:",
    "regime_lookahead:",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class CaptureClock:
    """Market time and the time CIP retrieved the value.

    ``source_timestamp`` is the provider's period or observation time.
    ``captured_at`` is when CIP stored the retrieval. For a completed daily bar,
    retrieval may follow the session close. For every other input it may not.
    """

    source: str
    source_timestamp: datetime | None
    captured_at: datetime


@dataclass(frozen=True)
class UniverseCapture:
    symbols: tuple[str, ...]
    observed_at: datetime
    clock: CaptureClock


@dataclass(frozen=True)
class PacketCapture:
    """Evidence for one snapshot symbol. The symbol is the filing key."""

    symbol: str
    candidate: ScanCandidate
    clock: CaptureClock


@dataclass(frozen=True)
class AbsenceCapture:
    """The producer looked. ``produced_at`` may follow the close."""

    symbol: str
    produced_at: datetime
    source: str


@dataclass(frozen=True)
class BarCapture:
    """One symbol's daily bars.

    The bar period is ``open_date``. ``clock.source_timestamp`` is that period's
    market time when the provider stamps one. ``clock.captured_at`` is the
    retrieval. A series that contains the session bar may be retrieved after
    the close. A bar from a later date is still lookahead.
    """

    symbol: str
    bars: tuple[DailyBar, ...]
    clock: CaptureClock


@dataclass(frozen=True)
class RegimeCapture:
    observation: Observation
    captured_at: datetime


@dataclass(frozen=True)
class FailureCapture:
    failure: CollectionFailure
    captured_at: datetime


@dataclass(frozen=True)
class SessionCaptures:
    """Captures already in hand. This object does not fetch them."""

    universe: UniverseCapture | None
    packets: tuple[PacketCapture, ...]
    candidate_absences: tuple[AbsenceCapture, ...]
    bars: tuple[BarCapture, ...]
    bar_absences: tuple[AbsenceCapture, ...]
    regime: tuple[RegimeCapture, ...]
    regime_failures: tuple[FailureCapture, ...]


class ProvenanceEntry(_Strict):
    name: str
    source: str
    source_timestamp: datetime | None
    captured_at: datetime
    session_close_at: datetime
    transformation: Literal["session-input/v1"]


class SessionManifest(_Strict):
    """The exact inputs that made one session ready. Later runs do not rewrite it."""

    session_date: date
    session_close_at: datetime
    finalized_at: datetime
    input_manifest_version: Literal[1]
    universe_snapshot_sha256: str
    bars_manifest_sha256: str
    candidate_manifest_sha256: str
    regime_inputs_sha256: str
    provenance: tuple[ProvenanceEntry, ...]
    git_sha: str
    score_weights: Literal["absent", "present"]


@dataclass(frozen=True)
class PopulationResult:
    readiness: SessionReadiness
    manifest: SessionManifest | None


@dataclass
class _Plan:
    snapshot: UniverseSnapshot | None
    bars: dict[str, tuple[DailyBar, ...]]
    candidates: dict[str, ScanCandidate | None]
    files_present: dict[str, bool]
    bar_absences: dict[str, bool]
    observations: tuple[Observation, ...]
    failures: frozenset[str]
    provenance: tuple[ProvenanceEntry, ...]
    documents: dict[str, bytes]
    absences: tuple[AbsentInput, ...]


def populate_session(
    root: Path,
    session: date,
    as_of: datetime,
    captures: SessionCaptures,
    *,
    git_sha: str,
    weights_present: bool,
) -> PopulationResult:
    """Store the closed slice and assess it. Finalize only when it is ready."""
    close = session_close(session)
    _require_utc(as_of, "as_of")
    if not git_sha:
        raise EvaluationError("git sha is missing")
    plan = _plan(session, close, captures)
    _attach_stage(root, session, plan)
    readiness = _assess(root, session, as_of, plan, weights_present)
    path = root / _manifest_path(session)
    if path.is_file():
        return _kept(root, session, plan, path, as_of, git_sha, weights_present)
    if "session_not_closed" in readiness.blocks or _corrupt(readiness.blocks):
        return PopulationResult(readiness, None)
    _write(root, session, plan)
    stored = _assess_stored(root, session, as_of, weights_present)
    if not stored.ready:
        return PopulationResult(stored, None)
    manifest = _seal(root, session, as_of, git_sha, weights_present, plan.provenance)
    return PopulationResult(stored, manifest)


def replay_session(root: Path, session: date) -> PopulationResult:
    """Assess a finalized session again from the frozen files."""
    session_close(session)
    path = root / _manifest_path(session)
    if not path.is_file():
        raise EvaluationError("session is not finalized")
    manifest = _read_manifest(path)
    if manifest.session_date != session or _hashes(root, session) != _hash_record(manifest):
        raise EvaluationError("finalized session does not match its inputs")
    weights_present = manifest.score_weights == "present"
    readiness = _assess_stored(root, session, manifest.finalized_at, weights_present)
    return PopulationResult(readiness, manifest)


def _plan(session: date, close: datetime, captures: SessionCaptures) -> _Plan:
    provenance: list[ProvenanceEntry] = []
    documents: dict[str, bytes] = {}
    snapshot = _universe(session, close, captures.universe, provenance)
    bars, bar_flags, bar_absences = _bars(session, close, captures, provenance, documents)
    candidates, present, candidate_absences = _candidates(session, close, captures, provenance)
    observations, failures = _regime(session, close, captures, provenance, documents)
    return _Plan(
        snapshot,
        bars,
        candidates,
        present,
        bar_flags,
        observations,
        failures,
        tuple(sorted(provenance, key=lambda item: item.name)),
        documents,
        tuple(bar_absences + candidate_absences),
    )


def _assess(
    root: Path, session: date, as_of: datetime, plan: _Plan, weights_present: bool
) -> SessionReadiness:
    from cip.evaluation.stage import settlements

    symbols = () if plan.snapshot is None else plan.snapshot.symbols
    return assess_session(
        session,
        as_of,
        snapshot=plan.snapshot,
        bars=plan.bars,
        bar_months_present={symbol: True for symbol in plan.bars},
        candidates=plan.candidates,
        candidate_files_present=plan.files_present,
        observations=plan.observations,
        regime_failures=plan.failures,
        weights_present=weights_present,
        bar_absences=plan.bar_absences,
        settled=settlements(root, session, symbols),
    )


def _assess_stored(
    root: Path, session: date, as_of: datetime, weights_present: bool
) -> SessionReadiness:
    from cip.evaluation.stage import settlements

    recorded = load_session(root, session)
    bars = _load_bars(root, session)
    observations, failures = _load_regime(root, session)
    symbols = () if recorded.snapshot is None else recorded.snapshot.symbols
    frozen = root / _stage_dir(session, "classification")
    return assess_session(
        session,
        as_of,
        snapshot=recorded.snapshot,
        bars=bars,
        bar_months_present={symbol: True for symbol in bars},
        candidates=recorded.candidates,
        candidate_files_present=recorded.candidate_files_present,
        observations=observations,
        regime_failures=failures,
        weights_present=weights_present,
        bar_absences=recorded.bar_absences,
        settled=settlements(root, session, symbols, sealed=True) if frozen.is_dir() else {},
    )


def _kept(
    root: Path,
    session: date,
    plan: _Plan,
    path: Path,
    as_of: datetime,
    git_sha: str,
    weights_present: bool,
) -> PopulationResult:
    """Return the first freeze, or refuse before any new file is created."""
    existing = _read_manifest(path)
    _attach_stage(root, session, plan)
    built = _planned_manifest(session, as_of, git_sha, weights_present, plan)
    if not _same_freeze(existing, built):
        raise EvaluationError("finalized session already exists with a different payload")
    readiness = _assess(
        root, session, existing.finalized_at, plan, existing.score_weights == "present"
    )
    return PopulationResult(readiness, existing)


def _write(root: Path, session: date, plan: _Plan) -> None:
    if plan.snapshot is not None:
        write_universe(root, plan.snapshot)
    for candidate in plan.candidates.values():
        if candidate is not None:
            write_candidate(root, session, candidate)
    for absence in plan.absences:
        write_absence(root, absence)
    for name, body in sorted(plan.documents.items()):
        _create(root / name, body)


def _seal(
    root: Path,
    session: date,
    as_of: datetime,
    git_sha: str,
    weights_present: bool,
    provenance: tuple[ProvenanceEntry, ...],
) -> SessionManifest:
    hashes = _hashes(root, session)
    built = _manifest_of(session, as_of, git_sha, weights_present, provenance, hashes)
    path = root / _manifest_path(session)
    if path.is_file():
        existing = _read_manifest(path)
        if _same_freeze(existing, built):
            return existing
        raise EvaluationError("finalized session already exists with a different payload")
    _create(path, _body(built))
    return built


def _planned_manifest(
    session: date,
    as_of: datetime,
    git_sha: str,
    weights_present: bool,
    plan: _Plan,
) -> SessionManifest:
    return _manifest_of(
        session, as_of, git_sha, weights_present, plan.provenance, _planned_hashes(plan)
    )


def _manifest_of(
    session: date,
    as_of: datetime,
    git_sha: str,
    weights_present: bool,
    provenance: tuple[ProvenanceEntry, ...],
    hashes: tuple[str, str, str, str],
) -> SessionManifest:
    universe, bars, candidates, regime = hashes
    return SessionManifest(
        session_date=session,
        session_close_at=session_close(session),
        finalized_at=as_of,
        input_manifest_version=1,
        universe_snapshot_sha256=universe,
        bars_manifest_sha256=bars,
        candidate_manifest_sha256=candidates,
        regime_inputs_sha256=regime,
        provenance=provenance,
        git_sha=git_sha,
        score_weights="present" if weights_present else "absent",
    )


def _universe(
    session: date,
    close: datetime,
    capture: UniverseCapture | None,
    provenance: list[ProvenanceEntry],
) -> UniverseSnapshot | None:
    if capture is None:
        return None
    _clock(capture.clock, close)
    _value_time(capture.observed_at, close, "observed_at")
    provenance.append(
        _entry(
            "universe",
            capture.clock.source,
            capture.clock.source_timestamp,
            capture.clock.captured_at,
            close,
        )
    )
    return UniverseSnapshot(
        session=session,
        symbols=capture.symbols,
        observed_at=capture.observed_at,
        provenance=_provenance_text(capture.clock),
    )


def _bars(
    session: date,
    close: datetime,
    captures: SessionCaptures,
    provenance: list[ProvenanceEntry],
    documents: dict[str, bytes],
) -> tuple[dict[str, tuple[DailyBar, ...]], dict[str, bool], list[AbsentInput]]:
    found: dict[str, tuple[DailyBar, ...]] = {}
    for series in captures.bars:
        if series.symbol in found:
            raise EvaluationError("conflicting bar evidence")
        _ticker(series.symbol)
        _bar_clock(session, close, series)
        ordered = tuple(sorted(series.bars, key=lambda bar: bar.open_date))
        found[series.symbol] = ordered
        provenance.append(
            _entry(
                f"bars:{series.symbol}",
                series.clock.source,
                series.clock.source_timestamp,
                series.clock.captured_at,
                close,
            )
        )
        documents[_bar_path(session, series.symbol).as_posix()] = _bar_body(
            session, close, series.symbol, ordered, series.clock
        )
    flags: dict[str, bool] = {}
    stored: list[AbsentInput] = []
    for item in captures.bar_absences:
        if item.symbol in found:
            raise EvaluationError("bar absence conflicts with stored bars")
        if item.symbol in flags:
            raise EvaluationError("conflicting absence evidence")
        _absence_clock(item)
        flags[item.symbol] = True
        provenance.append(
            _entry(f"bar_absence:{item.symbol}", item.source, None, item.produced_at, close)
        )
        stored.append(_absent(session, "daily_bar", item))
    return found, flags, stored


def _candidates(
    session: date,
    close: datetime,
    captures: SessionCaptures,
    provenance: list[ProvenanceEntry],
) -> tuple[dict[str, ScanCandidate | None], dict[str, bool], list[AbsentInput]]:
    found: dict[str, ScanCandidate | None] = {}
    present: dict[str, bool] = {}
    stored: list[AbsentInput] = []
    for packet in captures.packets:
        if packet.symbol in found:
            raise EvaluationError("conflicting candidate evidence")
        _clock(packet.clock, close)
        _value_time(packet.candidate.market.as_of, close, "market clock")
        _value_time(packet.candidate.coingecko.source_timestamp, close, "source_timestamp")
        _value_time(packet.candidate.cmc.source_timestamp, close, "source_timestamp")
        found[packet.symbol] = packet.candidate
        present[packet.symbol] = True
        provenance.append(
            _entry(
                f"candidate:{packet.symbol}",
                packet.clock.source,
                packet.clock.source_timestamp,
                packet.clock.captured_at,
                close,
            )
        )
    for item in captures.candidate_absences:
        if item.symbol in found:
            if found[item.symbol] is None:
                raise EvaluationError("conflicting absence evidence")
            raise EvaluationError("candidate absence conflicts with a packet")
        _absence_clock(item)
        found[item.symbol] = None
        present[item.symbol] = True
        provenance.append(
            _entry(f"candidate_absence:{item.symbol}", item.source, None, item.produced_at, close)
        )
        stored.append(_absent(session, "candidate", item))
    return found, present, stored


def _regime(
    session: date,
    close: datetime,
    captures: SessionCaptures,
    provenance: list[ProvenanceEntry],
    documents: dict[str, bytes],
) -> tuple[tuple[Observation, ...], frozenset[str]]:
    seen: set[str] = set()
    observations: list[Observation] = []
    failures: list[str] = []
    for item in captures.regime:
        series = item.observation.series
        # Funding and open interest are not session inputs, even when captured late.
        if series not in _REGIME:
            continue
        _value_time(item.captured_at, close, "captured_at")
        _value_time(item.observation.observed_at, close, "observed_at")
        _value_time(item.observation.source_timestamp, close, "source_timestamp")
        observed = item.observation.observed_at.astimezone(UTC).date()
        if observed < session:
            continue
        _claim(seen, series)
        observations.append(item.observation)
        if observed > session:
            continue
        provenance.append(
            _entry(
                f"regime:{series}",
                item.observation.provider,
                item.observation.source_timestamp,
                item.captured_at,
                close,
            )
        )
        documents[_regime_path(session, series).as_posix()] = _regime_body(session, close, item)
    for failure in captures.regime_failures:
        series = failure.failure.series
        if series not in _REGIME:
            continue
        _value_time(failure.captured_at, close, "captured_at")
        _value_time(failure.failure.observed_at, close, "observed_at")
        if failure.failure.observed_at.astimezone(UTC).date() != session:
            continue
        _claim(seen, series)
        failures.append(series)
        provenance.append(
            _entry(
                f"regime_failure:{series}",
                failure.failure.provider,
                None,
                failure.captured_at,
                close,
            )
        )
        body = _failure_body(session, close, failure)
        documents[_failure_path(session, series).as_posix()] = body
    return tuple(observations), frozenset(failures)


def _ticker(symbol: str) -> str:
    if _SYMBOL.fullmatch(symbol) is None:
        raise EvaluationError("symbol must be 1 to 20 uppercase letters or digits")
    return symbol


def _claim(seen: set[str], series: str) -> None:
    if series in seen:
        raise EvaluationError("conflicting regime evidence")
    seen.add(series)


def _absent(
    session: date, kind: Literal["candidate", "daily_bar"], item: AbsenceCapture
) -> AbsentInput:
    return AbsentInput(
        kind="absent_input",
        session=session,
        input=kind,
        symbol=item.symbol,
        produced_at=item.produced_at,
    )


def _corrupt(blocks: tuple[str, ...]) -> bool:
    return any(block in _CORRUPT_EXACT or block.startswith(_CORRUPT_PREFIX) for block in blocks)


def _bar_clock(session: date, close: datetime, series: BarCapture) -> None:
    """Accept a completed session bar retrieved after the close.

    History that does not contain the session bar must still have been retrieved
    at or before the close. The session bar's period ends at the close, so a
    retrieval before that instant is not evidence of the completed bar.
    Sessions through 2026-10-05 keep the stricter rule: a retrieval after the
    close is not evidence for that session.
    """
    if series.clock.source == "":
        raise EvaluationError("source is required")
    _require_utc(series.clock.captured_at, "captured_at")
    _value_time(series.clock.source_timestamp, close, "source_timestamp")
    includes_session = False
    for bar in series.bars:
        if bar.open_date > session:
            raise EvaluationError("capture is after the close")
        if bar.open_date == session:
            includes_session = True
    if includes_session and session > BLOCKED_THROUGH:
        if series.clock.captured_at < close:
            raise EvaluationError("session bar was retrieved before it closed")
        return
    if series.clock.captured_at > close:
        raise EvaluationError("capture is after the close")


def _clock(clock: CaptureClock, close: datetime) -> None:
    if clock.source == "":
        raise EvaluationError("source is required")
    _value_time(clock.captured_at, close, "captured_at")
    _value_time(clock.source_timestamp, close, "source_timestamp")


def _absence_clock(item: AbsenceCapture) -> None:
    if item.source == "":
        raise EvaluationError("source is required")
    _require_utc(item.produced_at, "produced_at")


def _value_time(moment: datetime | None, close: datetime, label: str) -> None:
    if moment is None:
        return
    _require_utc(moment, label)
    if moment > close:
        raise EvaluationError("capture is after the close")


def _require_utc(moment: datetime, label: str) -> None:
    if moment.tzinfo is None or moment.utcoffset() != timedelta(0):
        raise EvaluationError(f"{label} must be timezone-aware UTC")


def _entry(
    name: str,
    source: str,
    source_timestamp: datetime | None,
    captured_at: datetime,
    close: datetime,
) -> ProvenanceEntry:
    return ProvenanceEntry(
        name=name,
        source=source,
        source_timestamp=source_timestamp,
        captured_at=captured_at,
        session_close_at=close,
        transformation=TRANSFORMATION,
    )


def _provenance_text(clock: CaptureClock) -> str:
    stamp = "none" if clock.source_timestamp is None else _iso(clock.source_timestamp)
    return (
        f"{clock.source} source_timestamp={stamp} "
        f"captured_at={_iso(clock.captured_at)} transformation={TRANSFORMATION}"
    )


def _bar_body(
    session: date, close: datetime, symbol: str, bars: tuple[DailyBar, ...], clock: CaptureClock
) -> bytes:
    document = {
        "bars": [_bar_row(bar) for bar in bars],
        "captured_at": _iso(clock.captured_at),
        "session": session.isoformat(),
        "session_close_at": _iso(close),
        "source": clock.source,
        "source_timestamp": None
        if clock.source_timestamp is None
        else _iso(clock.source_timestamp),
        "symbol": symbol,
        "transformation": TRANSFORMATION,
    }
    return json.dumps(document, sort_keys=True).encode()


def _bar_row(bar: DailyBar) -> dict[str, object]:
    return {
        "close": format(bar.close, "f"),
        "high": format(bar.high, "f"),
        "low": format(bar.low, "f"),
        "open": format(bar.open, "f"),
        "open_date": bar.open_date.isoformat(),
        "quote_volume": format(bar.quote_volume, "f"),
        "symbol": bar.symbol,
        "taker_buy_base_volume": format(bar.taker_buy_base_volume, "f"),
        "taker_buy_quote_volume": format(bar.taker_buy_quote_volume, "f"),
        "trade_count": bar.trade_count,
        "volume": format(bar.volume, "f"),
    }


def _regime_body(session: date, close: datetime, item: RegimeCapture) -> bytes:
    observation = item.observation
    document = {
        "captured_at": _iso(item.captured_at),
        "observed_at": _iso(observation.observed_at),
        "provider": observation.provider,
        "series": observation.series,
        "session": session.isoformat(),
        "session_close_at": _iso(close),
        "source_timestamp": None
        if observation.source_timestamp is None
        else _iso(observation.source_timestamp),
        "symbol": observation.symbol,
        "transformation": TRANSFORMATION,
        "units": dict(observation.units),
        "values": {name: format(value, "f") for name, value in observation.values},
    }
    return json.dumps(document, sort_keys=True).encode()


def _failure_body(session: date, close: datetime, item: FailureCapture) -> bytes:
    failure = item.failure
    document = {
        "captured_at": _iso(item.captured_at),
        "error": failure.error,
        "observed_at": _iso(failure.observed_at),
        "provider": failure.provider,
        "series": failure.series,
        "session": session.isoformat(),
        "session_close_at": _iso(close),
        "symbol": failure.symbol,
        "transformation": TRANSFORMATION,
    }
    return json.dumps(document, sort_keys=True).encode()


def _load_bars(root: Path, session: date) -> dict[str, tuple[DailyBar, ...]]:
    found: dict[str, tuple[DailyBar, ...]] = {}
    for path in _files(root / _bar_path(session, "BTCUSDT").parent):
        document = _object(path)
        symbol = document.get("symbol")
        rows = document.get("bars")
        if not isinstance(symbol, str) or not isinstance(rows, list):
            raise EvaluationError("session input is unusable")
        found[symbol] = tuple(_load_bar(row) for row in rows)
    return found


def _load_bar(row: object) -> DailyBar:
    if not isinstance(row, dict):
        raise EvaluationError("session input is unusable")
    try:
        return DailyBar(
            symbol=row["symbol"],
            open_date=date.fromisoformat(row["open_date"]),
            open=_decimal(row["open"]),
            high=_decimal(row["high"]),
            low=_decimal(row["low"]),
            close=_decimal(row["close"]),
            volume=_decimal(row["volume"]),
            quote_volume=_decimal(row["quote_volume"]),
            trade_count=row["trade_count"],
            taker_buy_base_volume=_decimal(row["taker_buy_base_volume"]),
            taker_buy_quote_volume=_decimal(row["taker_buy_quote_volume"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError("session input is unusable") from error


def _load_regime(root: Path, session: date) -> tuple[tuple[Observation, ...], frozenset[str]]:
    observations: list[Observation] = []
    failures: list[str] = []
    for path in _files(root / _regime_path(session, "btc_dominance").parent):
        document = _object(path)
        if path.name.endswith(".failure.json"):
            series = document.get("series")
            if not isinstance(series, str):
                raise EvaluationError("session input is unusable")
            failures.append(series)
            continue
        observations.append(_load_observation(document))
    return tuple(observations), frozenset(failures)


def _load_observation(document: Mapping[str, object]) -> Observation:
    try:
        values = document["values"]
        units = document["units"]
        if not isinstance(values, dict) or not isinstance(units, dict):
            raise TypeError
        symbol = document["symbol"]
        return Observation(
            series=str(document["series"]),
            provider=str(document["provider"]),
            source_timestamp=_optional_time(document["source_timestamp"]),
            observed_at=_parse_time(document["observed_at"]),
            symbol=None if symbol is None else str(symbol),
            values=tuple((str(name), _decimal(value)) for name, value in values.items()),
            units=tuple((str(name), str(unit)) for name, unit in units.items()),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError("session input is unusable") from error


def _planned_hashes(plan: _Plan) -> tuple[str, str, str, str]:
    universe = _sha(b"" if plan.snapshot is None else _body(plan.snapshot))
    bars: list[tuple[str, bytes]] = []
    candidates: list[tuple[str, bytes]] = []
    regime: list[tuple[str, bytes]] = []
    for name, body in plan.documents.items():
        filename = Path(name).name
        if "bars" in Path(name).parts:
            bars.append((f"bars/{filename}", body))
        elif "classification" in Path(name).parts or "lane" in Path(name).parts:
            candidates.append((f"{Path(name).parent.name}/{filename}", body))
        else:
            regime.append((f"regime/{filename}", body))
    for absence in plan.absences:
        label = "bar-absence" if absence.input == "daily_bar" else "candidate-absence"
        bucket = bars if absence.input == "daily_bar" else candidates
        bucket.append((f"{label}/symbol={absence.symbol}.json", _body(absence)))
    for symbol, candidate in plan.candidates.items():
        if candidate is not None:
            candidates.append((f"candidates/symbol={symbol}.json", _body(candidate)))
    return (universe, _bundle(bars), _bundle(candidates), _bundle(regime))


def _hashes(root: Path, session: date) -> tuple[str, str, str, str]:
    universe = root / _universe_path(session)
    if not universe.is_file():
        raise EvaluationError("finalized session does not match its inputs")
    bars = _named(root / _bar_path(session, "BTCUSDT").parent, "bars")
    bars.extend(_named(root / _absence_dir(session, "daily_bar"), "bar-absence"))
    candidates = _named(root / _candidate_dir(session), "candidates")
    candidates.extend(_named(root / _absence_dir(session, "candidate"), "candidate-absence"))
    candidates.extend(_named(root / _stage_dir(session, "classification"), "classification"))
    candidates.extend(_named(root / _stage_dir(session, "lane"), "lane"))
    regime = _named(root / _regime_path(session, "btc_dominance").parent, "regime")
    return (_sha(universe.read_bytes()), _bundle(bars), _bundle(candidates), _bundle(regime))


def _hash_record(manifest: SessionManifest) -> tuple[str, str, str, str]:
    return (
        manifest.universe_snapshot_sha256,
        manifest.bars_manifest_sha256,
        manifest.candidate_manifest_sha256,
        manifest.regime_inputs_sha256,
    )


def _named(directory: Path, label: str) -> list[tuple[str, bytes]]:
    return [(f"{label}/{path.name}", path.read_bytes()) for path in _files(directory)]


def _files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and not path.name.startswith(".")
    )


def _bundle(items: list[tuple[str, bytes]]) -> str:
    payload = {name: _sha(body) for name, body in sorted(items)}
    return _sha(json.dumps(payload, sort_keys=True).encode())


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _same_freeze(existing: SessionManifest, built: SessionManifest) -> bool:
    left = existing.model_dump(mode="json")
    right = built.model_dump(mode="json")
    for key in ("finalized_at", "git_sha"):
        left.pop(key)
        right.pop(key)
    return left == right


def _read_manifest(path: Path) -> SessionManifest:
    try:
        return SessionManifest.model_validate(_object(path))
    except ValidationError as error:
        raise EvaluationError("session input is unusable") from error


def _object(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError("session input is unusable") from error
    if not isinstance(document, dict):
        raise EvaluationError("session input is unusable")
    return document


def _body(model: BaseModel) -> bytes:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True).encode()


def _create(path: Path, body: bytes) -> bool:
    if path.exists():
        return _same(path, body)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(body)
        try:
            os.link(temporary, path)
        except FileExistsError:
            return _same(path, body)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _same(path: Path, body: bytes) -> bool:
    if path.read_bytes() == body:
        return False
    raise EvaluationError(f"session input {path.name} already exists with a different payload")


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, str):
        raise EvaluationError("session input is unusable")
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise EvaluationError("session input is unusable") from error


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise EvaluationError("session input is unusable")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvaluationError("session input is unusable") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise EvaluationError("session input is unusable")
    return parsed.astimezone(UTC)


def _optional_time(value: object) -> datetime | None:
    if value is None:
        return None
    return _parse_time(value)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _universe_path(session: date) -> Path:
    return _session_dir(session) / "universe.json"


def _bar_path(session: date, symbol: str) -> Path:
    return _session_dir(session) / "bars" / f"symbol={symbol}.json"


def _candidate_dir(session: date) -> Path:
    return _session_dir(session) / "candidates"


def _stage_dir(session: date, label: str) -> Path:
    return _session_dir(session) / label


def _attach_stage(root: Path, session: date, plan: _Plan) -> None:
    """Freeze the classification and lane files that explain a terminal stage."""
    for label in ("classification", "lane"):
        source = root / "captures" / f"session={session.isoformat()}" / label
        for path in _files(source):
            plan.documents[(_stage_dir(session, label) / path.name).as_posix()] = path.read_bytes()


def _absence_dir(session: date, kind: str) -> Path:
    return _session_dir(session) / "absences" / f"input={kind}"


def _regime_path(session: date, series: str) -> Path:
    return _session_dir(session) / "regime" / f"{series}.json"


def _failure_path(session: date, series: str) -> Path:
    return _session_dir(session) / "regime" / f"{series}.failure.json"


def _manifest_path(session: date) -> Path:
    return _session_dir(session) / "manifest.json"


def _session_dir(session: date) -> Path:
    return Path("sessions") / f"date={session.isoformat()}"
