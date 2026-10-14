"""Accumulate one UTC session's evidence. The daily scan is not called.

The session comes from the clock. A scheduled payload cannot name a future
session. A temporary provider failure is left unstored while a retry remains
possible. One stored book is the session's only spread snapshot.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

from cip.adapters.market import Depth, DepthLevel
from cip.domain.errors import EvaluationError
from cip.domain.policy import UniverseHypotheses, load_policy
from cip.evaluation.classification import SymbolClassification, store_classification
from cip.evaluation.eligibility import Lane
from cip.evaluation.lane_capture import ListingUnavailable, capture_lanes
from cip.evaluation.liquidity_capture import (
    HourQuote,
    book_metrics,
    build_liquidity,
    spike_candle_count,
    store_liquidity,
)
from cip.evaluation.populate import (
    BLOCKED_THROUGH,
    RegimeCapture,
    UniverseCapture,
    populate_session,
)
from cip.evaluation.prospect import (
    load_closed,
    prospective_session,
    session_bar_capture,
    store_completed_bars,
    store_regime,
    store_universe,
)
from cip.evaluation.session import SessionReadiness, session_close
from cip.evaluation.stage import load_symbol, load_universe, needs_lane, settle
from cip.history.bars import DailyBar

_SEALED = date(2026, 10, 6)
_REGIME = ("btc_dominance", "stablecoin_supply")


class TemporaryFailure(Exception):
    """The provider did not answer. This is not evidence that the input is absent."""


class CaptureSource(Protocol):
    """One retrieval. The cycle decides whether it is allowed to ask."""

    def fetch(self, session: date, kind: str, symbol: str | None) -> object: ...


@dataclass(frozen=True)
class CaptureCycle:
    """The open session, and the previous session once it has closed."""

    now: datetime
    open_session: date
    pre_close: date | None
    post_close: date | None


@dataclass(frozen=True)
class BookShot:
    """One order-book observation. A session stores one of these."""

    spread_bps: Decimal
    source_timestamp: datetime | None
    captured_at: datetime
    bids: tuple[tuple[Decimal, Decimal], ...] = ()
    asks: tuple[tuple[Decimal, Decimal], ...] = ()


@dataclass(frozen=True)
class CycleReport:
    cycle: CaptureCycle
    calls: tuple[tuple[str, str | None], ...]
    skipped: tuple[str, ...]
    retries: tuple[str, ...]
    spike_counts: tuple[tuple[str, int | None], ...]
    readiness: SessionReadiness | None
    finalized: bool


def capture_cycle(now: datetime, requested: date | None = None) -> CaptureCycle:
    """Name the sessions this clock may touch. A future date is refused."""
    _utc(now, "as_of")
    if requested is not None and type(requested) is not date:
        raise EvaluationError("session is a date")
    opened = now.astimezone(UTC).date()
    pre = opened if _writable(opened) else None
    previous = opened - timedelta(days=1)
    post = previous if _writable(previous) and now >= session_close(previous) else None
    if requested is not None:
        if requested > opened:
            raise EvaluationError("future session")
        if requested == _SEALED:
            raise EvaluationError("sealed session stays sealed")
        if requested != pre and requested != post:
            raise EvaluationError("session is not open")
        pre = requested if requested == pre else None
        post = requested if requested == post else None
    return CaptureCycle(now, opened, pre, post)


def book_is_stored(root: Path, session: date, symbol: str) -> bool:
    """True when this session already holds one spread snapshot for the symbol."""
    if _book_path(root, session, symbol).is_file():
        return True
    path = _liquidity_path(root, session, symbol)
    if not path.is_file():
        return False
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        return False
    snapshots = document.get("spread_snapshots")
    spreads = document.get("spread_bps")
    if isinstance(snapshots, int) and not isinstance(snapshots, bool) and snapshots >= 1:
        return True
    return isinstance(spreads, list) and len(spreads) >= 1


def store_session_book(root: Path, session: date, symbol: str, book: BookShot) -> None:
    """Store one pre-close book. A different second book is a conflict."""
    _belongs(session, book.captured_at)
    prospective_session(session)
    if session == _SEALED:
        raise EvaluationError("sealed session stays sealed")
    if _manifest(root, session).is_file():
        raise EvaluationError("finalized session is sealed")
    close = session_close(session)
    _utc(book.captured_at, "captured_at")
    if book.captured_at > close:
        raise EvaluationError("capture is after the close")
    if book.source_timestamp is not None:
        _utc(book.source_timestamp, "source_timestamp")
        if book.source_timestamp > close:
            raise EvaluationError("capture is after the close")
    if book_is_stored(root, session, symbol) and not _book_path(root, session, symbol).is_file():
        raise EvaluationError("conflicting observation")
    _create(_book_path(root, session, symbol), _book_body(session, symbol, book))


def run_capture_cycle(
    root: Path,
    now: datetime,
    source: CaptureSource,
    symbols: tuple[str, ...],
    *,
    requested: date | None = None,
    git_sha: str = "0" * 40,
) -> CycleReport:
    """Capture the open session, then the session that just closed."""
    bind = getattr(source, "bind_store", None)
    if callable(bind):
        bind(root)
    cycle = capture_cycle(now, requested)
    calls: list[tuple[str, str | None]] = []
    skipped: list[str] = []
    retries: list[str] = []
    spikes: list[tuple[str, int | None]] = []
    if cycle.pre_close is not None and _manifest(root, cycle.pre_close).is_file():
        skipped.append(f"finalized:{cycle.pre_close.isoformat()}")
    elif cycle.pre_close is not None:
        _pre_close(root, cycle.pre_close, cycle.now, source, symbols, calls, skipped, retries)
    if cycle.post_close is not None and _manifest(root, cycle.post_close).is_file():
        skipped.append(f"finalized:{cycle.post_close.isoformat()}")
    elif cycle.post_close is not None:
        _post_close(root, cycle, source, symbols, calls, skipped, retries, spikes)
    readiness, finalized = _finalize(root, cycle, git_sha)
    return CycleReport(
        cycle,
        tuple(calls),
        tuple(skipped),
        tuple(retries),
        tuple(spikes),
        readiness,
        finalized,
    )


def _writable(session: date) -> bool:
    return session > BLOCKED_THROUGH and session != _SEALED


def _belongs(session: date, captured_at: datetime) -> None:
    _utc(captured_at, "captured_at")
    if session > captured_at.astimezone(UTC).date():
        raise EvaluationError("future session")


def _pre_close(
    root: Path,
    session: date,
    now: datetime,
    source: CaptureSource,
    symbols: tuple[str, ...],
    calls: list[tuple[str, str | None]],
    skipped: list[str],
    retries: list[str],
) -> None:
    _one(root, session, source, "universe", None, calls, skipped, retries, _keep_universe)
    _one(root, session, source, "ticker", None, calls, skipped, retries, _keep_ticker)
    _one(root, session, source, "peg", "USDCUSDT", calls, skipped, retries, _keep_peg)
    for series in _REGIME:
        _one(root, session, source, "regime", series, calls, skipped, retries, _keep_regime)
    catalog_failed = False
    catalog_error: EvaluationError | None = None
    for symbol in _captured_symbols(root, session):
        if catalog_failed:
            retries.append(f"classification:{symbol}")
            continue
        before = len(retries)
        try:
            _one(
                root,
                session,
                source,
                "classification",
                symbol,
                calls,
                skipped,
                retries,
                _keep_classification,
            )
        except EvaluationError as error:
            catalog_error = error
            catalog_failed = True
            retries.append(f"classification:{symbol}")
            continue
        catalog_failed = len(retries) > before
    lane_error: EvaluationError | None = None
    universe = None if catalog_failed else _lane_universe()
    if universe is not None:
        lane_error = _lanes(
            root,
            session,
            now,
            source,
            _captured_symbols(root, session),
            universe,
            calls,
            retries,
        )
    for symbol in symbols:
        if book_is_stored(root, session, symbol):
            skipped.append(f"book:{symbol}")
            continue
        try:
            payload = source.fetch(session, "book", symbol)
        except TemporaryFailure:
            retries.append(f"book:{symbol}")
            continue
        calls.append(("book", symbol))
        if not isinstance(payload, BookShot):
            raise EvaluationError("capture is unusable")
        store_session_book(root, session, symbol, payload)
    if catalog_error is None and lane_error is None:
        _acquire_liquidity(root, session, now, source, calls, retries)
    if catalog_error is not None:
        raise catalog_error
    if lane_error is not None:
        raise lane_error


def _lanes(
    root: Path,
    session: date,
    now: datetime,
    source: CaptureSource,
    symbols: tuple[str, ...],
    universe: UniverseHypotheses,
    calls: list[tuple[str, str | None]],
    retries: list[str],
) -> EvaluationError | None:
    def fetch_listing(symbol: str) -> object:
        try:
            payload = source.fetch(session, "listing", symbol)
        except TemporaryFailure as error:
            raise ListingUnavailable(str(error)) from error
        calls.append(("listing", symbol))
        return payload

    try:
        pending = capture_lanes(root, session, now, symbols, universe, fetch_listing)
    except EvaluationError as error:
        return error
    retries.extend(f"listing:{symbol}" for symbol in pending)
    return None


def _acquire_liquidity(
    root: Path,
    session: date,
    now: datetime,
    source: CaptureSource,
    calls: list[tuple[str, str | None]],
    retries: list[str],
) -> None:
    """One current book for each normal-lane passer. Older books are not copied."""
    universe = load_universe()
    loaded_policy = load_policy(_policy_path())
    band = loaded_policy.policy.recorders.depth_band
    rules = loaded_policy.policy.hypotheses.universe.manipulation
    peg_limit = Decimal(str(rules.stablecoin_peg_deviation))
    for symbol in _captured_symbols(root, session):
        if _liquidity_path(root, session, symbol).is_file():
            continue
        loaded = load_symbol(root, session, symbol)
        if loaded is None:
            continue
        item, lane = loaded
        if lane is None or lane.lane is not Lane.NORMAL or settle(item, lane, universe) is not None:
            continue
        if not book_is_stored(root, session, symbol):
            try:
                payload = source.fetch(session, "book", symbol)
            except TemporaryFailure:
                retries.append(f"book:{symbol}")
                continue
            calls.append(("book", symbol))
            if not isinstance(payload, BookShot):
                raise EvaluationError("capture is unusable")
            store_session_book(root, session, symbol, payload)
        _store_passer_liquidity(root, session, now, symbol, item, band, peg_limit)


def _store_passer_liquidity(
    root: Path,
    session: date,
    now: datetime,
    symbol: str,
    item: SymbolClassification,
    band: Decimal,
    peg_limit: Decimal,
) -> None:
    document = json.loads(_book_path(root, session, symbol).read_text())
    if not isinstance(document, dict):
        raise EvaluationError("capture is unusable")
    observed = _clock_of(document.get("captured_at"), "captured_at")
    levels = _levels(document.get("bids")), _levels(document.get("asks"))
    reading = book_metrics(
        Depth(0, levels[0], levels[1]),
        band=band,
        observed_at=observed,
    )
    if reading is None:
        return
    quote, ticker_at = _ticker_quote(root, session, symbol)
    closes, peg_at = _peg_closes(root, session)
    aggregate = _aggregate_volume(root, session, item.coin_id, item.market_cap_source_timestamp)
    store_liquidity(
        root,
        session,
        build_liquidity(
            symbol,
            session=session,
            captured_at=now,
            spread_bps=reading.spread_bps,
            bid_usd=reading.bid_usd,
            ask_usd=reading.ask_usd,
            book_observed_at=observed,
            quote_volume=quote,
            ticker_captured_at=ticker_at,
            market_cap=item.market_cap_usd,
            aggregate_volume=aggregate,
            market_source_timestamp=item.market_cap_source_timestamp,
            hourly_closes=closes,
            peg_limit=peg_limit,
            peg_captured_at=peg_at,
        ),
    )


def _levels(value: object) -> tuple[DepthLevel, ...]:
    if not isinstance(value, list):
        raise EvaluationError("capture is unusable")
    levels: list[DepthLevel] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise EvaluationError("capture is unusable")
        levels.append(DepthLevel(_money(row[0]), _money(row[1])))
    return tuple(levels)


def _money(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise EvaluationError("capture is unusable")
    try:
        parsed = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
    except InvalidOperation as error:
        raise EvaluationError("capture is unusable") from error
    if not parsed.is_finite() or parsed < 0:
        raise EvaluationError("capture is unusable")
    return parsed


def _ticker_quote(root: Path, session: date, symbol: str) -> tuple[Decimal | None, datetime | None]:
    path = _ticker_path(root, session)
    if not path.is_file():
        return None, None
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        return None, None
    captured = _clock_of(document.get("captured_at"), "captured_at")
    rows = document.get("rows")
    if not isinstance(rows, list):
        return None, None
    for row in rows:
        if not isinstance(row, dict) or row.get("symbol") != symbol:
            continue
        return _money(row.get("quote_volume")), captured
    return None, None


def _peg_closes(root: Path, session: date) -> tuple[tuple[Decimal, ...], datetime | None]:
    path = _peg_path(root, session)
    if not path.is_file():
        return (), None
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        return (), None
    raw = document.get("closes")
    if not isinstance(raw, list):
        return (), None
    captured = _clock_of(document.get("captured_at"), "captured_at")
    return tuple(_money(value) for value in raw), captured


def _aggregate_volume(
    root: Path, session: date, coin_id: str | None, expected: datetime | None
) -> Decimal | None:
    if coin_id is None or expected is None:
        return None
    path = (
        root
        / "captures"
        / f"session={session.isoformat()}"
        / "classification_sources"
        / "markets.json"
    )
    if not path.is_file():
        return None
    document = json.loads(path.read_text())
    payload = document.get("payload") if isinstance(document, dict) else None
    if not isinstance(payload, list):
        return None
    for row in payload:
        if not isinstance(row, dict) or row.get("id") != coin_id:
            continue
        stamp = _clock_of(row.get("last_updated"), "last_updated")
        if stamp != expected:
            return None
        return _money(row.get("total_volume"))
    return None


def _clock_of(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise EvaluationError("capture is unusable")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _utc(parsed, label)
    return parsed.astimezone(UTC)


def _lane_universe() -> UniverseHypotheses | None:
    raw = os.environ.get("POLICY_PATH")
    if not raw:
        return None
    return load_policy(Path(raw)).policy.hypotheses.universe


def _post_close(
    root: Path,
    cycle: CaptureCycle,
    source: CaptureSource,
    symbols: tuple[str, ...],
    calls: list[tuple[str, str | None]],
    skipped: list[str],
    retries: list[str],
    spikes: list[tuple[str, int | None]],
) -> None:
    session = cycle.post_close
    if session is None:
        return
    ordered: list[str] = []
    for symbol in (*symbols, "BTCUSDT"):
        if symbol not in ordered:
            ordered.append(symbol)
    for symbol in ordered:
        _one_bar(root, cycle, source, symbol, calls, skipped, retries, spikes)


def _one_bar(
    root: Path,
    cycle: CaptureCycle,
    source: CaptureSource,
    symbol: str,
    calls: list[tuple[str, str | None]],
    skipped: list[str],
    retries: list[str],
    spikes: list[tuple[str, int | None]],
) -> None:
    session = cycle.post_close
    if session is None:
        return
    final = _final_path(root, session, symbol)
    hours_path = _hours_path(root, session, symbol)
    if final.is_file():
        skipped.append(f"daily_bars:{symbol}")
        bars = _bars_of(final)
    else:
        try:
            payload = source.fetch(session, "daily_bars", symbol)
        except TemporaryFailure:
            retries.append(f"daily_bars:{symbol}")
            return
        calls.append(("daily_bars", symbol))
        if not isinstance(payload, tuple) or any(not isinstance(bar, DailyBar) for bar in payload):
            raise EvaluationError("capture is unusable")
        bars = tuple(bar for bar in payload if isinstance(bar, DailyBar))
        capture = session_bar_capture(session, symbol, bars, cycle.now)
        store_completed_bars(root, session, capture)
    if hours_path.is_file():
        skipped.append(f"hour_bars:{symbol}")
        return
    try:
        payload = source.fetch(session, "hour_bars", symbol)
    except TemporaryFailure:
        retries.append(f"hour_bars:{symbol}")
        return
    calls.append(("hour_bars", symbol))
    if not isinstance(payload, tuple) or any(not isinstance(hour, HourQuote) for hour in payload):
        raise EvaluationError("capture is unusable")
    hours = tuple(hour for hour in payload if isinstance(hour, HourQuote))
    grid = _exact_hours(hours, session)
    if grid is None:
        retries.append(f"hour_bars:{symbol}")
        return
    count = _spike(session, cycle.now, bars, grid)
    spikes.append((symbol, count))
    _create(hours_path, _hours_body(session, symbol, grid, cycle.now, count))


def _policy_path() -> Path:
    """The workload sets POLICY_PATH. A checkout falls back to the repo file."""
    configured = os.environ.get("POLICY_PATH")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "policies" / "investment-policy.yaml"


def _spike(
    session: date, now: datetime, bars: tuple[DailyBar, ...], hours: tuple[HourQuote, ...]
) -> int | None:
    rules = load_policy(_policy_path()).policy.hypotheses.universe.manipulation
    try:
        return spike_candle_count(
            bars,
            hours,
            session=session,
            as_of=now,
            formula=rules.spike_count,
            zscore_above=Decimal(str(rules.volume_zscore_above)),
        )
    except EvaluationError as error:
        if str(error) != "spike evidence is inconsistent":
            raise
        return None


def _exact_hours(hours: Sequence[HourQuote], session: date) -> tuple[HourQuote, ...] | None:
    """The session day's 24 exact hours, or nothing while that grid is incomplete."""
    hour_ms = 3_600_000
    opened = datetime(session.year, session.month, session.day, tzinfo=UTC)
    midnight = int(opened.timestamp()) * 1000
    expected = [midnight + offset * hour_ms for offset in range(24)]
    wanted = set(expected)
    found: dict[int, HourQuote] = {}
    seen: set[int] = set()
    for hour in hours:
        if hour.open_ms % hour_ms != 0 or hour.close_ms != hour.open_ms + hour_ms - 1:
            raise EvaluationError("capture is unusable")
        if not hour.quote_volume.is_finite() or hour.quote_volume < 0:
            raise EvaluationError("capture is unusable")
        if hour.open_ms in seen:
            raise EvaluationError("capture is unusable")
        seen.add(hour.open_ms)
        if hour.open_ms in wanted:
            found[hour.open_ms] = hour
    if len(found) != 24:
        return None
    return tuple(found[open_ms] for open_ms in expected)


def _finalize(
    root: Path, cycle: CaptureCycle, git_sha: str
) -> tuple[SessionReadiness | None, bool]:
    session = cycle.post_close
    if session is None:
        return None, False
    if _manifest(root, session).is_file():
        return None, True
    if not _stage_ready(root, session):
        return None, False
    result = populate_session(
        root,
        session,
        cycle.now,
        load_closed(root, session),
        git_sha=git_sha,
        weights_present=False,
    )
    return result.readiness, result.manifest is not None


def _stage_ready(root: Path, session: date) -> bool:
    universe = _universe_path(root, session)
    if not universe.is_file() or not _ticker_path(root, session).is_file():
        return False
    if not _peg_path(root, session).is_file():
        return False
    if any(not _regime_path(root, session, series).is_file() for series in _REGIME):
        return False
    document = json.loads(universe.read_text())
    raw = document.get("symbols") if isinstance(document, dict) else None
    if not isinstance(raw, list) or not raw or any(not isinstance(symbol, str) for symbol in raw):
        return False
    policy = load_universe()
    return all(_symbol_ready(root, session, symbol, policy) for symbol in raw)


def _symbol_ready(root: Path, session: date, symbol: str, universe: UniverseHypotheses) -> bool:
    """Downstream files are required only while the deterministic path is still open."""
    loaded = load_symbol(root, session, symbol)
    if loaded is None:
        return False
    item, lane = loaded
    if settle(item, lane, universe) is not None:
        return True
    if needs_lane(item, universe) and lane is None:
        return False
    return (
        _final_path(root, session, symbol).is_file()
        and _hours_path(root, session, symbol).is_file()
        and _packet_path(root, session, symbol).is_file()
        and book_is_stored(root, session, symbol)
    )


def _one(
    root: Path,
    session: date,
    source: CaptureSource,
    kind: str,
    symbol: str | None,
    calls: list[tuple[str, str | None]],
    skipped: list[str],
    retries: list[str],
    keep: Callable[[Path, date, str | None, object], None],
) -> None:
    label = kind if symbol is None else f"{kind}:{symbol}"
    if _present(root, session, kind, symbol):
        skipped.append(label)
        return
    try:
        payload = source.fetch(session, kind, symbol)
    except TemporaryFailure:
        retries.append(label)
        return
    calls.append((kind, symbol))
    keep(root, session, symbol, payload)


def _present(root: Path, session: date, kind: str, symbol: str | None) -> bool:
    if kind == "universe":
        return _universe_path(root, session).is_file()
    if kind == "ticker":
        return _ticker_path(root, session).is_file()
    if kind == "peg":
        return _peg_path(root, session).is_file()
    if kind == "regime" and symbol is not None:
        return _regime_path(root, session, symbol).is_file()
    if kind == "classification" and symbol is not None:
        return _classification_path(root, session, symbol).is_file()
    return False


def _keep_universe(root: Path, session: date, symbol: str | None, payload: object) -> None:
    del symbol
    if not isinstance(payload, UniverseCapture):
        raise EvaluationError("capture is unusable")
    store_universe(root, session, payload)


def _keep_ticker(root: Path, session: date, symbol: str | None, payload: object) -> None:
    del symbol
    if (
        not isinstance(payload, tuple)
        or len(payload) != 3
        or not isinstance(payload[0], list)
        or not isinstance(payload[2], datetime)
        or not (payload[1] is None or isinstance(payload[1], datetime))
    ):
        raise EvaluationError("capture is unusable")
    source_timestamp, captured_at = payload[1], payload[2]
    _belongs(session, captured_at)
    close = session_close(session)
    if captured_at > close:
        raise EvaluationError("capture is after the close")
    _source_before_close(source_timestamp, close)
    _create(
        _ticker_path(root, session),
        _json(
            {
                "captured_at": _iso(captured_at),
                "rows": payload[0],
                "session": session.isoformat(),
                "source_timestamp": None if source_timestamp is None else _iso(source_timestamp),
            }
        ),
    )


def _keep_peg(root: Path, session: date, symbol: str | None, payload: object) -> None:
    del symbol
    if (
        not isinstance(payload, tuple)
        or len(payload) != 3
        or not isinstance(payload[0], tuple)
        or not isinstance(payload[2], datetime)
        or not (payload[1] is None or isinstance(payload[1], datetime))
    ):
        raise EvaluationError("capture is unusable")
    closes, source_timestamp, captured_at = payload
    if any(not isinstance(value, Decimal) for value in closes):
        raise EvaluationError("capture is unusable")
    _belongs(session, captured_at)
    close = session_close(session)
    if captured_at > close:
        raise EvaluationError("capture is after the close")
    _source_before_close(source_timestamp, close)
    _create(
        _peg_path(root, session),
        _json(
            {
                "captured_at": _iso(captured_at),
                "closes": [format(value, "f") for value in closes],
                "session": session.isoformat(),
                "source_timestamp": None if source_timestamp is None else _iso(source_timestamp),
                "symbol": "USDCUSDT",
            }
        ),
    )


def _keep_regime(root: Path, session: date, symbol: str | None, payload: object) -> None:
    del symbol
    if not isinstance(payload, RegimeCapture):
        raise EvaluationError("capture is unusable")
    store_regime(root, session, payload)


def _keep_classification(root: Path, session: date, symbol: str | None, payload: object) -> None:
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise EvaluationError("capture is unusable")
    item, captured_at = payload
    if not isinstance(item, SymbolClassification) or not isinstance(captured_at, datetime):
        raise EvaluationError("capture is unusable")
    if symbol is not None and item.symbol != symbol:
        raise EvaluationError("conflicting observation")
    _belongs(session, captured_at)
    store_classification(root, session, item, captured_at)


def _bars_of(path: Path) -> tuple[DailyBar, ...]:
    document = json.loads(path.read_text())
    rows = document["bars"] if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise EvaluationError("capture is unusable")
    return tuple(_bar(row) for row in rows)


def _bar(row: object) -> DailyBar:
    if not isinstance(row, dict):
        raise EvaluationError("capture is unusable")
    return DailyBar(
        symbol=str(row["symbol"]),
        open_date=date.fromisoformat(str(row["open_date"])),
        open=Decimal(str(row["open"])),
        high=Decimal(str(row["high"])),
        low=Decimal(str(row["low"])),
        close=Decimal(str(row["close"])),
        volume=Decimal(str(row["volume"])),
        quote_volume=Decimal(str(row["quote_volume"])),
        trade_count=int(str(row["trade_count"])),
        taker_buy_base_volume=Decimal(str(row["taker_buy_base_volume"])),
        taker_buy_quote_volume=Decimal(str(row["taker_buy_quote_volume"])),
    )


def _book_body(session: date, symbol: str, book: BookShot) -> bytes:
    source_timestamp = None if book.source_timestamp is None else _iso(book.source_timestamp)
    return _json(
        {
            "session": session.isoformat(),
            "symbol": symbol,
            "spread_bps": [format(book.spread_bps, "f")],
            "spread_snapshots": 1,
            "source_timestamp": source_timestamp,
            "captured_at": _iso(book.captured_at),
            "bids": [[format(price, "f"), format(quantity, "f")] for price, quantity in book.bids],
            "asks": [[format(price, "f"), format(quantity, "f")] for price, quantity in book.asks],
        }
    )


def _hours_body(
    session: date,
    symbol: str,
    hours: Sequence[HourQuote],
    captured_at: datetime,
    count: int | None,
) -> bytes:
    return _json(
        {
            "session": session.isoformat(),
            "symbol": symbol,
            "captured_at": _iso(captured_at),
            "spike_candle_count": count,
            "hours": [
                {
                    "open_ms": hour.open_ms,
                    "close_ms": hour.close_ms,
                    "quote_volume": format(hour.quote_volume, "f"),
                }
                for hour in hours
            ],
        }
    )


def _source_before_close(moment: datetime | None, close: datetime) -> None:
    if moment is None:
        return
    _utc(moment, "source_timestamp")
    if moment > close:
        raise EvaluationError("capture is after the close")


def _utc(moment: datetime, label: str) -> None:
    if moment.tzinfo is None or moment.utcoffset() != timedelta(0):
        raise EvaluationError(f"{label} must be timezone-aware UTC")


def _create(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == body:
            return
        raise EvaluationError("conflicting observation")
    path.write_bytes(body)


def _json(document: dict[str, object]) -> bytes:
    return json.dumps(document, sort_keys=True).encode()


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _manifest(root: Path, session: date) -> Path:
    return root / "sessions" / f"date={session.isoformat()}" / "manifest.json"


def _captured_symbols(root: Path, session: date) -> tuple[str, ...]:
    """Symbols in the stored universe. Books are not this list."""
    path = _universe_path(root, session)
    if not path.is_file():
        return ()
    document = json.loads(path.read_text())
    raw = document.get("symbols") if isinstance(document, dict) else None
    if not isinstance(raw, list) or any(not isinstance(symbol, str) for symbol in raw):
        raise EvaluationError("capture is unusable")
    return tuple(raw)


def _universe_path(root: Path, session: date) -> Path:
    return root / "captures" / f"session={session.isoformat()}" / "universe.json"


def _ticker_path(root: Path, session: date) -> Path:
    return root / "captures" / f"session={session.isoformat()}" / "ticker.json"


def _peg_path(root: Path, session: date) -> Path:
    return root / "captures" / f"session={session.isoformat()}" / "peg" / "symbol=USDCUSDT.json"


def _book_path(root: Path, session: date, symbol: str) -> Path:
    return root / "captures" / f"session={session.isoformat()}" / "book" / f"symbol={symbol}.json"


def _liquidity_path(root: Path, session: date, symbol: str) -> Path:
    folder = root / "captures" / f"session={session.isoformat()}" / "liquidity"
    return folder / f"symbol={symbol}.json"


def _classification_path(root: Path, session: date, symbol: str) -> Path:
    folder = root / "captures" / f"session={session.isoformat()}" / "classification"
    return folder / f"symbol={symbol}.json"


def _regime_path(root: Path, session: date, series: str) -> Path:
    return root / "captures" / f"session={session.isoformat()}" / "regime" / f"{series}.json"


def _packet_path(root: Path, session: date, symbol: str) -> Path:
    folder = root / "captures" / f"session={session.isoformat()}" / "candidates"
    return folder / f"symbol={symbol}.json"


def _final_path(root: Path, session: date, symbol: str) -> Path:
    return root / "captures" / f"session={session.isoformat()}" / "final" / f"symbol={symbol}.json"


def _hours_path(root: Path, session: date, symbol: str) -> Path:
    return root / "captures" / f"session={session.isoformat()}" / "hours" / f"symbol={symbol}.json"
