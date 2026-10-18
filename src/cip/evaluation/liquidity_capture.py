"""Point-in-time liquidity observations for one prospective session.

This module does not call a provider and does not change liquidity thresholds.
A short sample stays missing. The session bar is not a taker ratio.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from cip.adapters.market import Depth, parse_klines
from cip.domain.errors import EvaluationError, MarketDataError, RecorderError
from cip.domain.policy import ManipulationHypotheses, SpikeCandleFormula
from cip.evaluation.features import _median
from cip.evaluation.prospect import prospective_session
from cip.evaluation.session import session_close
from cip.history.bars import DailyBar
from cip.recorders.observation import Observation
from cip.recorders.sources import book_observations

_SYMBOL = re.compile(r"^[A-Z0-9]{1,20}$")
_SEALED = date(2026, 10, 6)
_VOLUME_DAYS = 30
_BASELINE_DAYS = 30
_DAY_MS = 86_400_000
_HOUR_MS = 3_600_000


@dataclass(frozen=True)
class HourQuote:
    """One completed UTC hour. Quote volume is the Binance kline field at index 7."""

    open_ms: int
    close_ms: int
    quote_volume: Decimal


@dataclass(frozen=True)
class BookReading:
    """One order-book observation. The spread count is the number of books."""

    spread_bps: Decimal
    snapshots: int
    bid_usd: Decimal
    ask_usd: Decimal
    depth_usd_per_side: Decimal


@dataclass(frozen=True)
class LiquidityDerived:
    """Gate inputs reproduced from raw bars and quotes. Missing stays missing."""

    median_quote_volume_30d_usd: Decimal | None
    day_quote_volume_usd: Decimal | None
    volume_zscore: Decimal | None
    price_move: Decimal | None
    trade_size_stdev: Decimal | None
    taker_buy_ratio: None
    spike_candle_count: None
    turnover: Decimal | None
    binance_volume_share: Decimal | None


@dataclass(frozen=True)
class LiquidityCapture:
    """Raw observations and the inputs derived from them."""

    symbol: str
    bars: tuple[DailyBar, ...]
    median_quote_volume_30d_usd: Decimal | None
    day_quote_volume_usd: Decimal | None
    spread_bps: tuple[Decimal, ...]
    median_spread_bps: Decimal | None
    spread_snapshots: int | None
    bid_usd: Decimal | None
    ask_usd: Decimal | None
    depth_usd_per_side: Decimal | None
    binance_quote_volume_24h: Decimal | None
    market_cap_usd: Decimal | None
    aggregate_volume_usd: Decimal | None
    turnover: Decimal | None
    volume_zscore: Decimal | None
    price_move: Decimal | None
    taker_buy_ratio: Decimal | None
    spike_candle_count: int | None
    trade_size_stdev: Decimal | None
    binance_volume_share: Decimal | None
    stablecoin_peg_deviation: Decimal | None
    peg_deviation_hours: int | None
    hourly_closes: tuple[Decimal, ...]
    peg_limit: Decimal | None
    bar_captured_at: datetime
    book_observed_at: datetime | None
    ticker_captured_at: datetime | None
    market_source_timestamp: datetime | None
    peg_captured_at: datetime | None
    hour_quotes: tuple[HourQuote, ...] = ()


def parse_daily_klines(payload: object, *, symbol: str) -> tuple[DailyBar, ...]:
    """Daily bars from a kline payload. Taker volume is kept."""
    if _SYMBOL.fullmatch(symbol) is None:
        raise EvaluationError("capture is unusable")
    if not isinstance(payload, list):
        raise EvaluationError("capture is unusable")
    try:
        rows = parse_klines(payload)
    except MarketDataError as error:
        raise EvaluationError("capture is unusable") from error
    bars: list[DailyBar] = []
    for raw, row in zip(payload, rows, strict=True):
        if not isinstance(raw, list) or len(raw) < 11:
            raise EvaluationError("capture is unusable")
        taker_base = _decimal(raw[9])
        taker_quote = _decimal(raw[10])
        opened = _utc_day(row.open_time, row.close_time)
        bars.append(
            DailyBar(
                symbol=symbol,
                open_date=opened,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
                quote_volume=row.quote_volume,
                trade_count=row.trade_count,
                taker_buy_base_volume=taker_base,
                taker_buy_quote_volume=taker_quote,
            )
        )
    return tuple(bars)


def quote_window(bars: Sequence[DailyBar], session: date) -> tuple[DailyBar, ...] | None:
    """Consecutive completed days ending the day before the session."""
    prior = [bar for bar in bars if bar.open_date < session]
    if not prior:
        return None
    ordered = tuple(sorted(prior, key=lambda bar: bar.open_date))
    if ordered[-1].open_date != session - timedelta(days=1):
        return None
    dates = [bar.open_date for bar in ordered]
    if len(dates) != len(set(dates)):
        raise EvaluationError("capture is unusable")
    kept = [ordered[-1]]
    for bar in reversed(ordered[:-1]):
        if bar.open_date != kept[-1].open_date - timedelta(days=1):
            break
        kept.append(bar)
    kept.reverse()
    return tuple(kept)


def derive_liquidity(
    bars: Sequence[DailyBar],
    session: date,
    *,
    binance_quote_volume: Decimal | None = None,
    market_cap: Decimal | None = None,
    aggregate_volume: Decimal | None = None,
) -> LiquidityDerived:
    """Reproduce the gate inputs. The session bar is not a taker ratio."""
    window = quote_window(bars, session)
    median, day = _volume_pair(window)
    return LiquidityDerived(
        median_quote_volume_30d_usd=median,
        day_quote_volume_usd=day,
        volume_zscore=_zscore(window, _quote),
        price_move=_move(window),
        trade_size_stdev=_trade_distance(window),
        taker_buy_ratio=None,
        spike_candle_count=None,
        turnover=_ratio(binance_quote_volume, market_cap),
        binance_volume_share=volume_share(binance_quote_volume, aggregate_volume),
    )


def spike_candle_count(
    days: Sequence[DailyBar],
    hours: Sequence[HourQuote],
    *,
    session: date,
    as_of: datetime,
    formula: SpikeCandleFormula,
    zscore_above: Decimal,
) -> int | None:
    """Count material hours of an abnormal day. A contradiction is not a missing count."""
    if formula.minimum_daily_bars != formula.baseline_days + 1:
        raise EvaluationError("capture is unusable")
    event = _event_day(session, as_of)
    if event is None:
        return None
    window = _ending(days, event, formula.baseline_days + 1)
    if window is None:
        return None
    grid = _hour_grid(hours, event, formula.exact_hours)
    if grid is None:
        return None
    baseline = [bar.quote_volume for bar in window[:-1]]
    current = window[-1].quote_volume
    _usable_quotes([*baseline, current])
    if sum((hour.quote_volume for hour in grid), Decimal(0)) != current:
        raise EvaluationError("spike evidence is inconsistent")
    deviation = _sample_stdev(baseline)
    if deviation is None:
        return None
    mean = sum(baseline, Decimal(0)) / Decimal(len(baseline))
    if current - mean <= zscore_above * deviation:
        return 0
    pace = mean / Decimal(formula.exact_hours)
    return sum(1 for hour in grid if hour.quote_volume > pace)


def book_metrics(book: Depth, *, band: Decimal, observed_at: datetime) -> BookReading | None:
    """Spread and the thinner side of the ±2% book. An empty book is missing."""
    try:
        spread, depth = book_observations("BOOK", book, observed_at=observed_at, depth_band=band)
    except RecorderError:
        return None
    bid = _named(depth, "bid_usd")
    ask = _named(depth, "ask_usd")
    return BookReading(_named(spread, "spread_bps"), 1, bid, ask, min(bid, ask))


def volume_share(binance_quote: Decimal | None, aggregate: Decimal | None) -> Decimal | None:
    """Binance volume over CoinGecko aggregate. A larger Binance print is not a share."""
    if binance_quote is None or aggregate is None or aggregate <= 0 or binance_quote > aggregate:
        return None
    if binance_quote < 0:
        raise EvaluationError("capture is unusable")
    return binance_quote / aggregate


def peg_reading(closes: Sequence[Decimal], *, limit: Decimal) -> tuple[Decimal, int] | None:
    """Latest closed-hour distance from 1, and how many trailing hours exceed the band."""
    if not closes:
        return None
    hours = 0
    for close in reversed(closes):
        if abs(close - 1) > limit:
            hours += 1
            continue
        break
    return abs(closes[-1] - 1), hours


def closed_hour_closes(payload: object, *, captured_at: datetime) -> tuple[Decimal, ...]:
    """Hourly closes whose candle has finished. The open hour is dropped."""
    if not isinstance(payload, list):
        raise EvaluationError("capture is unusable")
    closes: list[Decimal] = []
    opened_at: list[int] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 7:
            raise EvaluationError("capture is unusable")
        open_time = row[0]
        close_time = row[6]
        if type(open_time) is not int or type(close_time) is not int:
            raise EvaluationError("capture is unusable")
        if open_time % _HOUR_MS != 0 or close_time != open_time + _HOUR_MS - 1:
            raise EvaluationError("capture is unusable")
        finished = _instant((close_time + 1) / 1000)
        if finished > captured_at:
            continue
        closes.append(_decimal(row[4]))
        opened_at.append(open_time)
    for earlier, later in itertools.pairwise(opened_at):
        if later - earlier != _HOUR_MS:
            raise EvaluationError("capture is unusable")
    return tuple(closes)


def build_liquidity(
    symbol: str,
    *,
    session: date,
    captured_at: datetime,
    spread_bps: Decimal,
    bid_usd: Decimal,
    ask_usd: Decimal,
    book_observed_at: datetime,
    prior_spreads: tuple[Decimal, ...] = (),
    quote_volume: Decimal | None = None,
    ticker_captured_at: datetime | None = None,
    market_cap: Decimal | None = None,
    aggregate_volume: Decimal | None = None,
    market_source_timestamp: datetime | None = None,
    hourly_closes: tuple[Decimal, ...] = (),
    peg_limit: Decimal | None = None,
    peg_captured_at: datetime | None = None,
) -> LiquidityCapture:
    """One session book plus earlier books for that symbol. Closed-bar fields stay missing."""
    spreads = (*prior_spreads, spread_bps)
    if any(not value.is_finite() or value < 0 for value in spreads):
        raise EvaluationError("capture is unusable")
    derived = derive_liquidity(
        (),
        session,
        binance_quote_volume=quote_volume,
        market_cap=market_cap,
        aggregate_volume=aggregate_volume,
    )
    pegged = (
        peg_reading(hourly_closes, limit=peg_limit)
        if hourly_closes and peg_limit is not None
        else None
    )
    return LiquidityCapture(
        symbol=symbol,
        bars=(),
        median_quote_volume_30d_usd=derived.median_quote_volume_30d_usd,
        day_quote_volume_usd=derived.day_quote_volume_usd,
        spread_bps=spreads,
        median_spread_bps=_median(list(spreads)),
        spread_snapshots=len(spreads),
        bid_usd=bid_usd,
        ask_usd=ask_usd,
        depth_usd_per_side=min(bid_usd, ask_usd),
        binance_quote_volume_24h=quote_volume,
        market_cap_usd=market_cap,
        aggregate_volume_usd=aggregate_volume,
        turnover=derived.turnover,
        volume_zscore=derived.volume_zscore,
        price_move=derived.price_move,
        taker_buy_ratio=None,
        spike_candle_count=None,
        trade_size_stdev=derived.trade_size_stdev,
        binance_volume_share=derived.binance_volume_share,
        stablecoin_peg_deviation=None if pegged is None else pegged[0],
        peg_deviation_hours=None if pegged is None else pegged[1],
        hourly_closes=hourly_closes if pegged is not None else (),
        peg_limit=peg_limit if pegged is not None else None,
        bar_captured_at=captured_at,
        book_observed_at=book_observed_at,
        ticker_captured_at=ticker_captured_at,
        market_source_timestamp=market_source_timestamp,
        peg_captured_at=peg_captured_at if pegged is not None else None,
    )


def store_liquidity(
    root: Path,
    session: date,
    item: LiquidityCapture,
    *,
    manipulation: ManipulationHypotheses | None = None,
) -> None:
    """Store one pre-close liquidity document. Derived inputs must match the raw bars."""
    prospective_session(session)
    if session == _SEALED:
        raise EvaluationError("sealed session stays sealed")
    if (root / "sessions" / f"date={session.isoformat()}" / "manifest.json").exists():
        raise EvaluationError("finalized session is sealed")
    _checked(session, item, manipulation)
    _create(_path(root, session, item.symbol), _body(session, item))


def _volume_pair(window: tuple[DailyBar, ...] | None) -> tuple[Decimal | None, Decimal | None]:
    if window is None or len(window) < _VOLUME_DAYS:
        return None, None
    quotes = [bar.quote_volume for bar in window[-_VOLUME_DAYS:]]
    return _median(quotes), min(quotes)


def _zscore(
    window: tuple[DailyBar, ...] | None,
    amount: Callable[[DailyBar], Decimal | None],
) -> Decimal | None:
    if window is None or len(window) < _BASELINE_DAYS + 1:
        return None
    span = window[-(_BASELINE_DAYS + 1) : -1]
    baseline = [amount(bar) for bar in span]
    if any(value is None for value in baseline):
        return None
    numbers = [value for value in baseline if value is not None]
    deviation = _sample_stdev(numbers)
    current = amount(window[-1])
    if deviation is None or current is None:
        return None
    mean = sum(numbers, Decimal(0)) / Decimal(len(numbers))
    return (current - mean) / deviation


def _quote(bar: DailyBar) -> Decimal:
    return bar.quote_volume


def _trade_size(bar: DailyBar) -> Decimal | None:
    if bar.trade_count <= 0:
        return None
    return bar.quote_volume / Decimal(bar.trade_count)


def _trade_distance(window: tuple[DailyBar, ...] | None) -> Decimal | None:
    distance = _zscore(window, _trade_size)
    if distance is None:
        return None
    return abs(distance)


def _move(window: tuple[DailyBar, ...] | None) -> Decimal | None:
    if window is None or len(window) < 2 or window[-2].close <= 0:
        return None
    return window[-1].close / window[-2].close - 1


def _sample_stdev(values: Sequence[Decimal]) -> Decimal | None:
    mean = sum(values, Decimal(0)) / Decimal(len(values))
    variance = sum((item - mean) ** 2 for item in values) / Decimal(len(values) - 1)
    if variance == 0:
        return None
    return variance.sqrt()


def _ratio(numerator: Decimal | None, denominator: Decimal | None) -> Decimal | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    if numerator < 0:
        raise EvaluationError("capture is unusable")
    return numerator / denominator


def _named(observation: Observation, name: str) -> Decimal:
    return next(value for label, value in observation.values if label == name)


def _utc_day(open_time: int, close_time: int) -> date:
    if open_time % _DAY_MS != 0 or close_time != open_time + _DAY_MS - 1:
        raise EvaluationError("capture is unusable")
    return _instant(open_time / 1000).date()


def _instant(seconds: float) -> datetime:
    try:
        return datetime.fromtimestamp(seconds, UTC)
    except (ValueError, OverflowError, OSError) as error:
        raise EvaluationError("capture is unusable") from error


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, str):
        raise EvaluationError("capture is unusable")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise EvaluationError("capture is unusable") from error
    if not parsed.is_finite() or parsed < 0:
        raise EvaluationError("capture is unusable")
    return parsed


def _event_day(session: date, as_of: datetime) -> date | None:
    if as_of.tzinfo is None or as_of.utcoffset() != timedelta(0):
        raise EvaluationError("captured_at must be timezone-aware UTC")
    candidate = session if as_of >= session_close(session) else session - timedelta(days=1)
    if as_of < session_close(candidate):
        return None
    return candidate


def _ending(days: Sequence[DailyBar], event: date, count: int) -> tuple[DailyBar, ...] | None:
    selected = [bar for bar in days if bar.open_date <= event]
    if not selected:
        return None
    ordered = tuple(sorted(selected, key=lambda bar: bar.open_date))
    dates = [bar.open_date for bar in ordered]
    if len(dates) != len(set(dates)):
        raise EvaluationError("capture is unusable")
    if ordered[-1].open_date != event or len(ordered) < count:
        return None
    window = ordered[-count:]
    expected = event
    for bar in reversed(window):
        if bar.open_date != expected:
            return None
        expected -= timedelta(days=1)
    return window


def _hour_grid(
    hours: Sequence[HourQuote], event: date, exact_hours: int
) -> tuple[HourQuote, ...] | None:
    midnight = int(datetime(event.year, event.month, event.day, tzinfo=UTC).timestamp()) * 1000
    expected = [midnight + offset * _HOUR_MS for offset in range(exact_hours)]
    by_open: dict[int, HourQuote] = {}
    for hour in hours:
        if hour.open_ms % _HOUR_MS != 0 or hour.close_ms != hour.open_ms + _HOUR_MS - 1:
            raise EvaluationError("capture is unusable")
        if not hour.quote_volume.is_finite() or hour.quote_volume < 0:
            raise EvaluationError("capture is unusable")
        if hour.open_ms in by_open:
            raise EvaluationError("capture is unusable")
        by_open[hour.open_ms] = hour
    chosen: list[HourQuote] = []
    for open_ms in expected:
        found = by_open.get(open_ms)
        if found is None:
            return None
        chosen.append(found)
    return tuple(chosen)


def _usable_quotes(quotes: Sequence[Decimal]) -> None:
    if any(not value.is_finite() or value < 0 for value in quotes):
        raise EvaluationError("capture is unusable")


def _checked(
    session: date, item: LiquidityCapture, manipulation: ManipulationHypotheses | None
) -> None:
    if _SYMBOL.fullmatch(item.symbol) is None:
        raise EvaluationError("capture is unusable")
    if item.taker_buy_ratio is not None:
        raise EvaluationError("session bar stays missing")
    close = session_close(session)
    _completed_days(session, item, close)
    _require_clock(_book_present(item), item.book_observed_at, close)
    _require_clock(item.binance_quote_volume_24h is not None, item.ticker_captured_at, close)
    market_present = item.market_cap_usd is not None or item.aggregate_volume_usd is not None
    _require_clock(market_present, item.market_source_timestamp, close)
    peg_present = (
        item.stablecoin_peg_deviation is not None
        or item.peg_deviation_hours is not None
        or bool(item.hourly_closes)
        or item.peg_limit is not None
    )
    _require_clock(peg_present, item.peg_captured_at, close)
    derived = derive_liquidity(
        item.bars,
        session,
        binance_quote_volume=item.binance_quote_volume_24h,
        market_cap=item.market_cap_usd,
        aggregate_volume=item.aggregate_volume_usd,
    )
    if (
        item.median_quote_volume_30d_usd != derived.median_quote_volume_30d_usd
        or item.day_quote_volume_usd != derived.day_quote_volume_usd
        or item.volume_zscore != derived.volume_zscore
        or item.price_move != derived.price_move
        or item.trade_size_stdev != derived.trade_size_stdev
        or item.turnover != derived.turnover
        or item.binance_volume_share != derived.binance_volume_share
    ):
        raise EvaluationError("capture does not match the raw observations")
    _spread_agrees(item)
    _peg_pair(item)
    _spike_agrees(session, item, manipulation)


def _spike_agrees(
    session: date, item: LiquidityCapture, manipulation: ManipulationHypotheses | None
) -> None:
    if not item.hour_quotes:
        if item.spike_candle_count is not None:
            raise EvaluationError("capture does not match the raw observations")
        return
    if manipulation is None:
        raise EvaluationError("capture is unusable")
    expected = spike_candle_count(
        item.bars,
        item.hour_quotes,
        session=session,
        as_of=item.bar_captured_at,
        formula=manipulation.spike_count,
        zscore_above=Decimal(str(manipulation.volume_zscore_above)),
    )
    if item.spike_candle_count != expected:
        raise EvaluationError("capture does not match the raw observations")


def _spread_agrees(item: LiquidityCapture) -> None:
    if not item.spread_bps:
        if item.median_spread_bps is not None or item.spread_snapshots is not None:
            raise EvaluationError("capture does not match the raw observations")
        if item.depth_usd_per_side is not None:
            raise EvaluationError("capture does not match the raw observations")
        return
    if item.median_spread_bps != _median(list(item.spread_bps)):
        raise EvaluationError("capture does not match the raw observations")
    if item.spread_snapshots != len(item.spread_bps):
        raise EvaluationError("capture does not match the raw observations")
    if item.bid_usd is None or item.ask_usd is None:
        raise EvaluationError("capture does not match the raw observations")
    if item.depth_usd_per_side != min(item.bid_usd, item.ask_usd):
        raise EvaluationError("capture does not match the raw observations")


def _completed_days(session: date, item: LiquidityCapture, close: datetime) -> None:
    _before_close(item.bar_captured_at, close)
    for bar in item.bars:
        if bar.open_date > session:
            raise EvaluationError("capture is after the close")
        if bar.open_date == session:
            raise EvaluationError("session bar was retrieved before it closed")
    if not item.bars:
        return
    period_end = session_close(max(bar.open_date for bar in item.bars))
    if item.bar_captured_at < period_end:
        raise EvaluationError("bar was retrieved before it closed")


def _book_present(item: LiquidityCapture) -> bool:
    if item.spread_bps:
        return True
    return any(
        value is not None
        for value in (
            item.median_spread_bps,
            item.spread_snapshots,
            item.bid_usd,
            item.ask_usd,
            item.depth_usd_per_side,
        )
    )


def _require_clock(present: bool, moment: datetime | None, close: datetime) -> None:
    if present and moment is None:
        raise EvaluationError("undated observation")
    if moment is not None:
        _before_close(moment, close)


def _peg_pair(item: LiquidityCapture) -> None:
    deviation = item.stablecoin_peg_deviation
    hours = item.peg_deviation_hours
    if (deviation is None) != (hours is None):
        raise EvaluationError("capture is unusable")
    if deviation is None:
        if item.hourly_closes or item.peg_limit is not None:
            raise EvaluationError("capture is unusable")
        return
    if not item.hourly_closes or item.peg_limit is None:
        raise EvaluationError("capture is unusable")
    if not item.peg_limit.is_finite() or item.peg_limit < 0:
        raise EvaluationError("capture is unusable")
    if peg_reading(item.hourly_closes, limit=item.peg_limit) != (deviation, hours):
        raise EvaluationError("capture does not match the raw observations")


def _before_close(moment: datetime, close: datetime) -> None:
    if moment.tzinfo is None or moment.utcoffset() != timedelta(0):
        raise EvaluationError("captured_at must be timezone-aware UTC")
    if moment > close:
        raise EvaluationError("capture is after the close")


def _path(root: Path, session: date, symbol: str) -> Path:
    folder = root / "captures" / f"session={session.isoformat()}" / "liquidity"
    return folder / f"symbol={symbol}.json"


def _body(session: date, item: LiquidityCapture) -> bytes:
    document = {
        "session": session.isoformat(),
        "symbol": item.symbol,
        "bars": [_bar(bar) for bar in item.bars],
        "median_quote_volume_30d_usd": _text(item.median_quote_volume_30d_usd),
        "day_quote_volume_usd": _text(item.day_quote_volume_usd),
        "spread_bps": [_text(value) for value in item.spread_bps],
        "median_spread_bps": _text(item.median_spread_bps),
        "spread_snapshots": item.spread_snapshots,
        "bid_usd": _text(item.bid_usd),
        "ask_usd": _text(item.ask_usd),
        "depth_usd_per_side": _text(item.depth_usd_per_side),
        "binance_quote_volume_24h": _text(item.binance_quote_volume_24h),
        "market_cap_usd": _text(item.market_cap_usd),
        "aggregate_volume_usd": _text(item.aggregate_volume_usd),
        "turnover": _text(item.turnover),
        "volume_zscore": _text(item.volume_zscore),
        "price_move": _text(item.price_move),
        "taker_buy_ratio": None,
        "spike_candle_count": item.spike_candle_count,
        "hour_quotes": [_hour(hour) for hour in item.hour_quotes],
        "trade_size_stdev": _text(item.trade_size_stdev),
        "binance_volume_share": _text(item.binance_volume_share),
        "stablecoin_peg_deviation": _text(item.stablecoin_peg_deviation),
        "peg_deviation_hours": item.peg_deviation_hours,
        "hourly_closes": [format(value, "f") for value in item.hourly_closes],
        "peg_limit": _text(item.peg_limit),
        "bar_captured_at": _iso(item.bar_captured_at),
        "book_observed_at": _stamp(item.book_observed_at),
        "ticker_captured_at": _stamp(item.ticker_captured_at),
        "market_source_timestamp": _stamp(item.market_source_timestamp),
        "peg_captured_at": _stamp(item.peg_captured_at),
    }
    return json.dumps(document, sort_keys=True).encode()


def _hour(hour: HourQuote) -> dict[str, object]:
    return {
        "open_ms": hour.open_ms,
        "close_ms": hour.close_ms,
        "quote_volume": format(hour.quote_volume, "f"),
    }


def _bar(bar: DailyBar) -> dict[str, object]:
    return {
        "open_date": bar.open_date.isoformat(),
        "close": format(bar.close, "f"),
        "quote_volume": format(bar.quote_volume, "f"),
        "trade_count": bar.trade_count,
        "taker_buy_quote_volume": format(bar.taker_buy_quote_volume, "f"),
    }


def _text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _stamp(moment: datetime | None) -> str | None:
    return None if moment is None else _iso(moment)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _create(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == body:
            return
        raise EvaluationError(f"capture {path.name} already exists with a different payload")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(body)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != body:
            raise EvaluationError(
                f"capture {path.name} already exists with a different payload"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
