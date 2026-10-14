"""Point-in-time lane evidence for symbols that already cleared earlier gates.

The eligibility rules are unchanged. A symbol rejected by an exclusion or by the
stored market-cap floor is not looked up. Unlock evidence has no approved
point-in-time source, so a high-risk schedule stays unavailable rather than false.
A candidate packet is evidence for a symbol that passes the lane. It is not a BUY.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, cast

from cip.domain.errors import EvaluationError
from cip.domain.policy import UniverseHypotheses
from cip.evaluation.classification import SymbolClassification, to_facts
from cip.evaluation.eligibility import Lane, _exclusions, _lane, assess
from cip.evaluation.fundamentals import CmcReading, CoinGeckoReading, UnlockReading
from cip.evaluation.lane_fields import (
    LaneFields,
    MarketFields,
    classify_unlock,
    history_days,
    overlay_lane,
    parse_first_daily_open,
    parse_market_fields,
    store_lane,
    unlock_schedule_known,
)
from cip.evaluation.liquidity import MarketSnapshot
from cip.evaluation.populate import CaptureClock, PacketCapture
from cip.evaluation.prospect import prospective_session, store_packet
from cip.evaluation.scan import ScanCandidate
from cip.evaluation.session import session_close

_MAPPING = frozenset({"unambiguous", "ambiguous", "unmapped"})
_UNLOCKS = frozenset({"future_unlocks", "no_applicable_future_unlock", "unavailable"})


class ListingUnavailable(Exception):
    """The listing open was not retrieved. This symbol stays unstored."""


def capture_lanes(
    root: Path,
    session: date,
    captured_at: datetime,
    symbols: tuple[str, ...],
    universe: UniverseHypotheses,
    fetch_listing: Callable[[str], object],
) -> tuple[str, ...]:
    """Store lane evidence for the open session. A closed session is not filled."""
    prospective_session(session)
    if session == date(2026, 10, 6):
        raise EvaluationError("sealed session stays sealed")
    close = session_close(session)
    _before_close(captured_at, close)
    if _manifest(root, session).exists():
        raise EvaluationError("finalized session is sealed")
    classified = _classifications(root, session, symbols)
    if classified is None:
        return ()
    markets, market_captured_at = _markets(root, session, close)
    retries: list[str] = []
    problem: EvaluationError | None = None
    for item in classified:
        try:
            if _one(
                root,
                session,
                captured_at,
                close,
                item,
                universe,
                markets,
                market_captured_at,
                fetch_listing,
            ):
                retries.append(item.symbol)
        except EvaluationError as error:
            if problem is None:
                problem = error
    if problem is not None:
        raise problem
    return tuple(retries)


def _one(
    root: Path,
    session: date,
    captured_at: datetime,
    close: datetime,
    item: SymbolClassification,
    universe: UniverseHypotheses,
    markets: dict[str, MarketFields],
    market_captured_at: datetime | None,
    fetch_listing: Callable[[str], object],
) -> bool:
    lane = _selected(item, universe)
    if lane is None:
        return False
    path = _lane_path(root, session, item.symbol)
    if path.is_file():
        _packet(root, session, item, universe, _load_lane(path))
        return False
    if item.coin_id is None:
        raise EvaluationError("capture is unusable")
    reading = markets.get(item.coin_id)
    if reading is not None and reading.source_timestamp > close:
        raise EvaluationError("capture is after the close")
    try:
        payload = fetch_listing(item.symbol)
    except ListingUnavailable:
        return True
    opened = parse_first_daily_open(payload)
    if opened is None:
        days: int | None = None
        history_open = None
    else:
        days = history_days(opened, session)
        history_open = datetime.fromtimestamp(opened / 1000, UTC)
    rank = None if lane is Lane.HIGH_RISK or reading is None else reading.market_cap_rank
    fdv = None if lane is Lane.HIGH_RISK or reading is None else reading.fdv_to_market_cap
    stored = LaneFields(
        symbol=item.symbol,
        base_asset=item.base_asset,
        lane=lane,
        coin_id=item.coin_id,
        circulating_ratio=None if reading is None else reading.circulating_ratio,
        history_days=days,
        history_open=history_open,
        market_cap_rank=rank,
        fdv_to_market_cap=fdv,
        unlock_state=classify_unlock(None, tracked=False, as_of=captured_at),
        market_source_timestamp=None if reading is None else reading.source_timestamp,
        market_captured_at=captured_at if market_captured_at is None else market_captured_at,
        history_captured_at=captured_at,
        unlock_provider=None,
        unlock_provider_id=None,
        unlock_source_timestamp=None,
    )
    store_lane(root, session, stored)
    _packet(root, session, item, universe, stored)
    return False


def _selected(item: SymbolClassification, universe: UniverseHypotheses) -> Lane | None:
    """The lane that still needs fields. An earlier rejection is not looked up."""
    facts = to_facts(item, quote_asset="USDT", status="TRADING")
    if _exclusions(facts, universe):
        return None
    lane, _reasons = _lane(facts, universe)
    return lane


def _packet(
    root: Path,
    session: date,
    item: SymbolClassification,
    universe: UniverseHypotheses,
    stored: LaneFields,
) -> None:
    path = _packet_path(root, session, item.symbol)
    if path.is_file():
        return
    facts = overlay_lane(
        to_facts(item, quote_asset="USDT", status="TRADING"),
        lane=stored.lane,
        circulating_ratio=stored.circulating_ratio,
        history_days=stored.history_days,
        market_cap_rank=stored.market_cap_rank,
        fdv_to_market_cap=stored.fdv_to_market_cap,
        unlock=stored.unlock_state,
    )
    if not assess(facts, universe).eligible:
        return
    known = unlock_schedule_known(stored.unlock_state)
    candidate = ScanCandidate(
        facts=facts,
        market=MarketSnapshot(
            as_of=stored.history_captured_at,
            median_quote_volume_30d_usd=None,
            day_quote_volume_usd=None,
            median_spread_bps=None,
            spread_snapshots=None,
            depth_usd_per_side=None,
            turnover=None,
            volume_zscore=None,
            price_move=None,
            spike_candle_count=None,
            trade_size_stdev=None,
            taker_buy_ratio=None,
            binance_volume_share=None,
            stablecoin_peg_deviation=None,
            peg_deviation_hours=None,
        ),
        coingecko=CoinGeckoReading(
            coingecko_id=item.coin_id,
            market_cap_usd=item.market_cap_usd,
            circulating_supply=None,
            total_supply=None,
            fully_diluted_valuation_usd=None,
            source_timestamp=item.market_cap_source_timestamp,
        ),
        cmc=CmcReading(
            market_cap_usd=item.cmc_market_cap_usd,
            source_timestamp=item.cmc_source_timestamp,
        ),
        unlocks=UnlockReading(schedule_known=known, pct_circ_14d=None, pct_circ_90d=None),
        tokenomics=None,
    )
    store_packet(
        root,
        session,
        PacketCapture(
            item.symbol,
            candidate,
            CaptureClock(
                "coingecko:markets",
                stored.market_source_timestamp,
                stored.history_captured_at,
            ),
        ),
    )


def _classifications(
    root: Path, session: date, symbols: tuple[str, ...]
) -> tuple[SymbolClassification, ...] | None:
    found: list[SymbolClassification] = []
    for symbol in symbols:
        path = _classification_path(root, session, symbol)
        if not path.is_file():
            return None
        found.append(_load_classification(path))
    return tuple(found)


def _markets(
    root: Path, session: date, close: datetime
) -> tuple[dict[str, MarketFields], datetime | None]:
    path = _markets_path(root, session)
    if not path.is_file():
        return {}, None
    document = _object(path)
    captured = _moment(document.get("captured_at"))
    if captured is None:
        raise EvaluationError("capture is unusable")
    if captured > close:
        raise EvaluationError("capture is after the close")
    return parse_market_fields(document.get("payload")), captured


def _load_classification(path: Path) -> SymbolClassification:
    document = _object(path)
    mapping = document.get("mapping")
    if mapping not in _MAPPING:
        raise EvaluationError("capture is unusable")
    named = cast(Literal["unambiguous", "ambiguous", "unmapped"], mapping)
    return SymbolClassification(
        symbol=_text(document.get("symbol")),
        base_asset=_text(document.get("base_asset")),
        eur_stable=_flag(document.get("eur_stable")),
        fan_token=_flag(document.get("fan_token")),
        monitoring_tag=_flag(document.get("monitoring_tag")),
        delisting=_flag(document.get("delisting")),
        deposits_suspended=_flag(document.get("deposits_suspended")),
        withdrawals_suspended=_flag(document.get("withdrawals_suspended")),
        pending_migration=_flag(document.get("pending_migration")),
        coin_id=_optional_text(document.get("coin_id")),
        mapping=named,
        market_cap_usd=_decimal(document.get("market_cap_usd")),
        market_cap_source_timestamp=_moment(document.get("market_cap_source_timestamp")),
        cmc_id=_optional_text(document.get("cmc_id")),
        cmc_market_cap_usd=_decimal(document.get("cmc_market_cap_usd")),
        cmc_source_timestamp=_moment(document.get("cmc_source_timestamp")),
    )


def _load_lane(path: Path) -> LaneFields:
    document = _object(path)
    lane = document.get("lane")
    unlock = document.get("unlock_state")
    if not isinstance(lane, str) or lane not in {item.value for item in Lane}:
        raise EvaluationError("capture is unusable")
    if not isinstance(unlock, str) or unlock not in _UNLOCKS:
        raise EvaluationError("capture is unusable")
    return LaneFields(
        symbol=_text(document.get("symbol")),
        base_asset=_text(document.get("base_asset")),
        lane=Lane(lane),
        coin_id=_text(document.get("coin_id")),
        circulating_ratio=_decimal(document.get("circulating_ratio")),
        history_days=_whole(document.get("history_days")),
        history_open=_moment(document.get("history_open")),
        market_cap_rank=_whole(document.get("market_cap_rank")),
        fdv_to_market_cap=_decimal(document.get("fdv_to_market_cap")),
        unlock_state=cast(
            Literal["future_unlocks", "no_applicable_future_unlock", "unavailable"], unlock
        ),
        market_source_timestamp=_moment(document.get("market_source_timestamp")),
        market_captured_at=_required_moment(document.get("market_captured_at")),
        history_captured_at=_required_moment(document.get("history_captured_at")),
        unlock_provider=_optional_text(document.get("unlock_provider")),
        unlock_provider_id=_optional_text(document.get("unlock_provider_id")),
        unlock_source_timestamp=_moment(document.get("unlock_source_timestamp")),
    )


def _object(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise EvaluationError("capture is unusable")
    return document


def _text(value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise EvaluationError("capture is unusable")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _text(value)


def _flag(value: object) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise EvaluationError("capture is unusable")


def _decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvaluationError("capture is unusable")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise EvaluationError("capture is unusable") from error
    if not parsed.is_finite():
        raise EvaluationError("capture is unusable")
    return parsed


def _whole(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise EvaluationError("capture is unusable")
    return value


def _moment(value: object) -> datetime | None:
    if value is None:
        return None
    return _required_moment(value)


def _required_moment(value: object) -> datetime:
    if not isinstance(value, str):
        raise EvaluationError("capture is unusable")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvaluationError("capture is unusable") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise EvaluationError("capture is unusable")
    return parsed.astimezone(UTC)


def _before_close(moment: datetime, close: datetime) -> None:
    if moment.tzinfo is None or moment.utcoffset() != timedelta(0):
        raise EvaluationError("captured_at must be timezone-aware UTC")
    if moment > close:
        raise EvaluationError("capture is after the close")


def _classification_path(root: Path, session: date, symbol: str) -> Path:
    return (
        root
        / "captures"
        / f"session={session.isoformat()}"
        / "classification"
        / f"symbol={symbol}.json"
    )


def _markets_path(root: Path, session: date) -> Path:
    return (
        root
        / "captures"
        / f"session={session.isoformat()}"
        / "classification_sources"
        / "markets.json"
    )


def _lane_path(root: Path, session: date, symbol: str) -> Path:
    return root / "captures" / f"session={session.isoformat()}" / "lane" / f"symbol={symbol}.json"


def _packet_path(root: Path, session: date, symbol: str) -> Path:
    folder = root / "captures" / f"session={session.isoformat()}" / "candidates"
    return folder / f"symbol={symbol}.json"


def _manifest(root: Path, session: date) -> Path:
    return root / "sessions" / f"date={session.isoformat()}" / "manifest.json"
