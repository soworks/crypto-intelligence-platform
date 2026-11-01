"""Build scan candidates from one session's stored evidence.

This module does not call a provider and does not rewrite a session. A null
field stays null. A symbol that already failed before liquidity does not need
a book, a bar, or an hour grid.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from cip.domain.errors import EvaluationError
from cip.domain.policy import UniverseHypotheses
from cip.evaluation.classification import SymbolClassification, to_facts
from cip.evaluation.fundamentals import CmcReading, CoinGeckoReading, UnlockReading
from cip.evaluation.lane_fields import LaneFields, overlay_lane
from cip.evaluation.liquidity import MarketSnapshot
from cip.evaluation.liquidity_capture import derive_liquidity
from cip.evaluation.populate import _load_bars
from cip.evaluation.scan import ScanCandidate, UniverseSnapshot
from cip.evaluation.session import session_close
from cip.evaluation.stage import load_symbol, settle
from cip.history.bars import DailyBar


def assemble_stored_candidates(
    root: Path,
    session: date,
    snapshot: UniverseSnapshot | None,
    packets: Mapping[str, ScanCandidate | None],
    universe: UniverseHypotheses,
) -> dict[str, ScanCandidate | None]:
    """One candidate per universe symbol, using only that session's files."""
    if snapshot is None:
        raise EvaluationError("universe snapshot is missing")
    bars = _load_bars(root, session)
    found: dict[str, ScanCandidate | None] = {}
    for symbol in snapshot.symbols:
        loaded = load_symbol(root, session, symbol, sealed=True)
        if loaded is None:
            raise EvaluationError("classification is missing")
        item, lane = loaded
        settled = settle(item, lane, universe) is not None
        packet = packets.get(symbol)
        if packet is None and settled:
            candidate = _from_classification(session, item, lane)
        elif packet is None:
            raise EvaluationError("candidate evidence is missing")
        else:
            candidate = packet
        if settled:
            stored = _read_liquidity(root, session, symbol)
            if stored is not None:
                candidate = candidate.model_copy(
                    update={"market": _overlay(session, candidate.market, stored, (), None)}
                )
        else:
            candidate = candidate.model_copy(
                update={"market": _required_market(root, session, symbol, candidate.market, bars)}
            )
        found[symbol] = candidate
    return found


def _from_classification(
    session: date, item: SymbolClassification, lane: LaneFields | None
) -> ScanCandidate:
    facts = to_facts(item, quote_asset="USDT", status="TRADING")
    if lane is not None:
        facts = overlay_lane(
            facts,
            lane=lane.lane,
            circulating_ratio=lane.circulating_ratio,
            history_days=lane.history_days,
            market_cap_rank=lane.market_cap_rank,
            fdv_to_market_cap=lane.fdv_to_market_cap,
            unlock=lane.unlock_state,
        )
    known = None if lane is None else _known(lane.unlock_state)
    return ScanCandidate(
        facts=facts,
        market=_blank(session),
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


def _known(state: str) -> bool | None:
    if state == "unavailable":
        return None
    if state in {"future_unlocks", "no_applicable_future_unlock"}:
        return True
    raise EvaluationError("capture is unusable")


def _required_market(
    root: Path,
    session: date,
    symbol: str,
    market: MarketSnapshot,
    bars: Mapping[str, tuple[DailyBar, ...]],
) -> MarketSnapshot:
    series = bars.get(symbol)
    if series is None:
        raise EvaluationError("completed bars are missing")
    return _overlay(
        session,
        market,
        _required_liquidity(root, session, symbol),
        series,
        _spike(root, session, symbol),
    )


def _required_liquidity(root: Path, session: date, symbol: str) -> dict[str, object]:
    document = _read_liquidity(root, session, symbol)
    if document is not None:
        return document
    return _book_fields(root, session, symbol)


def _book_fields(root: Path, session: date, symbol: str) -> dict[str, object]:
    """The stored book. Depth is not on that file and stays missing."""
    path = _capture(root, session, "book", symbol)
    if not path.is_file():
        raise EvaluationError("book evidence is missing")
    document = _object(path)
    if not isinstance(document, dict):
        raise EvaluationError("capture is unusable")
    spreads = document.get("spread_bps")
    snapshots = document.get("spread_snapshots")
    if not isinstance(spreads, list) or len(spreads) != 1 or type(snapshots) is not int:
        raise EvaluationError("capture is unusable")
    return {"median_spread_bps": spreads[0], "spread_snapshots": snapshots}


def _read_liquidity(root: Path, session: date, symbol: str) -> dict[str, object] | None:
    path = _capture(root, session, "liquidity", symbol)
    if not path.is_file():
        return None
    document = _object(path)
    if not isinstance(document, dict):
        raise EvaluationError("capture is unusable")
    return document


def _spike(root: Path, session: date, symbol: str) -> int | None:
    path = _capture(root, session, "hours", symbol)
    if not path.is_file():
        raise EvaluationError("hour grid is missing")
    document = _object(path)
    count = document.get("spike_candle_count") if isinstance(document, dict) else None
    if count is not None and type(count) is not int:
        raise EvaluationError("capture is unusable")
    return count


def _overlay(
    session: date,
    market: MarketSnapshot,
    document: Mapping[str, object],
    bars: tuple[DailyBar, ...],
    spike: int | None,
) -> MarketSnapshot:
    derived = derive_liquidity(
        bars,
        session,
        binance_quote_volume=_money(document.get("binance_quote_volume_24h")),
        market_cap=_money(document.get("market_cap_usd")),
        aggregate_volume=_money(document.get("aggregate_volume_usd")),
    )
    taker = _taker(bars, session)
    return market.model_copy(
        update={
            "as_of": session_close(session),
            "median_quote_volume_30d_usd": _first(
                document.get("median_quote_volume_30d_usd"), derived.median_quote_volume_30d_usd
            ),
            "day_quote_volume_usd": _first(
                document.get("day_quote_volume_usd"), derived.day_quote_volume_usd
            ),
            "median_spread_bps": _first(
                document.get("median_spread_bps"), market.median_spread_bps
            ),
            "spread_snapshots": _count(document.get("spread_snapshots"), market.spread_snapshots),
            "depth_usd_per_side": _first(
                document.get("depth_usd_per_side"), market.depth_usd_per_side
            ),
            "turnover": _first(document.get("turnover"), market.turnover),
            "volume_zscore": _first(document.get("volume_zscore"), derived.volume_zscore),
            "price_move": _first(document.get("price_move"), derived.price_move),
            "spike_candle_count": spike if spike is not None else market.spike_candle_count,
            "trade_size_stdev": _first(document.get("trade_size_stdev"), derived.trade_size_stdev),
            "taker_buy_ratio": taker if taker is not None else market.taker_buy_ratio,
            "binance_volume_share": _first(
                document.get("binance_volume_share"), market.binance_volume_share
            ),
            "stablecoin_peg_deviation": _first(
                document.get("stablecoin_peg_deviation"), market.stablecoin_peg_deviation
            ),
            "peg_deviation_hours": _count(
                document.get("peg_deviation_hours"), market.peg_deviation_hours
            ),
        }
    )


def _blank(session: date) -> MarketSnapshot:
    return MarketSnapshot(
        as_of=session_close(session),
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
    )


def _first(stored: object, derived: Decimal | None) -> Decimal | None:
    if stored is not None:
        return _money(stored)
    return derived


def _count(stored: object, current: int | None) -> int | None:
    if stored is None:
        return current
    if type(stored) is not int or stored < 0:
        raise EvaluationError("capture is unusable")
    return stored


def _money(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = Decimal(value)
        except InvalidOperation as error:
            raise EvaluationError("capture is unusable") from error
    else:
        raise EvaluationError("capture is unusable")
    if not parsed.is_finite():
        raise EvaluationError("capture is unusable")
    return parsed


def _taker(bars: tuple[DailyBar, ...], session: date) -> Decimal | None:
    matched = [bar for bar in bars if bar.open_date == session]
    if len(matched) != 1 or matched[0].quote_volume <= 0:
        return None
    ratio = matched[0].taker_buy_quote_volume / matched[0].quote_volume
    if ratio < 0 or ratio > 1:
        return None
    return ratio


def _capture(root: Path, session: date, kind: str, symbol: str) -> Path:
    return root / "captures" / f"session={session.isoformat()}" / kind / f"symbol={symbol}.json"


def _object(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise EvaluationError("capture is unusable") from error
