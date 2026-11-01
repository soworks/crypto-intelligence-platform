from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from cip.domain.errors import EvaluationError
from cip.domain.events import EventType
from cip.domain.policy import load_policy
from cip.evaluation.decision import Cohort, Disposition, decision_id
from cip.evaluation.eligibility import CandidateFacts
from cip.evaluation.features import Tokenomics
from cip.evaluation.fundamentals import CmcReading, CoinGeckoReading, UnlockReading
from cip.evaluation.liquidity import MarketSnapshot
from cip.evaluation.regime import PriorSession
from cip.evaluation.scan import ScanCandidate, UniverseSnapshot, run_daily_scan
from cip.evaluation.store import FileDecisionWriter
from cip.history.bars import DailyBar
from cip.recorders.observation import Observation

_POLICY_PATH = Path(__file__).parents[3] / "policies" / "investment-policy.yaml"
POLICY = load_policy(_POLICY_PATH)
UNIVERSE = POLICY.policy.hypotheses.universe
SESSION = date(2026, 10, 3)
CLOSE = datetime(2026, 10, 4, tzinfo=UTC)
SHA = "a" * 40


def _money(value: float) -> Decimal:
    return Decimal(str(value))


def _facts(symbol: str, base: str, **overrides: object) -> CandidateFacts:
    lane = UNIVERSE.normal
    values: dict[str, object] = {
        "symbol": symbol,
        "base_asset": base,
        "quote_asset": "USDT",
        "status": "TRADING",
        "eur_stable": False,
        "fan_token": False,
        "monitoring_tag": False,
        "delisting": False,
        "deposits_suspended": False,
        "withdrawals_suspended": False,
        "pending_migration": False,
        "market_cap_usd": _money(lane.minimum_market_cap_usd),
        "market_cap_rank": lane.market_cap_rank_ceiling,
        "circulating_ratio": _money(lane.minimum_circulating_ratio),
        "fdv_to_market_cap": _money(lane.maximum_fdv_to_market_cap),
        "history_days": lane.minimum_history_days,
        "unlock_schedule_known": None,
    }
    values.update(overrides)
    return CandidateFacts(**values)  # type: ignore[arg-type]


def _market() -> MarketSnapshot:
    normal = UNIVERSE.normal
    rules = UNIVERSE.manipulation
    share = (
        _money(rules.binance_volume_share_below) + _money(rules.binance_volume_share_above)
    ) / 2
    return MarketSnapshot(
        as_of=CLOSE,
        median_quote_volume_30d_usd=_money(normal.median_quote_volume_30d_usd),
        day_quote_volume_usd=_money(normal.minimum_day_quote_volume_usd),
        median_spread_bps=_money(normal.maximum_median_spread_bps),
        spread_snapshots=normal.minimum_spread_snapshots,
        depth_usd_per_side=_money(normal.minimum_depth_usd_per_side),
        turnover=_money(normal.turnover_min),
        volume_zscore=Decimal("0"),
        price_move=Decimal("0"),
        spike_candle_count=rules.spike_candle_ceiling,
        trade_size_stdev=Decimal("0"),
        taker_buy_ratio=Decimal("0"),
        binance_volume_share=share,
        stablecoin_peg_deviation=Decimal("0"),
        peg_deviation_hours=0,
    )


def _candidate(
    symbol: str,
    base: str,
    gecko_id: str,
    *,
    ratio: str = "1",
    **fact_overrides: object,
) -> ScanCandidate:
    return ScanCandidate(
        facts=_facts(symbol, base, **fact_overrides),
        market=_market(),
        coingecko=CoinGeckoReading(
            coingecko_id=gecko_id,
            market_cap_usd=Decimal("100"),
            circulating_supply=Decimal("50"),
            total_supply=Decimal("100"),
            fully_diluted_valuation_usd=Decimal("200"),
            source_timestamp=CLOSE,
        ),
        cmc=CmcReading(market_cap_usd=Decimal("100"), source_timestamp=CLOSE),
        unlocks=UnlockReading(
            schedule_known=True, pct_circ_14d=Decimal("0"), pct_circ_90d=Decimal("0")
        ),
        tokenomics=Tokenomics(
            circulating_ratio=Decimal(ratio),
            fdv_to_market_cap=Decimal("2"),
            unlock_pct_14d=Decimal("0"),
            unlock_pct_90d=Decimal("0"),
        ),
    )


def _bar(symbol: str, day: date, close: str, *, high: str | None = None) -> DailyBar:
    price = Decimal(close)
    peak = Decimal(high) if high is not None else price
    return DailyBar(
        symbol=symbol,
        open_date=day,
        open=price,
        high=peak,
        low=min(price, peak),
        close=price,
        volume=Decimal("1"),
        quote_volume=Decimal("1"),
        trade_count=1,
        taker_buy_base_volume=Decimal("1"),
        taker_buy_quote_volume=Decimal("1"),
    )


def _series(
    symbol: str,
    days: int,
    close: str = "100",
    *,
    last: str | None = None,
    last_days: int = 1,
) -> tuple[DailyBar, ...]:
    bars: list[DailyBar] = []
    for offset in range(days - 1, -1, -1):
        day = SESSION - timedelta(days=offset)
        price = last if last is not None and offset < last_days else close
        bars.append(_bar(symbol, day, price))
    return tuple(bars)


def _observations() -> tuple[Observation, ...]:
    moment = datetime(SESSION.year, SESSION.month, SESSION.day, tzinfo=UTC)
    return (
        Observation(
            series="btc_dominance",
            provider="coingecko",
            source_timestamp=moment,
            observed_at=moment,
            symbol=None,
            values=(("btc_dominance", Decimal("1")),),
            units=(("btc_dominance", "percent"),),
        ),
        Observation(
            series="stablecoin_supply",
            provider="defillama",
            source_timestamp=moment,
            observed_at=moment,
            symbol=None,
            values=(("stablecoin_supply_usd", Decimal("100")),),
            units=(("stablecoin_supply_usd", "usd"),),
        ),
    )


def _snapshot(*symbols: str) -> UniverseSnapshot:
    return UniverseSnapshot(
        session=SESSION,
        symbols=symbols,
        observed_at=CLOSE,
        provenance="exchangeInfo/2026-10-03",
    )


def _scan(tmp_path: Path, **overrides: object) -> object:
    values: dict[str, object] = {
        "session": SESSION,
        "as_of": CLOSE,
        "cohort": Cohort.SHADOW,
        "policy": POLICY,
        "git_sha": SHA,
        "correlation_id": "scan-1",
        "snapshot": _snapshot("SOLUSDT"),
        "candidates": {"SOLUSDT": _candidate("SOLUSDT", "SOL", "solana")},
        "bars": {},
        "observations": (),
        "prior": (),
        "verified_ids": {"SOL": "solana", "ETH": "ethereum"},
        "weights": None,
        "writer": FileDecisionWriter(tmp_path),
    }
    values.update(overrides)
    return run_daily_scan(**values)  # type: ignore[arg-type]


def test_a_lane_rejection_reads_an_observed_book_and_ignores_a_blank_one(tmp_path: Path) -> None:
    blank = _market().model_copy(
        update={name: None for name in MarketSnapshot.model_fields if name != "as_of"}
    )
    thin = _market().model_copy(update={"depth_usd_per_side": Decimal("0")})
    result = _scan(
        tmp_path,
        snapshot=_snapshot("SOLUSDT", "ETHUSDT"),
        candidates={
            "SOLUSDT": _candidate("SOLUSDT", "SOL", "solana", market_cap_rank=None).model_copy(
                update={"market": blank}
            ),
            "ETHUSDT": _candidate("ETHUSDT", "ETH", "ethereum", market_cap_rank=None).model_copy(
                update={"market": thin}
            ),
        },
    )
    by_symbol = {record.symbol: record for record in result.records}
    assert by_symbol["SOLUSDT"].reason_codes == ("missing_market_cap_rank",)
    assert by_symbol["ETHUSDT"].reason_codes == ("missing_market_cap_rank", "depth_below_minimum")


def test_a_session_before_the_utc_close_records_nothing(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="has not closed"):
        _scan(tmp_path, as_of=CLOSE - timedelta(seconds=1))
    assert list(tmp_path.rglob("*.json")) == []


def test_a_missing_snapshot_records_nothing(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="snapshot is missing"):
        _scan(tmp_path, snapshot=None)
    assert list(tmp_path.rglob("*.json")) == []


def test_a_repeated_snapshot_symbol_records_nothing(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="repeats"):
        _scan(tmp_path, snapshot=_snapshot("SOLUSDT", "SOLUSDT"))
    assert list(tmp_path.rglob("*.json")) == []


def test_a_candidate_for_a_different_symbol_records_nothing(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="does not match"):
        _scan(tmp_path, candidates={"SOLUSDT": _candidate("ETHUSDT", "ETH", "ethereum")})
    assert list(tmp_path.rglob("*.json")) == []


def test_rejections_are_recorded_and_weights_are_not_invented(tmp_path: Path) -> None:
    bars = {
        "BTCUSDT": _series("BTCUSDT", 200, last="110"),
        "SOLUSDT": _series("SOLUSDT", 200, last="110"),
        "ETHUSDT": _series("ETHUSDT", 200),
    }
    result = _scan(
        tmp_path,
        snapshot=_snapshot("FDUSDUSDT", "SOLUSDT", "ETHUSDT"),
        candidates={
            "FDUSDUSDT": _candidate("FDUSDUSDT", "FDUSD", "fdusd"),
            "SOLUSDT": _candidate("SOLUSDT", "SOL", "solana"),
            "STRAYUSDT": _candidate("STRAYUSDT", "STRAY", "stray"),
        },
        bars=bars,
        observations=_observations(),
        prior=(
            PriorSession(session=SESSION - timedelta(days=1), raw="RISK_ON", published="RISK_ON"),
        ),
    )
    assert [record.symbol for record in result.records] == ["FDUSDUSDT", "SOLUSDT", "ETHUSDT"]
    stable, scored, missing = result.records
    assert stable.disposition is Disposition.INELIGIBLE
    assert stable.reason_codes == ("stablecoin",)
    assert stable.score is None
    assert scored.disposition is Disposition.INELIGIBLE
    assert scored.reason_codes == ("score_weights_not_frozen",)
    assert scored.regime == "RISK_ON"
    assert scored.features["circulating_ratio"] == Decimal("1")
    assert scored.score is None
    assert scored.rank is None
    assert missing.disposition is Disposition.INELIGIBLE
    assert missing.reason_codes == ("missing_candidate",)
    assert missing.features["ema_20"] == Decimal("100")
    assert missing.score is None
    assert missing.rank is None
    assert result.stored[1].created is True
    assert result.stored[1].key == (
        f"decisions/cohort=SHADOW/date=2026-10-04/symbol=SOLUSDT/{decision_id(scored)}.json"
    )
    assert (tmp_path / result.stored[1].key).read_bytes()
    document = scored.to_document()
    assert document["score"] is None
    assert "order_id" not in document
    assert "quantity" not in document
    assert result.events[1].event_type is EventType.DECISION_RECORDED
    assert result.events[1].payload["decision_key"] == result.stored[1].key
    assert result.events[1].payload["sha256"] == result.stored[1].sha256
    assert result.events[1].asset == "SOLUSDT"
    assert result.events[1].idempotency_key == decision_id(scored)


def test_the_same_scan_keeps_the_first_decision(tmp_path: Path) -> None:
    bars = {"BTCUSDT": _series("BTCUSDT", 200, last="110"), "SOLUSDT": _series("SOLUSDT", 200)}
    kwargs = {
        "bars": bars,
        "observations": _observations(),
        "prior": (
            PriorSession(session=SESSION - timedelta(days=1), raw="RISK_ON", published="RISK_ON"),
        ),
        "writer": FileDecisionWriter(tmp_path),
    }
    first = _scan(tmp_path, **kwargs)
    second = _scan(tmp_path, **kwargs)
    assert first.stored[0].created is True
    assert second.stored[0].created is False
    assert second.stored[0].sha256 == first.stored[0].sha256


def test_a_score_at_the_minimum_is_a_recommendation_and_a_lower_one_is_ranked(
    tmp_path: Path,
) -> None:
    bars = {
        "BTCUSDT": _series("BTCUSDT", 200, last="110"),
        "ETHUSDT": _series("ETHUSDT", 200, last="110"),
        "SOLUSDT": _series("SOLUSDT", 200, last="110"),
    }
    result = _scan(
        tmp_path,
        snapshot=_snapshot("ETHUSDT", "SOLUSDT"),
        candidates={
            "ETHUSDT": _candidate("ETHUSDT", "ETH", "ethereum", ratio="0.5"),
            "SOLUSDT": _candidate("SOLUSDT", "SOL", "solana", ratio="1"),
        },
        bars=bars,
        observations=_observations(),
        prior=(
            PriorSession(session=SESSION - timedelta(days=1), raw="RISK_ON", published="RISK_ON"),
        ),
        weights={"circulating_ratio": Decimal("70")},
    )
    eth, sol = result.records
    assert eth.disposition is Disposition.SCORED
    assert eth.reason_codes == ("below_min_score",)
    assert eth.score == Decimal("35")
    assert eth.rank == 2
    assert sol.disposition is Disposition.BUY
    assert sol.reason_codes == ("buy_recommendation",)
    assert sol.score == Decimal("70")
    assert sol.rank == 1
    assert sol.regime == "RISK_ON"
    assert "order_id" not in sol.to_document()


def test_risk_off_keeps_the_score_and_does_not_recommend(tmp_path: Path) -> None:
    bars = {
        "BTCUSDT": _series("BTCUSDT", 200, last="70"),
        "SOLUSDT": _series("SOLUSDT", 200),
    }
    result = _scan(
        tmp_path,
        bars=bars,
        observations=_observations(),
        weights={"circulating_ratio": Decimal("70")},
    )
    record = result.records[0]
    assert record.regime == "RISK_OFF"
    assert record.disposition is Disposition.SCORED
    assert record.reason_codes == ("risk_off",)
    assert record.score == Decimal("70")
    assert record.rank == 1


def test_neutral_refuses_a_non_positive_30_day_spread(tmp_path: Path) -> None:
    bars = {
        "BTCUSDT": _series("BTCUSDT", 200, last="110"),
        "AAAUSDT": _series("AAAUSDT", 60, last="110"),
        "BBBUSDT": _series("BBBUSDT", 60),
        "CCCUSDT": _series("CCCUSDT", 60),
        "SOLUSDT": _series("SOLUSDT", 200),
    }
    result = _scan(
        tmp_path,
        bars=bars,
        observations=_observations(),
        prior=(
            PriorSession(session=SESSION - timedelta(days=1), raw="NEUTRAL", published="NEUTRAL"),
        ),
        weights={"circulating_ratio": Decimal("100")},
    )
    record = result.records[0]
    assert record.regime == "NEUTRAL"
    assert record.disposition is Disposition.SCORED
    assert record.reason_codes == ("rs_30d_not_positive",)
    assert record.features["rs_30d_skip_1"] <= 0


def test_neutral_can_recommend_when_the_spread_is_positive(tmp_path: Path) -> None:
    bars = {
        "BTCUSDT": _series("BTCUSDT", 200, last="110", last_days=5),
        "AAAUSDT": _series("AAAUSDT", 60),
        "BBBUSDT": _series("BBBUSDT", 60),
        "SOLUSDT": _series("SOLUSDT", 200, last="200", last_days=5),
    }
    result = _scan(
        tmp_path,
        bars=bars,
        observations=_observations(),
        prior=(
            PriorSession(session=SESSION - timedelta(days=1), raw="NEUTRAL", published="NEUTRAL"),
        ),
        weights={"circulating_ratio": Decimal("100")},
    )
    record = result.records[0]
    assert record.regime == "NEUTRAL"
    assert record.disposition is Disposition.BUY
    assert record.reason_codes == ("buy_recommendation",)
    assert record.features["rs_30d_skip_1"] > 0


def test_neutral_refuses_a_missing_30_day_spread(tmp_path: Path) -> None:
    bars = {
        "BTCUSDT": _series("BTCUSDT", 200, last="110"),
        "AAAUSDT": _series("AAAUSDT", 60, last="110"),
        "BBBUSDT": _series("BBBUSDT", 60),
        "CCCUSDT": _series("CCCUSDT", 60),
        "SOLUSDT": _series("SOLUSDT", 10),
    }
    result = _scan(
        tmp_path,
        bars=bars,
        observations=_observations(),
        prior=(
            PriorSession(session=SESSION - timedelta(days=1), raw="NEUTRAL", published="NEUTRAL"),
        ),
        weights={"circulating_ratio": Decimal("100")},
    )
    record = result.records[0]
    assert record.regime == "NEUTRAL"
    assert record.disposition is Disposition.SCORED
    assert record.reason_codes == ("missing_rs_30d",)
    assert "rs_30d_skip_1" not in record.features


def test_a_fundamentals_mismatch_is_recorded(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        candidates={"SOLUSDT": _candidate("SOLUSDT", "SOL", "not-solana")},
    )
    record = result.records[0]
    assert record.reason_codes == ("coingecko_id_mismatch",)
    assert record.score is None
    assert record.disposition is Disposition.INELIGIBLE


def test_an_illegal_weight_is_a_refusal_not_a_score(tmp_path: Path) -> None:
    bars = {
        "BTCUSDT": _series("BTCUSDT", 200, last="110"),
        "SOLUSDT": _series("SOLUSDT", 200, last="110"),
    }
    result = _scan(
        tmp_path,
        bars=bars,
        observations=_observations(),
        prior=(
            PriorSession(session=SESSION - timedelta(days=1), raw="RISK_ON", published="RISK_ON"),
        ),
        weights={"liquidity": Decimal("1")},
    )
    record = result.records[0]
    assert record.disposition is Disposition.INELIGIBLE
    assert record.reason_codes == ("score_refused",)
    assert record.score is None


def test_a_passing_symbol_without_a_published_regime_is_not_scored(tmp_path: Path) -> None:
    result = _scan(tmp_path, weights={"circulating_ratio": Decimal("70")})
    record = result.records[0]
    assert record.disposition is Disposition.INELIGIBLE
    assert record.regime is None
    assert "missing_btc_sma" in record.reason_codes
    assert record.score is None


def test_every_snapshot_symbol_is_one_decision_including_rejections(tmp_path: Path) -> None:
    symbols = ("FDUSDUSDT", "SOLUSDT", "ETHUSDT", "MISSINGUSDT")
    bars = {
        "BTCUSDT": _series("BTCUSDT", 200, last="110"),
        "SOLUSDT": _series("SOLUSDT", 200, last="110"),
        "ETHUSDT": _series("ETHUSDT", 200),
        "MISSINGUSDT": _series("MISSINGUSDT", 200),
    }
    result = _scan(
        tmp_path,
        snapshot=_snapshot(*symbols),
        candidates={
            "FDUSDUSDT": _candidate("FDUSDUSDT", "FDUSD", "fdusd"),
            "SOLUSDT": _candidate("SOLUSDT", "SOL", "solana"),
            "ETHUSDT": _candidate("ETHUSDT", "ETH", "ethereum"),
        },
        bars=bars,
        observations=_observations(),
        prior=(
            PriorSession(session=SESSION - timedelta(days=1), raw="RISK_ON", published="RISK_ON"),
        ),
    )
    assert [record.symbol for record in result.records] == list(symbols)
    assert len({record.symbol for record in result.records}) == len(symbols)
    assert {record.disposition for record in result.records} == {Disposition.INELIGIBLE}
    written = {path.parent.name.removeprefix("symbol=") for path in tmp_path.rglob("*.json")}
    assert written == set(symbols)


def test_identical_stored_inputs_write_byte_identical_decisions(tmp_path: Path) -> None:
    bars = {
        "BTCUSDT": _series("BTCUSDT", 200, last="110"),
        "SOLUSDT": _series("SOLUSDT", 200, last="110"),
        "ETHUSDT": _series("ETHUSDT", 200),
    }
    shared = {
        "snapshot": _snapshot("FDUSDUSDT", "SOLUSDT", "ETHUSDT"),
        "candidates": {
            "FDUSDUSDT": _candidate("FDUSDUSDT", "FDUSD", "fdusd"),
            "SOLUSDT": _candidate("SOLUSDT", "SOL", "solana"),
        },
        "bars": bars,
        "observations": _observations(),
        "prior": (
            PriorSession(session=SESSION - timedelta(days=1), raw="RISK_ON", published="RISK_ON"),
        ),
    }
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left = _scan(left_root, writer=FileDecisionWriter(left_root), **shared)
    right = _scan(right_root, writer=FileDecisionWriter(right_root), **shared)
    assert left.records == right.records
    for stored, again in zip(left.stored, right.stored, strict=True):
        assert stored.key == again.key
        assert stored.sha256 == again.sha256
        assert decision_id(stored.record) == decision_id(again.record)
        assert (left_root / stored.key).read_bytes() == (right_root / again.key).read_bytes()
        document = stored.record.to_document()
        assert document["reason_codes"] == list(stored.record.reason_codes)
        assert document["rank"] == stored.record.rank
        assert document["regime"] == stored.record.regime
        assert document["raw_regime"] == stored.record.raw_regime
        assert document["score"] == (
            None if stored.record.score is None else format(stored.record.score, "f")
        )
    source = Path(run_daily_scan.__code__.co_filename).read_text()
    for forbidden in ("cip.adapters", "boto3", "urllib", "requests"):
        assert forbidden not in source


def test_hysteresis_keeps_the_raw_regime_on_the_decision(tmp_path: Path) -> None:
    bars = {
        "BTCUSDT": _series("BTCUSDT", 200, last="110"),
        "SOLUSDT": _series("SOLUSDT", 200, last="110"),
        "ETHUSDT": _series("ETHUSDT", 200, last="110"),
    }
    withheld = _scan(
        tmp_path,
        bars=bars,
        observations=_observations(),
        prior=(),
        weights={"circulating_ratio": Decimal("1")},
    )
    opening = withheld.records[0]
    assert opening.disposition is Disposition.INELIGIBLE
    assert opening.regime is None
    assert opening.raw_regime == "RISK_ON"
    assert opening.reason_codes == ("hysteresis",)
    held_root = tmp_path / "held"
    held = _scan(
        held_root,
        writer=FileDecisionWriter(held_root),
        bars={
            "BTCUSDT": _series("BTCUSDT", 200, last="99"),
            "SOLUSDT": _series("SOLUSDT", 200, last="110"),
            "ETHUSDT": _series("ETHUSDT", 200, last="110"),
        },
        observations=_observations(),
        prior=(
            PriorSession(session=SESSION - timedelta(days=1), raw="NEUTRAL", published="RISK_OFF"),
        ),
        weights={"circulating_ratio": Decimal("1")},
    )
    decision = held.records[0]
    assert decision.disposition is Disposition.SCORED
    assert decision.reason_codes == ("risk_off",)
    assert decision.regime == "RISK_OFF"
    assert decision.raw_regime == "NEUTRAL"
    assert decision.to_document()["raw_regime"] == "NEUTRAL"


def test_the_scan_does_not_import_a_provider() -> None:
    source = Path(run_daily_scan.__code__.co_filename).read_text()
    assert "cip.adapters" not in source
    assert "boto3" not in source
    assert "urllib" not in source


def test_a_snapshot_for_another_session_records_nothing(tmp_path: Path) -> None:
    snapshot = UniverseSnapshot(
        session=SESSION - timedelta(days=1),
        symbols=("SOLUSDT",),
        observed_at=CLOSE,
        provenance="exchangeInfo/2026-10-02",
    )
    with pytest.raises(EvaluationError, match="different session"):
        _scan(tmp_path, snapshot=snapshot)
    assert list(tmp_path.rglob("*.json")) == []


def test_a_blank_correlation_id_records_nothing(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="correlation"):
        _scan(tmp_path, correlation_id="")
    assert list(tmp_path.rglob("*.json")) == []


def test_a_naive_clock_records_nothing(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="UTC"):
        _scan(tmp_path, as_of=datetime(2026, 10, 4))  # noqa: DTZ001
    assert list(tmp_path.rglob("*.json")) == []


def test_snapshot_fields_reject_a_blank_provenance() -> None:
    with pytest.raises(ValidationError):
        UniverseSnapshot(session=SESSION, symbols=("SOLUSDT",), observed_at=CLOSE, provenance="")
