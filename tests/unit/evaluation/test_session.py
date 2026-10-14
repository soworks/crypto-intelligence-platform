from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from cip.domain.errors import EvaluationError
from cip.evaluation.eligibility import CandidateFacts
from cip.evaluation.fundamentals import CmcReading, CoinGeckoReading, UnlockReading
from cip.evaluation.liquidity import MarketSnapshot
from cip.evaluation.scan import ScanCandidate, UniverseSnapshot
from cip.evaluation.session import InputPresence, SessionReadiness, assess_session, session_close
from cip.history.bars import DailyBar
from cip.recorders.observation import Observation


def test_the_close_is_the_next_utc_midnight() -> None:
    assert session_close(date(2026, 10, 5)) == datetime(2026, 10, 6, tzinfo=UTC)


def test_presence_names_the_four_states() -> None:
    assert [item.value for item in InputPresence] == [
        "present",
        "absent",
        "not_produced",
        "lookahead",
    ]


def test_a_boolean_session_is_refused() -> None:
    with pytest.raises(EvaluationError, match="session is a date"):
        session_close(True)  # type: ignore[arg-type]


def test_weights_are_named_without_blocking_the_session() -> None:
    result = SessionReadiness(
        ready=True,
        blocks=(),
        decision_notes=("score_weights_not_frozen",),
        score_weights="absent",
    )
    assert result.ready is True
    assert result.score_weights == "absent"


def test_a_block_is_not_ready() -> None:
    result = SessionReadiness(
        ready=False,
        blocks=("snapshot_not_produced",),
        decision_notes=(),
        score_weights="absent",
    )
    assert result.ready is False


SESSION = date(2026, 10, 5)
CLOSE = datetime(2026, 10, 6, tzinfo=UTC)
SYMBOL = "SOLUSDT"


def _bar(symbol: str, day: date) -> DailyBar:
    price = Decimal("10")
    return DailyBar(
        symbol=symbol,
        open_date=day,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1"),
        quote_volume=Decimal("1"),
        trade_count=1,
        taker_buy_base_volume=Decimal("1"),
        taker_buy_quote_volume=Decimal("1"),
    )


def _candidate(
    *,
    symbol: str = SYMBOL,
    base: str = "SOL",
    as_of: datetime = CLOSE,
    gecko_cap: Decimal | None = Decimal("1"),
    gecko_stamp: datetime | None = CLOSE,
    cmc_cap: Decimal | None = None,
    cmc_stamp: datetime | None = None,
) -> ScanCandidate:
    blanks: dict[str, object] = {
        "status": None,
        "eur_stable": None,
        "fan_token": None,
        "monitoring_tag": None,
        "delisting": None,
        "deposits_suspended": None,
        "withdrawals_suspended": None,
        "pending_migration": None,
        "market_cap_usd": None,
        "market_cap_rank": None,
        "circulating_ratio": None,
        "fdv_to_market_cap": None,
        "history_days": None,
        "unlock_schedule_known": None,
    }
    return ScanCandidate(
        facts=CandidateFacts(symbol=symbol, base_asset=base, quote_asset="USDT", **blanks),  # type: ignore[arg-type]
        market=MarketSnapshot(
            as_of=as_of,
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
            coingecko_id="solana",
            market_cap_usd=gecko_cap,
            circulating_supply=None,
            total_supply=None,
            fully_diluted_valuation_usd=None,
            source_timestamp=gecko_stamp,
        ),
        cmc=CmcReading(market_cap_usd=cmc_cap, source_timestamp=cmc_stamp),
        unlocks=UnlockReading(schedule_known=None, pct_circ_14d=None, pct_circ_90d=None),
        tokenomics=None,
    )


def _observation(series: str, day: date, source: datetime | None) -> Observation:
    moment = datetime(day.year, day.month, day.day, tzinfo=UTC)
    if series == "funding":
        return Observation(
            series=series,
            provider="binance",
            source_timestamp=source,
            observed_at=moment,
            symbol="BTCUSDT",
            values=(("funding_rate", Decimal("0.0001")),),
            units=(("funding_rate", "rate"),),
        )
    name = "btc_dominance" if series == "btc_dominance" else "stablecoin_supply_usd"
    unit = "percent" if series == "btc_dominance" else "usd"
    provider = "coingecko" if series == "btc_dominance" else "defillama"
    return Observation(
        series=series,
        provider=provider,
        source_timestamp=source,
        observed_at=moment,
        symbol=None,
        values=((name, Decimal("1")),),
        units=((name, unit),),
    )


def _assess(**overrides: object) -> SessionReadiness:
    values: dict[str, object] = {
        "session": SESSION,
        "as_of": CLOSE,
        "snapshot": UniverseSnapshot(
            session=SESSION,
            symbols=(SYMBOL,),
            observed_at=CLOSE,
            provenance="stored-exchange-info",
        ),
        "bars": {
            "BTCUSDT": (_bar("BTCUSDT", SESSION),),
            SYMBOL: (_bar(SYMBOL, SESSION),),
        },
        "bar_months_present": {"BTCUSDT": True, SYMBOL: True},
        "candidates": {SYMBOL: _candidate()},
        "candidate_files_present": {SYMBOL: True},
        "observations": (
            _observation("btc_dominance", SESSION, CLOSE),
            _observation("stablecoin_supply", SESSION, None),
        ),
        "regime_failures": frozenset(),
        "weights_present": True,
    }
    values.update(overrides)
    return assess_session(**values)  # type: ignore[arg-type]


def test_a_ready_session_cannot_carry_a_block() -> None:
    with pytest.raises(EvaluationError, match="a ready session has no blocks"):
        SessionReadiness(
            ready=True,
            blocks=("snapshot_not_produced",),
            decision_notes=(),
            score_weights="absent",
        )


def test_an_open_session_is_not_ready() -> None:
    result = _assess(as_of=CLOSE - timedelta(seconds=1))
    assert result.ready is False
    assert result.blocks == ("session_not_closed",)


def test_a_missing_snapshot_blocks_the_session() -> None:
    result = _assess(snapshot=None)
    assert result.blocks == ("snapshot_not_produced",)


def test_a_snapshot_observed_after_the_close_is_lookahead() -> None:
    later = UniverseSnapshot(
        session=SESSION,
        symbols=(SYMBOL,),
        observed_at=CLOSE + timedelta(seconds=1),
        provenance="stored-exchange-info",
    )
    assert _assess(snapshot=later).blocks == ("snapshot_lookahead",)


def test_a_snapshot_for_another_session_is_not_this_session() -> None:
    other = UniverseSnapshot(
        session=date(2026, 10, 4),
        symbols=(SYMBOL,),
        observed_at=CLOSE,
        provenance="stored-exchange-info",
    )
    assert _assess(snapshot=other).blocks == ("snapshot_session_mismatch",)


def test_a_repeated_symbol_blocks_even_though_the_snapshot_model_stores_it() -> None:
    repeated = UniverseSnapshot(
        session=SESSION,
        symbols=(SYMBOL, SYMBOL),
        observed_at=CLOSE,
        provenance="stored-exchange-info",
    )
    assert repeated.symbols == (SYMBOL, SYMBOL)
    assert _assess(snapshot=repeated).blocks == ("snapshot_repeats_symbol",)


def test_missing_btc_month_blocks_the_session() -> None:
    present = {"BTCUSDT": False, SYMBOL: True}
    assert _assess(bar_months_present=present).blocks == ("btc_bars_not_produced",)


def test_a_btc_month_without_the_session_bar_is_not_the_next_day() -> None:
    bars = {"BTCUSDT": (_bar("BTCUSDT", date(2026, 10, 4)),), SYMBOL: (_bar(SYMBOL, SESSION),)}
    assert _assess(bars=bars).blocks == ("btc_session_bar_absent",)


def test_an_open_symbol_still_requires_btc_bars_after_btc_settles() -> None:
    snapshot = UniverseSnapshot(
        session=SESSION,
        symbols=("BTCUSDT", SYMBOL),
        observed_at=CLOSE,
        provenance="stored-exchange-info",
    )
    result = _assess(
        snapshot=snapshot,
        bars={SYMBOL: (_bar(SYMBOL, SESSION),)},
        bar_months_present={SYMBOL: True},
        candidates={SYMBOL: _candidate()},
        candidate_files_present={SYMBOL: True},
        settled={"BTCUSDT": ("policy_rejected", ("fan_token",))},
    )
    assert result.blocks == ("btc_bars_not_produced",)


def test_a_settled_symbol_still_refuses_a_lookahead_bar() -> None:
    result = _assess(
        bars={
            "BTCUSDT": (_bar("BTCUSDT", SESSION),),
            SYMBOL: (_bar(SYMBOL, SESSION + timedelta(days=1)),),
        },
        bar_months_present={"BTCUSDT": True, SYMBOL: True},
        candidates={},
        candidate_files_present={},
        settled={SYMBOL: ("policy_rejected", ("fan_token",))},
    )
    assert result.blocks == (f"bars_lookahead:{SYMBOL}",)


def test_a_policy_rejection_does_not_require_downstream_bars() -> None:
    result = _assess(
        bars={"BTCUSDT": (_bar("BTCUSDT", SESSION),)},
        bar_months_present={"BTCUSDT": True},
        candidates={},
        candidate_files_present={},
        settled={SYMBOL: ("policy_rejected", ("fan_token", "missing_eur_stable_classification"))},
    )
    assert result.ready is True
    assert result.blocks == ()
    assert f"policy_rejected:{SYMBOL}:fan_token" in result.decision_notes
    assert f"evidence_blocked:{SYMBOL}:missing_eur_stable_classification" in result.decision_notes


def test_an_evidence_block_is_not_recorded_as_a_policy_rejection() -> None:
    result = _assess(
        bars={"BTCUSDT": (_bar("BTCUSDT", SESSION),)},
        bar_months_present={"BTCUSDT": True},
        candidates={},
        candidate_files_present={},
        settled={SYMBOL: ("evidence_blocked", ("missing_unlock_schedule",))},
    )
    assert result.ready is True
    assert f"evidence_blocked:{SYMBOL}:missing_unlock_schedule" in result.decision_notes
    assert not any(note.startswith(f"policy_rejected:{SYMBOL}:") for note in result.decision_notes)


def test_a_settled_symbol_still_refuses_a_later_packet() -> None:
    candidate = _candidate()
    market = candidate.market.model_copy(update={"as_of": CLOSE + timedelta(seconds=1)})
    later = candidate.model_copy(update={"market": market})
    result = _assess(
        candidates={SYMBOL: later},
        candidate_files_present={SYMBOL: True},
        settled={SYMBOL: ("policy_rejected", ("fan_token",))},
    )
    assert result.blocks == (f"candidate_lookahead:{SYMBOL}",)


def test_a_symbol_without_a_month_file_is_not_produced() -> None:
    present = {"BTCUSDT": True, SYMBOL: False}
    assert _assess(bar_months_present=present).blocks == (f"bars_not_produced:{SYMBOL}",)


def test_a_stored_month_without_the_session_bar_stays_a_decision_note() -> None:
    bars = {"BTCUSDT": (_bar("BTCUSDT", SESSION),), SYMBOL: ()}
    result = _assess(bars=bars)
    assert result.ready is True
    assert result.blocks == ()
    assert result.decision_notes == (f"daily_bar_absent:{SYMBOL}",)


def test_an_explicit_daily_bar_absence_is_not_a_missing_producer() -> None:
    present = {"BTCUSDT": True, SYMBOL: False}
    result = _assess(bar_months_present=present, bar_absences={SYMBOL: True})
    assert result.ready is True
    assert result.blocks == ()
    assert result.decision_notes == (f"daily_bar_absent:{SYMBOL}",)


def test_an_explicit_btc_bar_absence_still_blocks_the_session() -> None:
    present = {"BTCUSDT": False, SYMBOL: True}
    result = _assess(bar_months_present=present, bar_absences={"BTCUSDT": True})
    assert result.blocks == ("btc_bars_not_produced",)


def test_a_missing_candidate_file_is_not_an_absence() -> None:
    result = _assess(candidate_files_present={SYMBOL: False})
    assert result.blocks == (f"candidate_not_produced:{SYMBOL}",)


def test_an_explicit_candidate_absence_stays_a_decision_note() -> None:
    result = _assess(candidates={SYMBOL: None})
    assert result.ready is True
    assert result.decision_notes == (f"missing_candidate:{SYMBOL}",)


def test_a_market_clock_after_the_close_is_lookahead() -> None:
    later = _candidate(as_of=CLOSE + timedelta(seconds=1))
    assert _assess(candidates={SYMBOL: later}).blocks == (f"candidate_lookahead:{SYMBOL}",)


def test_an_undated_market_cap_is_not_the_close() -> None:
    packet = _candidate(gecko_stamp=None)
    assert _assess(candidates={SYMBOL: packet}).blocks == (f"undated_fundamental:{SYMBOL}",)


def test_a_fundamental_timestamp_after_the_close_is_lookahead() -> None:
    packet = _candidate(cmc_stamp=CLOSE + timedelta(seconds=1))
    assert _assess(candidates={SYMBOL: packet}).blocks == (f"candidate_lookahead:{SYMBOL}",)


def test_a_bar_after_the_session_is_lookahead() -> None:
    bars = {
        "BTCUSDT": (_bar("BTCUSDT", SESSION), _bar("BTCUSDT", date(2026, 10, 6))),
        SYMBOL: (_bar(SYMBOL, SESSION),),
    }
    assert _assess(bars=bars).blocks == ("bars_lookahead:BTCUSDT",)


def test_regime_observations_for_the_session_do_not_block() -> None:
    result = _assess()
    assert result.ready is True
    assert result.blocks == ()


def test_a_regime_observation_from_the_next_day_is_lookahead() -> None:
    observations = (
        _observation("btc_dominance", date(2026, 10, 6), CLOSE + timedelta(days=1)),
        _observation("stablecoin_supply", SESSION, CLOSE),
    )
    assert _assess(observations=observations).blocks == ("regime_lookahead:btc_dominance",)


def test_a_regime_source_after_the_close_is_lookahead() -> None:
    observations = (
        _observation("btc_dominance", SESSION, CLOSE + timedelta(seconds=1)),
        _observation("stablecoin_supply", SESSION, CLOSE),
    )
    assert _assess(observations=observations).blocks == ("regime_lookahead:btc_dominance",)


def test_a_missing_regime_series_is_not_produced() -> None:
    observations = (_observation("stablecoin_supply", SESSION, CLOSE),)
    result = _assess(observations=observations)
    assert result.blocks == ("regime_not_produced:btc_dominance",)
    stablecoin = (_observation("btc_dominance", SESSION, CLOSE),)
    assert _assess(observations=stablecoin).blocks == ("regime_not_produced:stablecoin_supply",)


def test_a_regime_collection_failure_blocks_without_an_observation() -> None:
    observations = (_observation("stablecoin_supply", SESSION, CLOSE),)
    result = _assess(observations=observations, regime_failures=frozenset({"btc_dominance"}))
    assert result.blocks == ("regime_failed:btc_dominance",)


def test_funding_and_open_interest_do_not_affect_readiness() -> None:
    observations = (
        _observation("btc_dominance", SESSION, CLOSE),
        _observation("stablecoin_supply", SESSION, CLOSE),
        _observation("funding", SESSION, CLOSE + timedelta(days=1)),
    )
    result = _assess(
        observations=observations,
        regime_failures=frozenset({"funding", "open_interest"}),
    )
    assert result.ready is True
    assert result.blocks == ()
    assert "funding" not in " ".join(result.decision_notes)


def test_absent_weights_do_not_block_a_ready_session() -> None:
    result = _assess(weights_present=False)
    assert result.ready is True
    assert result.score_weights == "absent"
    assert result.decision_notes == ("score_weights_not_frozen",)


def test_present_weights_leave_the_weight_note_off() -> None:
    result = _assess(weights_present=True)
    assert result.score_weights == "present"
    assert result.decision_notes == ()


def test_a_candidate_for_another_symbol_does_not_look_ready() -> None:
    packet = _candidate()
    snapshot = UniverseSnapshot(
        session=SESSION,
        symbols=("ETHUSDT",),
        observed_at=CLOSE,
        provenance="stored-exchange-info",
    )
    result = _assess(
        snapshot=snapshot,
        bars={"BTCUSDT": (_bar("BTCUSDT", SESSION),), "ETHUSDT": (_bar("ETHUSDT", SESSION),)},
        bar_months_present={"BTCUSDT": True, "ETHUSDT": True},
        candidates={"ETHUSDT": packet},
        candidate_files_present={"ETHUSDT": True},
    )
    assert result.blocks == ("candidate_symbol_mismatch:ETHUSDT",)


def test_a_bar_stored_under_another_symbol_does_not_look_ready() -> None:
    bars = {"BTCUSDT": (_bar("ETHUSDT", SESSION),), SYMBOL: (_bar(SYMBOL, SESSION),)}
    assert _assess(bars=bars).blocks == ("bar_symbol_mismatch:BTCUSDT",)


def test_a_duplicate_bar_date_does_not_look_ready() -> None:
    bars = {
        "BTCUSDT": (_bar("BTCUSDT", SESSION), _bar("BTCUSDT", SESSION)),
        SYMBOL: (_bar(SYMBOL, SESSION),),
    }
    assert _assess(bars=bars).blocks == ("duplicate_bar_date:BTCUSDT",)


def test_an_empty_reason_is_refused() -> None:
    with pytest.raises(EvaluationError, match="non-empty"):
        SessionReadiness(ready=False, blocks=("",), decision_notes=(), score_weights="absent")


def test_btc_in_the_snapshot_is_not_checked_twice() -> None:
    snapshot = UniverseSnapshot(
        session=SESSION,
        symbols=("BTCUSDT", SYMBOL),
        observed_at=CLOSE,
        provenance="stored-exchange-info",
    )
    result = _assess(
        snapshot=snapshot,
        candidates={"BTCUSDT": _candidate(symbol="BTCUSDT", base="BTC"), SYMBOL: _candidate()},
        candidate_files_present={"BTCUSDT": True, SYMBOL: True},
    )
    assert result.ready is True
    assert result.blocks == ()


def test_a_naive_snapshot_clock_is_refused() -> None:
    snapshot = UniverseSnapshot(
        session=SESSION,
        symbols=(SYMBOL,),
        observed_at=datetime(2026, 10, 6),  # noqa: DTZ001
        provenance="stored-exchange-info",
    )
    with pytest.raises(EvaluationError, match="observed_at must be timezone-aware UTC"):
        _assess(snapshot=snapshot)


def test_a_naive_clock_is_refused() -> None:
    with pytest.raises(EvaluationError, match="as_of must be timezone-aware UTC"):
        _assess(as_of=datetime(2026, 10, 6))  # noqa: DTZ001
    later = datetime(2026, 10, 6, tzinfo=timezone(timedelta(hours=1)))
    with pytest.raises(EvaluationError, match="as_of must be timezone-aware UTC"):
        _assess(as_of=later)
