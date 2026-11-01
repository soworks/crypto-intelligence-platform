import json
import os
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

import cip.evaluation.populate as population
from cip.domain.errors import EvaluationError
from cip.evaluation.eligibility import CandidateFacts
from cip.evaluation.fundamentals import CmcReading, CoinGeckoReading, UnlockReading
from cip.evaluation.liquidity import MarketSnapshot
from cip.evaluation.populate import (
    AbsenceCapture,
    BarCapture,
    CaptureClock,
    FailureCapture,
    PacketCapture,
    RegimeCapture,
    SessionCaptures,
    UniverseCapture,
    populate_session,
    replay_session,
)
from cip.evaluation.scan import ScanCandidate
from cip.history.bars import DailyBar
from cip.recorders.observation import CollectionFailure, Observation

SESSION = date(2026, 10, 5)
CLOSE = datetime(2026, 10, 6, tzinfo=UTC)
LATER = CLOSE + timedelta(seconds=1)
SYMBOL = "SOLUSDT"
OTHER = "ETHUSDT"
SHA = "a" * 40


def _clock(
    *,
    source: str = "binance",
    source_timestamp: datetime | None = CLOSE,
    captured_at: datetime = CLOSE,
) -> CaptureClock:
    return CaptureClock(source, source_timestamp, captured_at)


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


def _history(symbol: str, *, days: int = 200, end: date = SESSION) -> tuple[DailyBar, ...]:
    return tuple(_bar(symbol, end - timedelta(days=offset)) for offset in range(days - 1, -1, -1))


def _packet(
    symbol: str = SYMBOL,
    *,
    as_of: datetime = CLOSE,
    gecko_cap: Decimal | None = Decimal("1"),
    gecko_stamp: datetime | None = CLOSE,
    cmc_cap: Decimal | None = Decimal("1"),
    cmc_stamp: datetime | None = CLOSE,
    clock: CaptureClock | None = None,
) -> PacketCapture:
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
    candidate = ScanCandidate(
        facts=CandidateFacts(
            symbol=symbol,
            base_asset=symbol.removesuffix("USDT"),
            quote_asset="USDT",
            **blanks,  # type: ignore[arg-type]
        ),
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
            coingecko_id=symbol.lower(),
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
    return PacketCapture(symbol, candidate, clock or _clock(source="coingecko"))


def _session_observation(
    series: str, observed_at: datetime, source: datetime | None
) -> Observation:
    name = "btc_dominance" if series == "btc_dominance" else "stablecoin_supply_usd"
    unit = "percent" if series == "btc_dominance" else "usd"
    provider = "coingecko" if series == "btc_dominance" else "defillama"
    return Observation(
        series=series,
        provider=provider,
        source_timestamp=source,
        observed_at=observed_at,
        symbol=None,
        values=((name, Decimal("1")),),
        units=((name, unit),),
    )


def _observation(series: str, source: datetime | None) -> Observation:
    moment = datetime(SESSION.year, SESSION.month, SESSION.day, 12, tzinfo=UTC)
    if series == "funding":
        return Observation(
            series="funding",
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


def _ready(*, symbols: tuple[str, ...] = (SYMBOL,), days: int = 200) -> SessionCaptures:
    return SessionCaptures(
        universe=UniverseCapture(symbols, CLOSE, _clock(source="binance/exchangeInfo")),
        packets=tuple(_packet(symbol) for symbol in symbols),
        candidate_absences=(),
        bars=(
            BarCapture("BTCUSDT", _history("BTCUSDT", days=days), _clock(source="binance/klines")),
            *(
                BarCapture(symbol, _history(symbol, days=days), _clock(source="binance/klines"))
                for symbol in symbols
            ),
        ),
        bar_absences=(),
        regime=(
            RegimeCapture(_observation("btc_dominance", CLOSE), CLOSE),
            RegimeCapture(_observation("stablecoin_supply", None), CLOSE),
        ),
        regime_failures=(),
    )


def _populate(tmp_path: Path, captures: SessionCaptures, **overrides: object) -> object:
    values: dict[str, object] = {
        "root": tmp_path,
        "session": SESSION,
        "as_of": CLOSE,
        "captures": captures,
        "git_sha": SHA,
        "weights_present": False,
    }
    values.update(overrides)
    return populate_session(**values)  # type: ignore[arg-type]


def _manifest(tmp_path: Path) -> Path:
    return tmp_path / "sessions" / "date=2026-10-05" / "manifest.json"


def _stored(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_a_closed_capture_becomes_ready_and_names_the_frozen_inputs(tmp_path: Path) -> None:
    result = _populate(tmp_path, _ready())
    assert result.readiness.ready is True
    assert result.readiness.blocks == ()
    assert result.readiness.decision_notes == ("score_weights_not_frozen",)
    assert result.readiness.score_weights == "absent"
    manifest = result.manifest
    assert manifest is not None
    assert manifest.session_date == SESSION
    assert manifest.session_close_at == CLOSE
    assert manifest.finalized_at == CLOSE
    assert manifest.input_manifest_version == 2
    assert manifest.liquidity_inputs_sha256 is not None
    assert manifest.hour_grid_sha256 is not None
    assert manifest.book_inputs_sha256 is not None
    assert manifest.git_sha == SHA
    assert manifest.score_weights == "absent"
    assert len(manifest.universe_snapshot_sha256) == 64
    assert manifest.provenance[0].transformation == "session-input/v1"
    assert manifest.provenance[0].session_close_at == CLOSE
    bar_file = tmp_path / "sessions" / "date=2026-10-05" / "bars" / "symbol=BTCUSDT.json"
    document = json.loads(bar_file.read_text())
    assert len(document["bars"]) == 200
    assert document["bars"][0]["open_date"] == "2026-03-20"
    assert document["bars"][-1]["open_date"] == "2026-10-05"
    assert document["captured_at"] == "2026-10-06T00:00:00Z"
    replay = replay_session(tmp_path, SESSION)
    assert replay.readiness.ready is True
    assert replay.manifest == manifest


def test_a_new_seal_hashes_liquidity_and_the_hour_grid(tmp_path: Path) -> None:
    folder = tmp_path / "captures" / "session=2026-10-05"
    liquidity = folder / "liquidity" / "symbol=SOLUSDT.json"
    hours = folder / "hours" / "symbol=SOLUSDT.json"
    liquidity.parent.mkdir(parents=True)
    hours.parent.mkdir(parents=True)
    liquidity.write_text('{"depth_usd_per_side":"1"}')
    hours.write_text('{"spike_candle_count":0}')
    result = _populate(tmp_path, _ready(days=1))
    manifest = result.manifest
    assert manifest is not None
    assert manifest.input_manifest_version == 2
    assert manifest.liquidity_inputs_sha256 is not None
    assert manifest.hour_grid_sha256 is not None
    assert manifest.book_inputs_sha256 is not None
    assert len(manifest.liquidity_inputs_sha256) == 64
    assert len(manifest.hour_grid_sha256) == 64
    assert len(manifest.book_inputs_sha256) == 64
    sealed = _manifest(tmp_path).read_bytes()
    liquidity.write_text('{"depth_usd_per_side":"2"}')
    with pytest.raises(EvaluationError, match="does not match its inputs"):
        replay_session(tmp_path, SESSION)
    liquidity.write_text('{"depth_usd_per_side":"1"}')
    hours.unlink()
    with pytest.raises(EvaluationError, match="does not match its inputs"):
        replay_session(tmp_path, SESSION)
    assert _manifest(tmp_path).read_bytes() == sealed


def test_a_version_one_manifest_is_replayed_and_not_rewritten(tmp_path: Path) -> None:
    _populate(tmp_path, _ready(days=1))
    document = json.loads(_manifest(tmp_path).read_text())
    document["input_manifest_version"] = 1
    document.pop("liquidity_inputs_sha256", None)
    document.pop("hour_grid_sha256", None)
    document.pop("book_inputs_sha256", None)
    body = json.dumps(document, sort_keys=True).encode()
    _manifest(tmp_path).write_bytes(body)
    replayed = replay_session(tmp_path, SESSION)
    assert replayed.manifest is not None
    assert replayed.manifest.input_manifest_version == 1
    assert replayed.manifest.liquidity_inputs_sha256 is None
    assert replayed.manifest.hour_grid_sha256 is None
    assert replayed.manifest.book_inputs_sha256 is None
    assert _manifest(tmp_path).read_bytes() == body
    again = _populate(tmp_path, _ready(days=1))
    assert again.manifest == replayed.manifest
    assert _manifest(tmp_path).read_bytes() == body
    added = tmp_path / "captures" / "session=2026-10-05" / "liquidity" / "symbol=SOLUSDT.json"
    added.parent.mkdir(parents=True)
    added.write_text("{}")
    assert replay_session(tmp_path, SESSION).readiness.ready is True
    assert _manifest(tmp_path).read_bytes() == body


def test_a_manifest_version_must_match_its_hash_coverage(tmp_path: Path) -> None:
    _populate(tmp_path, _ready(days=1))
    document = json.loads(_manifest(tmp_path).read_text())
    document["input_manifest_version"] = 1
    _manifest(tmp_path).write_text(json.dumps(document, sort_keys=True))
    with pytest.raises(EvaluationError, match="session input is unusable"):
        replay_session(tmp_path, SESSION)
    document["input_manifest_version"] = 2
    document.pop("hour_grid_sha256")
    _manifest(tmp_path).write_text(json.dumps(document, sort_keys=True))
    with pytest.raises(EvaluationError, match="session input is unusable"):
        replay_session(tmp_path, SESSION)


def test_the_close_instant_is_accepted_and_a_later_instant_is_rejected(tmp_path: Path) -> None:
    assert _populate(tmp_path, _ready()).readiness.ready is True
    later = _ready()
    shifted = UniverseCapture(later.universe.symbols, LATER, later.universe.clock)  # type: ignore[union-attr]
    with pytest.raises(EvaluationError, match="capture is after the close"):
        _populate(
            tmp_path,
            SessionCaptures(
                shifted,
                later.packets,
                later.candidate_absences,
                later.bars,
                later.bar_absences,
                later.regime,
                later.regime_failures,
            ),
        )
    stored = (tmp_path / "sessions" / "date=2026-10-05" / "universe.json").read_bytes()
    captured = SessionCaptures(
        UniverseCapture((SYMBOL,), CLOSE, _clock(captured_at=LATER)),
        later.packets,
        (),
        later.bars,
        (),
        later.regime,
        (),
    )
    with pytest.raises(EvaluationError, match="capture is after the close"):
        _populate(tmp_path, captured)
    assert (tmp_path / "sessions" / "date=2026-10-05" / "universe.json").read_bytes() == stored


def test_a_market_or_fundamental_after_the_close_is_rejected(tmp_path: Path) -> None:
    late_market = _ready()
    packet = _packet(as_of=LATER)
    with pytest.raises(EvaluationError, match="capture is after the close"):
        _populate(tmp_path, _swap_packets(late_market, (packet,)))
    assert list(tmp_path.rglob("*.json")) == []
    late_stamp = _swap_packets(_ready(), (_packet(gecko_stamp=LATER),))
    with pytest.raises(EvaluationError, match="capture is after the close"):
        _populate(tmp_path, late_stamp)
    assert list(tmp_path.rglob("*.json")) == []


def test_a_bar_after_the_session_is_rejected(tmp_path: Path) -> None:
    ready = _ready(days=1)
    bars = (
        BarCapture(
            "BTCUSDT",
            (*_history("BTCUSDT", days=1), _bar("BTCUSDT", SESSION + timedelta(days=1))),
            _clock(source="binance/klines"),
        ),
        *ready.bars[1:],
    )
    with pytest.raises(EvaluationError, match="capture is after the close"):
        _populate(tmp_path, _swap_bars(ready, bars))
    assert list(tmp_path.rglob("*.json")) == []


def test_a_historical_session_bar_retrieved_after_the_close_is_rejected(tmp_path: Path) -> None:
    ready = _ready(days=1)
    late = _clock(source="binance/klines", captured_at=LATER)
    bars = (
        BarCapture("BTCUSDT", ready.bars[0].bars, late),
        *ready.bars[1:],
    )
    with pytest.raises(EvaluationError, match="capture is after the close"):
        _populate(tmp_path, _swap_bars(ready, bars))
    assert list(tmp_path.rglob("*.json")) == []


def test_a_later_session_accepts_a_bar_retrieved_after_its_close(tmp_path: Path) -> None:
    session = date(2026, 10, 6)
    close = datetime(2026, 10, 7, tzinfo=UTC)
    later = close + timedelta(seconds=1)
    observed = datetime(2026, 10, 6, 12, tzinfo=UTC)
    clock = CaptureClock("binance/klines", close, later)
    packet = _packet(
        as_of=close,
        gecko_stamp=close,
        cmc_stamp=close,
        clock=CaptureClock("coingecko", close, close),
    )
    captures = SessionCaptures(
        universe=UniverseCapture(
            (SYMBOL,), close, CaptureClock("binance/exchangeInfo", close, close)
        ),
        packets=(packet,),
        candidate_absences=(),
        bars=(
            BarCapture("BTCUSDT", _history("BTCUSDT", days=1, end=session), clock),
            BarCapture(SYMBOL, _history(SYMBOL, days=1, end=session), clock),
        ),
        bar_absences=(),
        regime=(
            RegimeCapture(_session_observation("btc_dominance", observed, close), observed),
            RegimeCapture(_session_observation("stablecoin_supply", observed, None), observed),
        ),
        regime_failures=(),
    )
    result = populate_session(
        tmp_path,
        session,
        later,
        captures,
        git_sha=SHA,
        weights_present=False,
    )
    assert result.readiness.ready is True
    assert result.manifest is not None
    btc = next(entry for entry in result.manifest.provenance if entry.name == "bars:BTCUSDT")
    assert btc.captured_at == later
    assert btc.source_timestamp == close


def test_a_later_session_bar_retrieved_before_the_close_is_rejected(tmp_path: Path) -> None:
    session = date(2026, 10, 6)
    close = datetime(2026, 10, 7, tzinfo=UTC)
    early = CaptureClock("binance/klines", close, close - timedelta(seconds=1))
    bars = (BarCapture("BTCUSDT", _history("BTCUSDT", days=1, end=session), early),)
    captures = _swap_bars(_ready(days=1), bars)
    with pytest.raises(EvaluationError, match="session bar was retrieved before it closed"):
        populate_session(tmp_path, session, close, captures, git_sha=SHA, weights_present=False)
    assert list(tmp_path.rglob("*.json")) == []


def test_history_retrieved_after_the_close_is_rejected(tmp_path: Path) -> None:
    ready = _ready(days=1)
    prior = _bar("BTCUSDT", SESSION - timedelta(days=1))
    bars = (BarCapture("BTCUSDT", (prior,), _clock(source="binance/klines", captured_at=LATER)),)
    with pytest.raises(EvaluationError, match="capture is after the close"):
        _populate(tmp_path, _swap_bars(ready, bars))
    assert list(tmp_path.rglob("*.json")) == []


def test_a_bar_retrieval_needs_a_source(tmp_path: Path) -> None:
    ready = _ready(days=1)
    bars = (BarCapture("BTCUSDT", ready.bars[0].bars, _clock(source="")), *ready.bars[1:])
    with pytest.raises(EvaluationError, match="source is required"):
        _populate(tmp_path, _swap_bars(ready, bars))
    assert list(tmp_path.rglob("*.json")) == []


def test_a_regime_source_after_the_close_is_rejected(tmp_path: Path) -> None:
    ready = _ready()
    regime = (
        RegimeCapture(_observation("btc_dominance", LATER), CLOSE),
        ready.regime[1],
    )
    with pytest.raises(EvaluationError, match="capture is after the close"):
        _populate(tmp_path, _swap_regime(ready, regime))
    assert list(tmp_path.rglob("*.json")) == []


def test_a_bar_symbol_mismatch_is_a_blocker_and_writes_nothing(tmp_path: Path) -> None:
    ready = _ready(days=1)
    bars = (
        BarCapture("BTCUSDT", (_bar(OTHER, SESSION),), _clock(source="binance/klines")),
        *ready.bars[1:],
    )
    result = _populate(tmp_path, _swap_bars(ready, bars))
    assert result.readiness.ready is False
    assert result.readiness.blocks == ("bar_symbol_mismatch:BTCUSDT",)
    assert result.manifest is None
    assert list(tmp_path.rglob("*.json")) == []


def test_a_candidate_symbol_mismatch_is_a_blocker_and_writes_nothing(tmp_path: Path) -> None:
    ready = _ready(symbols=(OTHER,), days=1)
    packet = _packet(SYMBOL)
    mismatched = PacketCapture(OTHER, packet.candidate, packet.clock)
    result = _populate(tmp_path, _swap_packets(ready, (mismatched,)))
    assert result.readiness.blocks == (f"candidate_symbol_mismatch:{OTHER}",)
    assert result.manifest is None
    assert list(tmp_path.rglob("*.json")) == []


def test_a_duplicate_bar_date_is_a_blocker_and_writes_nothing(tmp_path: Path) -> None:
    ready = _ready(days=1)
    duplicate = _bar("BTCUSDT", SESSION)
    bars = (
        BarCapture("BTCUSDT", (duplicate, duplicate), _clock(source="binance/klines")),
        *ready.bars[1:],
    )
    result = _populate(tmp_path, _swap_bars(ready, bars))
    assert result.readiness.blocks == ("duplicate_bar_date:BTCUSDT",)
    assert result.manifest is None
    assert list(tmp_path.rglob("*.json")) == []


def test_a_missing_btc_session_bar_blocks_without_finalizing(tmp_path: Path) -> None:
    ready = _ready(days=1)
    prior = SESSION - timedelta(days=1)
    bars = (
        BarCapture("BTCUSDT", (_bar("BTCUSDT", prior),), _clock(source="binance/klines")),
        *ready.bars[1:],
    )
    result = _populate(tmp_path, _swap_bars(ready, bars))
    assert result.readiness.ready is False
    assert result.readiness.blocks == ("btc_session_bar_absent",)
    assert result.manifest is None
    document = json.loads(
        (tmp_path / "sessions" / "date=2026-10-05" / "bars" / "symbol=BTCUSDT.json").read_text()
    )
    assert [row["open_date"] for row in document["bars"]] == ["2026-10-04"]


def test_an_explicit_absence_stays_a_note_on_a_ready_session(tmp_path: Path) -> None:
    ready = _ready(symbols=(SYMBOL, OTHER), days=1)
    packets = tuple(packet for packet in ready.packets if packet.symbol != OTHER)
    bars = tuple(series for series in ready.bars if series.symbol != OTHER)
    captures = SessionCaptures(
        UniverseCapture((SYMBOL, OTHER), CLOSE, ready.universe.clock),  # type: ignore[union-attr]
        packets,
        (AbsenceCapture(OTHER, LATER, "coingecko"),),
        bars,
        (AbsenceCapture(OTHER, LATER, "binance/klines"),),
        ready.regime,
        (),
    )
    result = _populate(tmp_path, captures)
    assert result.readiness.ready is True
    assert result.readiness.blocks == ()
    assert result.readiness.decision_notes == (
        f"daily_bar_absent:{OTHER}",
        f"missing_candidate:{OTHER}",
        "score_weights_not_frozen",
    )
    assert result.manifest is not None
    again = _populate(tmp_path, captures, as_of=CLOSE + timedelta(hours=1), git_sha="b" * 40)
    assert again.manifest == result.manifest


def test_partial_symbol_coverage_returns_the_missing_producers(tmp_path: Path) -> None:
    ready = _ready(symbols=(SYMBOL, OTHER), days=1)
    packets = tuple(packet for packet in ready.packets if packet.symbol != OTHER)
    bars = tuple(series for series in ready.bars if series.symbol != OTHER)
    captures = SessionCaptures(
        UniverseCapture((SYMBOL, OTHER), CLOSE, ready.universe.clock),  # type: ignore[union-attr]
        packets,
        (),
        bars,
        (),
        ready.regime,
        (),
    )
    result = _populate(tmp_path, captures)
    assert result.readiness.ready is False
    assert result.readiness.blocks == (
        f"bars_not_produced:{OTHER}",
        f"candidate_not_produced:{OTHER}",
    )
    assert result.manifest is None
    assert (tmp_path / "sessions" / "date=2026-10-05" / "universe.json").is_file()
    completed = SessionCaptures(
        captures.universe,
        packets,
        (AbsenceCapture(OTHER, CLOSE, "coingecko"),),
        bars,
        (AbsenceCapture(OTHER, CLOSE, "binance/klines"),),
        captures.regime,
        (),
    )
    sealed = _populate(tmp_path, completed)
    assert sealed.readiness.ready is True
    assert sealed.manifest is not None


def test_an_identical_rerun_keeps_the_frozen_manifest(tmp_path: Path) -> None:
    ready = _ready(days=2)
    first = _populate(tmp_path, ready)
    body = _manifest(tmp_path).read_bytes()
    reversed_bars = tuple(
        BarCapture(series.symbol, tuple(reversed(series.bars)), series.clock)
        for series in ready.bars
    )
    second = _populate(
        tmp_path,
        _swap_bars(ready, reversed_bars),
        as_of=CLOSE + timedelta(hours=1),
        git_sha="b" * 40,
    )
    assert second.manifest == first.manifest
    assert second.manifest.finalized_at == CLOSE
    assert second.manifest.git_sha == SHA
    assert _manifest(tmp_path).read_bytes() == body
    assert second.readiness.ready is True


def test_conflicting_evidence_does_not_replace_a_finalized_session(tmp_path: Path) -> None:
    ready = _ready(days=1)
    _populate(tmp_path, ready)
    body = _manifest(tmp_path).read_bytes()
    candidate = (
        tmp_path / "sessions" / "date=2026-10-05" / "candidates" / f"symbol={SYMBOL}.json"
    ).read_bytes()
    changed = _swap_packets(ready, (_packet(gecko_cap=Decimal("2")),))
    with pytest.raises(EvaluationError, match="already exists with a different payload"):
        _populate(tmp_path, changed)
    assert _manifest(tmp_path).read_bytes() == body
    assert (
        tmp_path / "sessions" / "date=2026-10-05" / "candidates" / f"symbol={SYMBOL}.json"
    ).read_bytes() == candidate


def test_a_failed_finalize_can_be_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = os.link

    def fail_manifest(source: str, destination: str) -> None:
        if str(destination).endswith("manifest.json"):
            raise OSError("disk")
        real(source, destination)

    monkeypatch.setattr(os, "link", fail_manifest)
    with pytest.raises(OSError, match="disk"):
        _populate(tmp_path, _ready(days=1))
    assert (tmp_path / "sessions" / "date=2026-10-05" / "universe.json").is_file()
    assert not _manifest(tmp_path).exists()
    monkeypatch.undo()
    result = _populate(tmp_path, _ready(days=1))
    assert result.readiness.ready is True
    assert result.manifest is not None
    assert list(tmp_path.rglob(".*.tmp")) == []


def test_a_writer_temp_file_does_not_block_population(tmp_path: Path) -> None:
    directory = tmp_path / "sessions" / "date=2026-10-05" / "candidates"
    directory.mkdir(parents=True)
    (directory / f".symbol={SYMBOL}.json.1.tmp").write_text("partial")
    result = _populate(tmp_path, _ready(days=1))
    assert result.readiness.ready is True


def test_replay_refuses_a_changed_frozen_input(tmp_path: Path) -> None:
    _populate(tmp_path, _ready(days=1))
    path = tmp_path / "sessions" / "date=2026-10-05" / "universe.json"
    original = path.read_bytes()
    path.write_bytes(original.replace(b"binance/exchangeInfo", b"binance/rewritten!!!!"))
    with pytest.raises(EvaluationError, match="finalized session does not match its inputs"):
        replay_session(tmp_path, SESSION)
    assert _manifest(tmp_path).read_bytes()
    path.write_bytes(original)


def test_an_open_session_is_not_populated(tmp_path: Path) -> None:
    result = _populate(tmp_path, _ready(days=1), as_of=CLOSE - timedelta(seconds=1))
    assert result.readiness.blocks == ("session_not_closed",)
    assert result.manifest is None
    assert list(tmp_path.rglob("*.json")) == []


def test_an_undated_fundamental_blocks_without_a_final_manifest(tmp_path: Path) -> None:
    captures = _swap_packets(_ready(days=1), (_packet(gecko_stamp=None),))
    result = _populate(tmp_path, captures)
    assert result.readiness.blocks == (f"undated_fundamental:{SYMBOL}",)
    assert result.manifest is None
    assert list(tmp_path.rglob("*.json")) == []


def test_a_regime_failure_blocks_the_session(tmp_path: Path) -> None:
    ready = _ready(days=1)
    failure = CollectionFailure(
        series="btc_dominance",
        provider="coingecko",
        observed_at=datetime(SESSION.year, SESSION.month, SESSION.day, tzinfo=UTC),
        symbol=None,
        error="host returned 451",
    )
    captures = SessionCaptures(
        ready.universe,
        ready.packets,
        (),
        ready.bars,
        (),
        (ready.regime[1],),
        (FailureCapture(failure, CLOSE),),
    )
    result = _populate(tmp_path, captures)
    assert "regime_failed:btc_dominance" in result.readiness.blocks
    assert "regime_not_produced:btc_dominance" not in result.readiness.blocks
    assert result.manifest is None


def test_funding_after_the_close_is_not_a_regime_input(tmp_path: Path) -> None:
    ready = _ready(days=1)
    funding = RegimeCapture(_observation("funding", LATER), LATER)
    captures = SessionCaptures(
        ready.universe,
        ready.packets,
        (),
        ready.bars,
        (),
        (*ready.regime, funding),
        (),
    )
    result = _populate(tmp_path, captures)
    assert result.readiness.ready is True
    regime = tmp_path / "sessions" / "date=2026-10-05" / "regime"
    assert sorted(path.name for path in regime.iterdir()) == [
        "btc_dominance.json",
        "stablecoin_supply.json",
    ]


def test_present_weights_are_recorded_on_the_freeze(tmp_path: Path) -> None:
    result = _populate(tmp_path, _ready(days=1), weights_present=True)
    assert result.readiness.ready is True
    assert result.readiness.decision_notes == ()
    assert result.readiness.score_weights == "present"
    assert result.manifest is not None
    assert result.manifest.score_weights == "present"
    with pytest.raises(EvaluationError, match="different payload"):
        _populate(tmp_path, _ready(days=1), weights_present=False)


def test_a_naive_clock_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="as_of must be timezone-aware UTC"):
        _populate(tmp_path, _ready(days=1), as_of=datetime(2026, 10, 6))  # noqa: DTZ001
    shifted = datetime(2026, 10, 6, tzinfo=timezone(timedelta(hours=1)))
    with pytest.raises(EvaluationError, match="as_of must be timezone-aware UTC"):
        _populate(tmp_path, _ready(days=1), as_of=shifted)
    assert list(tmp_path.rglob("*.json")) == []


def test_a_repeated_snapshot_symbol_blocks_without_writing(tmp_path: Path) -> None:
    ready = _ready(days=1)
    universe = UniverseCapture((SYMBOL, SYMBOL), CLOSE, ready.universe.clock)  # type: ignore[union-attr]
    captures = SessionCaptures(universe, ready.packets, (), ready.bars, (), ready.regime, ())
    result = _populate(tmp_path, captures)
    assert result.readiness.blocks == ("snapshot_repeats_symbol",)
    assert result.manifest is None
    assert list(tmp_path.rglob("*.json")) == []


def test_conflicting_candidate_evidence_is_rejected(tmp_path: Path) -> None:
    ready = _ready(days=1)
    with pytest.raises(EvaluationError, match="conflicting candidate evidence"):
        _populate(tmp_path, _swap_packets(ready, ready.packets + ready.packets))
    assert list(tmp_path.rglob("*.json")) == []


def test_conflicting_bar_evidence_is_rejected(tmp_path: Path) -> None:
    ready = _ready(days=1)
    with pytest.raises(EvaluationError, match="conflicting bar evidence"):
        _populate(tmp_path, _swap_bars(ready, ready.bars + ready.bars))
    assert list(tmp_path.rglob("*.json")) == []


def test_conflicting_regime_evidence_is_rejected(tmp_path: Path) -> None:
    ready = _ready(days=1)
    with pytest.raises(EvaluationError, match="conflicting regime evidence"):
        _populate(tmp_path, _swap_regime(ready, (*ready.regime, ready.regime[0])))
    assert list(tmp_path.rglob("*.json")) == []


def test_a_packet_and_an_absence_cannot_describe_the_same_symbol(tmp_path: Path) -> None:
    ready = _ready(days=1)
    captures = SessionCaptures(
        ready.universe,
        ready.packets,
        (AbsenceCapture(SYMBOL, CLOSE, "coingecko"),),
        ready.bars,
        (),
        ready.regime,
        (),
    )
    with pytest.raises(EvaluationError, match="candidate absence conflicts with a packet"):
        _populate(tmp_path, captures)
    assert list(tmp_path.rglob("*.json")) == []


def test_a_blank_source_is_rejected(tmp_path: Path) -> None:
    ready = _ready(days=1)
    universe = UniverseCapture((SYMBOL,), CLOSE, CaptureClock("", CLOSE, CLOSE))
    captures = SessionCaptures(universe, ready.packets, (), ready.bars, (), ready.regime, ())
    with pytest.raises(EvaluationError, match="source is required"):
        _populate(tmp_path, captures)
    assert list(tmp_path.rglob("*.json")) == []


def test_a_missing_git_sha_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="git sha is missing"):
        _populate(tmp_path, _ready(days=1), git_sha="")
    assert list(tmp_path.rglob("*.json")) == []


def test_replay_of_an_unfinalized_session_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="session is not finalized"):
        replay_session(tmp_path, SESSION)


def test_a_missing_universe_is_not_finalized(tmp_path: Path) -> None:
    ready = _ready(days=1)
    captures = SessionCaptures(None, ready.packets, (), ready.bars, (), ready.regime, ())
    result = _populate(tmp_path, captures)
    assert result.readiness.blocks == ("snapshot_not_produced",)
    assert result.manifest is None
    assert not (tmp_path / "sessions" / "date=2026-10-05" / "universe.json").exists()


def test_a_bar_absence_cannot_override_stored_bars(tmp_path: Path) -> None:
    ready = _ready(days=1)
    captures = SessionCaptures(
        ready.universe,
        ready.packets,
        (),
        ready.bars,
        (AbsenceCapture("BTCUSDT", CLOSE, "binance/klines"),),
        ready.regime,
        (),
    )
    with pytest.raises(EvaluationError, match="bar absence conflicts with stored bars"):
        _populate(tmp_path, captures)
    assert list(tmp_path.rglob("*.json")) == []


def test_a_repeated_bar_absence_is_rejected(tmp_path: Path) -> None:
    ready = _ready(days=1)
    absence = AbsenceCapture(OTHER, CLOSE, "binance/klines")
    captures = SessionCaptures(
        ready.universe,
        ready.packets,
        (),
        ready.bars,
        (absence, absence),
        ready.regime,
        (),
    )
    with pytest.raises(EvaluationError, match="conflicting absence evidence"):
        _populate(tmp_path, captures)


def test_a_blank_absence_source_is_rejected(tmp_path: Path) -> None:
    ready = _ready(days=1)
    captures = SessionCaptures(
        ready.universe,
        ready.packets,
        (AbsenceCapture(OTHER, CLOSE, ""),),
        ready.bars,
        (),
        ready.regime,
        (),
    )
    with pytest.raises(EvaluationError, match="source is required"):
        _populate(tmp_path, captures)


def test_a_funding_failure_is_not_a_regime_input(tmp_path: Path) -> None:
    ready = _ready(days=1)
    failure = CollectionFailure(
        series="funding",
        provider="binance",
        observed_at=CLOSE,
        symbol="BTCUSDT",
        error="host returned 451",
    )
    captures = SessionCaptures(
        ready.universe,
        ready.packets,
        (),
        ready.bars,
        (),
        ready.regime,
        (FailureCapture(failure, CLOSE),),
    )
    assert _populate(tmp_path, captures).readiness.ready is True


def test_replay_refuses_a_missing_universe_file(tmp_path: Path) -> None:
    _populate(tmp_path, _ready(days=1))
    (tmp_path / "sessions" / "date=2026-10-05" / "universe.json").unlink()
    with pytest.raises(EvaluationError, match="does not match its inputs"):
        replay_session(tmp_path, SESSION)


def test_replay_refuses_an_unreadable_manifest(tmp_path: Path) -> None:
    _populate(tmp_path, _ready(days=1))
    _manifest(tmp_path).write_text("{}")
    with pytest.raises(EvaluationError, match="session input is unusable"):
        replay_session(tmp_path, SESSION)
    _manifest(tmp_path).write_text("{")
    with pytest.raises(EvaluationError, match="session input is unusable"):
        replay_session(tmp_path, SESSION)
    _manifest(tmp_path).write_text("[]")
    with pytest.raises(EvaluationError, match="session input is unusable"):
        replay_session(tmp_path, SESSION)


def test_a_damaged_bar_file_is_unusable(tmp_path: Path) -> None:
    directory = tmp_path / "sessions" / "date=2026-10-05" / "bars"
    directory.mkdir(parents=True)
    (directory / "symbol=ADAUSDT.json").write_text('{"symbol": 1, "bars": []}')
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))
    (directory / "symbol=ADAUSDT.json").write_text('{"symbol": "ADAUSDT", "bars": 1}')
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))
    (directory / "symbol=ADAUSDT.json").write_text('{"symbol": "ADAUSDT", "bars": [1]}')
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))
    (directory / "symbol=ADAUSDT.json").write_text('{"symbol": "ADAUSDT", "bars": [{}]}')
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))
    (directory / "symbol=ADAUSDT.json").write_text(
        '{"symbol": "ADAUSDT", "bars": [{"symbol": "ADAUSDT", "open_date": "2026-10-05", '
        '"open": true, "high": "1", "low": "1", "close": "1", "volume": "1", '
        '"quote_volume": "1", "trade_count": 1, "taker_buy_base_volume": "1", '
        '"taker_buy_quote_volume": "1"}]}'
    )
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))
    (directory / "symbol=ADAUSDT.json").write_text(
        '{"symbol": "ADAUSDT", "bars": [{"symbol": "ADAUSDT", "open_date": "2026-10-05", '
        '"open": 1, "high": "1", "low": "1", "close": "1", "volume": "1", '
        '"quote_volume": "1", "trade_count": 1, "taker_buy_base_volume": "1", '
        '"taker_buy_quote_volume": "1"}]}'
    )
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))
    (directory / "symbol=ADAUSDT.json").write_text(
        '{"symbol": "ADAUSDT", "bars": [{"symbol": "ADAUSDT", "open_date": "2026-10-05", '
        '"open": "nope", "high": "1", "low": "1", "close": "1", "volume": "1", '
        '"quote_volume": "1", "trade_count": 1, "taker_buy_base_volume": "1", '
        '"taker_buy_quote_volume": "1"}]}'
    )
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))


def test_a_damaged_regime_file_is_unusable(tmp_path: Path) -> None:
    directory = tmp_path / "sessions" / "date=2026-10-05" / "regime"
    directory.mkdir(parents=True)
    failure = directory / "btc_dominance.failure.json"
    failure.write_text('{"series": 1}')
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))
    failure.unlink()
    (directory / "note.json").write_text("{}")
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))
    (directory / "note.json").write_text('{"values": [], "units": {}}')
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))
    (directory / "note.json").write_text(
        '{"values": {}, "units": {}, "series": "btc_dominance", "provider": "coingecko", '
        '"source_timestamp": 1, "observed_at": "2026-10-05T12:00:00Z", "symbol": null}'
    )
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))
    (directory / "note.json").write_text(
        '{"values": {}, "units": {}, "series": "btc_dominance", "provider": "coingecko", '
        '"source_timestamp": "2026-10-05T12:00:00", "observed_at": "2026-10-05T12:00:00Z", '
        '"symbol": null}'
    )
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))
    (directory / "note.json").write_text(
        '{"values": {}, "units": {}, "series": "btc_dominance", "provider": "coingecko", '
        '"source_timestamp": "yesterday", "observed_at": "2026-10-05T12:00:00Z", "symbol": null}'
    )
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))
    (directory / "note.json").write_text(
        '{"values": {}, "units": {}, "series": "btc_dominance", "provider": "coingecko", '
        '"source_timestamp": "2026-10-06T00:00:00+01:00", '
        '"observed_at": "2026-10-05T12:00:00Z", "symbol": null}'
    )
    with pytest.raises(EvaluationError, match="session input is unusable"):
        _populate(tmp_path, _ready(days=1))


def test_a_lost_bar_create_keeps_the_first_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = _ready(days=1)
    real = os.link

    def lose_the_race(source: str, destination: str) -> None:
        if str(destination).endswith("symbol=BTCUSDT.json"):
            real(source, destination)
            raise FileExistsError
        real(source, destination)

    monkeypatch.setattr(os, "link", lose_the_race)
    result = _populate(tmp_path, ready)
    assert result.readiness.ready is True


def test_a_different_bar_payload_is_not_replaced(tmp_path: Path) -> None:
    directory = tmp_path / "sessions" / "date=2026-10-05" / "bars"
    directory.mkdir(parents=True)
    (directory / "symbol=BTCUSDT.json").write_bytes(b"{}")
    with pytest.raises(EvaluationError, match="already exists with a different payload"):
        _populate(tmp_path, _ready(days=1))


def test_a_finalized_session_rejects_a_new_file_before_writing_it(tmp_path: Path) -> None:
    ready = _ready(days=1)
    _populate(tmp_path, ready)
    stored = _stored(tmp_path)
    extra = BarCapture("ADAUSDT", (_bar("ADAUSDT", SESSION),), _clock(source="binance/klines"))
    with pytest.raises(EvaluationError, match="different payload"):
        _populate(tmp_path, _swap_bars(ready, (*ready.bars, extra)))
    assert _stored(tmp_path) == stored
    assert replay_session(tmp_path, SESSION).readiness.ready is True


def test_a_finalized_session_rejects_a_regime_failure_swap(tmp_path: Path) -> None:
    ready = _ready(days=1)
    _populate(tmp_path, ready)
    stored = _stored(tmp_path)
    failure = CollectionFailure(
        series="btc_dominance",
        provider="coingecko",
        observed_at=datetime(SESSION.year, SESSION.month, SESSION.day, 12, tzinfo=UTC),
        symbol=None,
        error="host returned 451",
    )
    captures = SessionCaptures(
        ready.universe,
        ready.packets,
        (),
        ready.bars,
        (),
        (ready.regime[1],),
        (FailureCapture(failure, CLOSE),),
    )
    with pytest.raises(EvaluationError, match="different payload"):
        _populate(tmp_path, captures)
    assert _stored(tmp_path) == stored
    assert replay_session(tmp_path, SESSION).readiness.ready is True


def test_a_finalized_session_rejects_a_removed_universe(tmp_path: Path) -> None:
    ready = _ready(days=1)
    _populate(tmp_path, ready)
    stored = _stored(tmp_path)
    captures = SessionCaptures(None, ready.packets, (), ready.bars, (), ready.regime, ())
    with pytest.raises(EvaluationError, match="different payload"):
        _populate(tmp_path, captures)
    assert _stored(tmp_path) == stored


def test_an_earlier_regime_observation_does_not_occupy_the_series(tmp_path: Path) -> None:
    ready = _ready(days=1)
    early = replace(
        ready.regime[0].observation,
        observed_at=datetime(2026, 10, 4, 12, tzinfo=UTC),
    )
    captures = _swap_regime(ready, (RegimeCapture(early, CLOSE), ready.regime[1]))
    result = _populate(tmp_path, captures)
    assert "regime_not_produced:btc_dominance" in result.readiness.blocks
    regime = tmp_path / "sessions" / "date=2026-10-05" / "regime"
    assert not (regime / "btc_dominance.json").exists()
    sealed = _populate(tmp_path, ready)
    assert sealed.readiness.ready is True
    assert (regime / "btc_dominance.json").is_file()


def test_an_earlier_regime_failure_does_not_occupy_the_series(tmp_path: Path) -> None:
    ready = _ready(days=1)
    failure = CollectionFailure(
        series="btc_dominance",
        provider="coingecko",
        observed_at=datetime(2026, 10, 4, 12, tzinfo=UTC),
        symbol=None,
        error="host returned 451",
    )
    captures = SessionCaptures(
        ready.universe,
        ready.packets,
        (),
        ready.bars,
        (),
        (ready.regime[1],),
        (FailureCapture(failure, CLOSE),),
    )
    result = _populate(tmp_path, captures)
    assert "regime_not_produced:btc_dominance" in result.readiness.blocks
    regime = tmp_path / "sessions" / "date=2026-10-05" / "regime"
    assert not (regime / "btc_dominance.failure.json").exists()
    sealed = _populate(tmp_path, ready)
    assert sealed.readiness.ready is True


def test_a_regime_observation_at_the_next_midnight_writes_nothing(tmp_path: Path) -> None:
    ready = _ready(days=1)
    late = replace(ready.regime[0].observation, observed_at=CLOSE)
    captures = _swap_regime(ready, (RegimeCapture(late, CLOSE), ready.regime[1]))
    result = _populate(tmp_path, captures)
    assert "regime_lookahead:btc_dominance" in result.readiness.blocks
    assert result.manifest is None
    assert list(tmp_path.rglob("*.json")) == []


def test_a_btc_absence_does_not_satisfy_the_session_bar(tmp_path: Path) -> None:
    ready = _ready(days=1)
    bars = tuple(series for series in ready.bars if series.symbol != "BTCUSDT")
    captures = SessionCaptures(
        ready.universe,
        ready.packets,
        (),
        bars,
        (AbsenceCapture("BTCUSDT", LATER, "binance/klines"),),
        ready.regime,
        (),
    )
    result = _populate(tmp_path, captures)
    assert "btc_bars_not_produced" in result.readiness.blocks
    assert result.manifest is None
    absence = (
        tmp_path
        / "sessions"
        / "date=2026-10-05"
        / "absences"
        / "input=daily_bar"
        / "symbol=BTCUSDT.json"
    )
    assert absence.is_file()


def test_a_repeated_candidate_absence_is_rejected(tmp_path: Path) -> None:
    ready = _ready(days=1)
    absence = AbsenceCapture(OTHER, CLOSE, "coingecko")
    captures = SessionCaptures(
        ready.universe,
        ready.packets,
        (absence, absence),
        ready.bars,
        (),
        ready.regime,
        (),
    )
    with pytest.raises(EvaluationError, match="conflicting absence evidence"):
        _populate(tmp_path, captures)
    assert list(tmp_path.rglob("*.json")) == []


def test_a_bar_symbol_cannot_leave_the_session_directory(tmp_path: Path) -> None:
    ready = _ready(days=1)
    escaped = BarCapture("../OUTSIDE", ready.bars[0].bars, ready.bars[0].clock)
    with pytest.raises(EvaluationError, match="uppercase letters or digits"):
        _populate(tmp_path, _swap_bars(ready, (escaped, *ready.bars[1:])))
    assert list(tmp_path.rglob("*")) == []


def test_a_manifest_that_appears_during_the_write_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = _ready(days=1)
    first = _populate(tmp_path, ready)
    stored = _manifest(tmp_path).read_bytes()
    _manifest(tmp_path).unlink()
    real = population._write

    def restore(root: Path, session: date, plan: object) -> None:
        _manifest(tmp_path).write_bytes(stored)
        monkeypatch.setattr(population, "_write", real)
        real(root, session, plan)

    monkeypatch.setattr(population, "_write", restore)
    second = _populate(tmp_path, ready, as_of=CLOSE + timedelta(hours=1), git_sha="b" * 40)
    assert second.manifest == first.manifest
    assert _manifest(tmp_path).read_bytes() == stored


def test_a_different_manifest_that_appears_during_the_write_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = _ready(days=1)
    _populate(tmp_path, ready)
    document = json.loads(_manifest(tmp_path).read_text())
    document["score_weights"] = "present"
    planted = json.dumps(document, sort_keys=True).encode()
    _manifest(tmp_path).unlink()
    real = population._write

    def restore(root: Path, session: date, plan: object) -> None:
        _manifest(tmp_path).write_bytes(planted)
        monkeypatch.setattr(population, "_write", real)
        real(root, session, plan)

    monkeypatch.setattr(population, "_write", restore)
    with pytest.raises(EvaluationError, match="different payload"):
        _populate(tmp_path, ready)
    assert _manifest(tmp_path).read_bytes() == planted


def test_a_nested_bar_directory_is_not_part_of_the_freeze(tmp_path: Path) -> None:
    result = _populate(tmp_path, _ready(days=1))
    (tmp_path / "sessions" / "date=2026-10-05" / "bars" / "nested").mkdir()
    assert replay_session(tmp_path, SESSION).manifest == result.manifest


def _swap_packets(captures: SessionCaptures, packets: tuple[PacketCapture, ...]) -> SessionCaptures:
    return SessionCaptures(
        captures.universe,
        packets,
        captures.candidate_absences,
        captures.bars,
        captures.bar_absences,
        captures.regime,
        captures.regime_failures,
    )


def _swap_bars(captures: SessionCaptures, bars: tuple[BarCapture, ...]) -> SessionCaptures:
    return SessionCaptures(
        captures.universe,
        captures.packets,
        captures.candidate_absences,
        bars,
        captures.bar_absences,
        captures.regime,
        captures.regime_failures,
    )


def _swap_regime(captures: SessionCaptures, regime: tuple[RegimeCapture, ...]) -> SessionCaptures:
    return SessionCaptures(
        captures.universe,
        captures.packets,
        captures.candidate_absences,
        captures.bars,
        captures.bar_absences,
        regime,
        captures.regime_failures,
    )
