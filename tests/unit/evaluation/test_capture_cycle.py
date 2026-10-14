import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

import cip.evaluation.capture_cycle as cycle_module
from cip.adapters.market import ExchangeInfo
from cip.domain.errors import EvaluationError, PolicyError
from cip.evaluation.capture_cycle import (
    BookShot,
    CycleReport,
    TemporaryFailure,
    _one_bar,
    _policy_path,
    _post_close,
    _present,
    _spike,
    book_is_stored,
    capture_cycle,
    run_capture_cycle,
    store_session_book,
)
from cip.evaluation.classification import SymbolClassification
from cip.evaluation.eligibility import CandidateFacts
from cip.evaluation.fundamentals import CmcReading, CoinGeckoReading, UnlockReading
from cip.evaluation.liquidity import MarketSnapshot
from cip.evaluation.liquidity_capture import HourQuote
from cip.evaluation.populate import CaptureClock, PacketCapture, RegimeCapture
from cip.evaluation.prospect import store_packet, universe_capture
from cip.evaluation.scan import ScanCandidate
from cip.history.bars import DailyBar
from cip.recorders.observation import Observation

OPEN = date(2026, 10, 7)
CLOSED = date(2026, 10, 8)
NOW = datetime(2026, 10, 7, 19, tzinfo=UTC)
AFTER = datetime(2026, 10, 8, 1, tzinfo=UTC)
_HOUR_MS = 3_600_000
Answers = dict[tuple[date, str, str | None], object]
Failure = frozenset[tuple[date, str, str | None]] | set[tuple[date, str, str | None]]


class Source:
    def __init__(
        self,
        answers: Answers,
        fail: Failure = frozenset(),
    ) -> None:
        self.answers = answers
        self.fail = fail
        self.calls: list[tuple[str, str | None, date]] = []

    def fetch(self, session: date, kind: str, symbol: str | None) -> object:
        self.calls.append((kind, symbol, session))
        if (session, kind, symbol) in self.fail:
            raise TemporaryFailure("provider")
        return self.answers[(session, kind, symbol)]


def _info() -> ExchangeInfo:
    return ExchangeInfo.model_validate(
        {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                }
            ]
        }
    )


def _classification(captured: datetime) -> tuple[SymbolClassification, datetime]:
    item = SymbolClassification(
        symbol="BTCUSDT",
        base_asset="BTC",
        eur_stable=None,
        fan_token=False,
        monitoring_tag=False,
        delisting=False,
        deposits_suspended=None,
        withdrawals_suspended=None,
        pending_migration=None,
        coin_id="bitcoin",
        mapping="unambiguous",
        market_cap_usd=Decimal("1"),
        market_cap_source_timestamp=captured,
        cmc_id=None,
        cmc_market_cap_usd=None,
        cmc_source_timestamp=None,
    )
    return item, captured


def _regime(series: str, captured: datetime) -> RegimeCapture:
    name = "btc_dominance" if series == "btc_dominance" else "stablecoin_supply_usd"
    unit = "percent" if series == "btc_dominance" else "usd"
    return RegimeCapture(
        Observation(
            series=series,
            provider="coingecko",
            source_timestamp=captured,
            observed_at=captured,
            symbol=None,
            values=((name, Decimal("59")),),
            units=((name, unit),),
        ),
        captured,
    )


def _pre_answers(session: date, captured: datetime) -> dict[tuple[date, str, str | None], object]:
    return {
        (session, "universe", None): universe_capture(session, _info(), captured, captured),
        (session, "ticker", None): (["BTCUSDT"], captured, captured),
        (session, "peg", "USDCUSDT"): ((Decimal("1"),), captured, captured),
        (session, "regime", "btc_dominance"): _regime("btc_dominance", captured),
        (session, "regime", "stablecoin_supply"): _regime("stablecoin_supply", captured),
        (session, "classification", "BTCUSDT"): _classification(captured),
        (session, "book", "BTCUSDT"): BookShot(Decimal("5"), captured, captured),
    }


def _bar(day: date, quote: Decimal) -> DailyBar:
    price = Decimal("10")
    return DailyBar(
        symbol="BTCUSDT",
        open_date=day,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1"),
        quote_volume=quote,
        trade_count=1,
        taker_buy_base_volume=Decimal("1"),
        taker_buy_quote_volume=Decimal("1"),
    )


def _days(event: date, last: Decimal) -> tuple[DailyBar, ...]:
    baseline = [Decimal("2400")] * 28 + [Decimal("1200"), Decimal("3600")]
    start = event - timedelta(days=30)
    quotes = [*baseline, last]
    return tuple(_bar(start + timedelta(days=index), quote) for index, quote in enumerate(quotes))


def _hours(event: date, quotes: tuple[Decimal, ...]) -> tuple[HourQuote, ...]:
    midnight = int(datetime(event.year, event.month, event.day, tzinfo=UTC).timestamp()) * 1000
    hours: list[HourQuote] = []
    for index, quote in enumerate(quotes):
        opened = midnight + index * _HOUR_MS
        hours.append(HourQuote(opened, opened + _HOUR_MS - 1, quote))
    return tuple(hours)


def _packet(session: date, captured: datetime) -> PacketCapture:
    candidate = ScanCandidate(
        facts=CandidateFacts(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            status="TRADING",
            eur_stable=None,
            fan_token=None,
            monitoring_tag=None,
            delisting=None,
            deposits_suspended=None,
            withdrawals_suspended=None,
            pending_migration=None,
            market_cap_usd=None,
            market_cap_rank=None,
            circulating_ratio=None,
            fdv_to_market_cap=None,
            history_days=None,
            unlock_schedule_known=None,
        ),
        market=MarketSnapshot(
            as_of=captured,
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
            coingecko_id=None,
            market_cap_usd=None,
            circulating_supply=None,
            total_supply=None,
            fully_diluted_valuation_usd=None,
            source_timestamp=None,
        ),
        cmc=CmcReading(market_cap_usd=None, source_timestamp=None),
        unlocks=UnlockReading(schedule_known=None, pct_circ_14d=None, pct_circ_90d=None),
        tokenomics=None,
    )
    clock = CaptureClock("binance/ticker", captured, captured)
    return PacketCapture(candidate.facts.symbol, candidate, clock)


def test_the_clock_names_the_open_session_and_refuses_a_future_one() -> None:
    cycle = capture_cycle(NOW)
    assert cycle.open_session == OPEN
    assert cycle.pre_close == OPEN
    assert cycle.post_close is None
    with pytest.raises(EvaluationError, match="future session"):
        capture_cycle(NOW, CLOSED)
    with pytest.raises(EvaluationError, match="sealed session stays sealed"):
        capture_cycle(NOW, date(2026, 10, 6))
    with pytest.raises(EvaluationError, match="session is not open"):
        capture_cycle(NOW, date(2026, 10, 5))
    with pytest.raises(EvaluationError, match="session is a date"):
        capture_cycle(NOW, datetime(2026, 10, 7, tzinfo=UTC))
    with pytest.raises(EvaluationError, match="timezone-aware UTC"):
        capture_cycle(datetime(2026, 10, 7, 19))  # noqa: DTZ001


def test_a_closed_clock_keeps_today_pre_close_and_yesterday_post_close() -> None:
    cycle = capture_cycle(AFTER)
    assert cycle.pre_close == CLOSED
    assert cycle.post_close == OPEN
    only = capture_cycle(AFTER, OPEN)
    assert only.pre_close is None
    assert only.post_close == OPEN


def test_today_is_not_stored_as_a_future_session_book(tmp_path: Path) -> None:
    book = BookShot(Decimal("5"), NOW, NOW)
    with pytest.raises(EvaluationError, match="future session"):
        store_session_book(tmp_path, CLOSED, "BTCUSDT", book)
    with pytest.raises(EvaluationError, match="sealed session stays sealed"):
        store_session_book(tmp_path, date(2026, 10, 6), "BTCUSDT", book)
    assert list(tmp_path.rglob("*.json")) == []


def test_pre_close_stores_one_book_and_a_retry_does_not_fetch_another(tmp_path: Path) -> None:
    source = Source(_pre_answers(OPEN, NOW))
    first = run_capture_cycle(tmp_path, NOW, source, ("BTCUSDT",))
    assert first.cycle.post_close is None
    assert ("book", "BTCUSDT") in first.calls
    assert ("daily_bars", "BTCUSDT") not in first.calls
    assert book_is_stored(tmp_path, OPEN, "BTCUSDT")
    stored = json.loads(
        (tmp_path / "captures" / "session=2026-10-07" / "book" / "symbol=BTCUSDT.json").read_text()
    )
    assert stored["spread_snapshots"] == 1
    assert stored["source_timestamp"] == "2026-10-07T19:00:00Z"
    again = Source(_pre_answers(OPEN, NOW))
    second = run_capture_cycle(tmp_path, NOW, again, ("BTCUSDT",))
    assert again.calls == []
    assert second.retries == ()
    assert "book:BTCUSDT" in second.skipped


def test_a_different_second_book_is_a_conflict_and_the_first_remains(tmp_path: Path) -> None:
    original = BookShot(Decimal("5"), NOW, NOW)
    store_session_book(tmp_path, OPEN, "BTCUSDT", original)
    before = (tmp_path / "captures/session=2026-10-07/book/symbol=BTCUSDT.json").read_bytes()
    with pytest.raises(EvaluationError, match="conflicting observation"):
        store_session_book(tmp_path, OPEN, "BTCUSDT", BookShot(Decimal("9"), NOW, NOW))
    store_session_book(tmp_path, OPEN, "BTCUSDT", original)
    stored = tmp_path / "captures/session=2026-10-07/book/symbol=BTCUSDT.json"
    assert stored.read_bytes() == before
    manifest = tmp_path / "sessions/date=2026-10-07/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    with pytest.raises(EvaluationError, match="finalized session is sealed"):
        store_session_book(tmp_path, OPEN, "ETHUSDT", original)


def test_a_universe_file_without_symbols_is_unusable(tmp_path: Path) -> None:
    path = tmp_path / "captures/session=2026-10-07/universe.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run_capture_cycle(tmp_path, NOW, Source(_pre_answers(OPEN, NOW)), ("BTCUSDT",))


def test_a_conflicting_classification_source_still_stores_the_book(tmp_path: Path) -> None:
    class Conflict(Source):
        def fetch(self, session: date, kind: str, symbol: str | None) -> object:
            if kind == "classification":
                self.calls.append((kind, symbol, session))
                raise EvaluationError("conflicting observation")
            return super().fetch(session, kind, symbol)

    source = Conflict(_pre_answers(OPEN, NOW))
    with pytest.raises(EvaluationError, match="conflicting observation"):
        run_capture_cycle(tmp_path, NOW, source, ("BTCUSDT",))
    assert book_is_stored(tmp_path, OPEN, "BTCUSDT")


def test_one_incomplete_catalog_does_not_block_the_book(tmp_path: Path) -> None:
    info = ExchangeInfo.model_validate(
        {
            "symbols": [
                {
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "status": "TRADING",
                    "symbol": "BTCUSDT",
                },
                {
                    "baseAsset": "ETH",
                    "quoteAsset": "USDT",
                    "status": "TRADING",
                    "symbol": "ETHUSDT",
                },
            ]
        }
    )
    answers = _pre_answers(OPEN, NOW)
    answers[(OPEN, "universe", None)] = universe_capture(OPEN, info, NOW, NOW)
    source = Source(answers, fail={(OPEN, "classification", "BTCUSDT")})
    report = run_capture_cycle(tmp_path, NOW, source, ("BTCUSDT",))
    assert source.calls.count(("classification", "BTCUSDT", OPEN)) == 1
    assert ("classification", "ETHUSDT", OPEN) not in source.calls
    assert "classification:BTCUSDT" in report.retries
    assert "classification:ETHUSDT" in report.retries
    assert ("book", "BTCUSDT") in report.calls
    assert not (tmp_path / "captures/session=2026-10-07/classification").exists()


def test_classification_follows_the_stored_universe(tmp_path: Path) -> None:
    info = ExchangeInfo.model_validate(
        {
            "symbols": [
                {
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "status": "TRADING",
                    "symbol": "BTCUSDT",
                },
                {
                    "baseAsset": "ETH",
                    "quoteAsset": "USDT",
                    "status": "TRADING",
                    "symbol": "ETHUSDT",
                },
            ]
        }
    )
    answers = _pre_answers(OPEN, NOW)
    answers[(OPEN, "universe", None)] = universe_capture(OPEN, info, NOW, NOW)
    eth, captured = _classification(NOW)
    answers[(OPEN, "classification", "ETHUSDT")] = (
        SymbolClassification(
            symbol="ETHUSDT",
            base_asset="ETH",
            eur_stable=eth.eur_stable,
            fan_token=eth.fan_token,
            monitoring_tag=eth.monitoring_tag,
            delisting=eth.delisting,
            deposits_suspended=eth.deposits_suspended,
            withdrawals_suspended=eth.withdrawals_suspended,
            pending_migration=eth.pending_migration,
            coin_id=eth.coin_id,
            mapping=eth.mapping,
            market_cap_usd=eth.market_cap_usd,
            market_cap_source_timestamp=captured,
            cmc_id=None,
            cmc_market_cap_usd=None,
            cmc_source_timestamp=None,
        ),
        captured,
    )
    source = Source(answers)
    report = run_capture_cycle(tmp_path, NOW, source, ("BTCUSDT",))
    folder = tmp_path / "captures/session=2026-10-07"
    assert (folder / "classification/symbol=BTCUSDT.json").is_file()
    assert (folder / "classification/symbol=ETHUSDT.json").is_file()
    assert not (folder / "book/symbol=ETHUSDT.json").exists()
    assert ("classification", "ETHUSDT") in report.calls
    again = Source(answers)
    second = run_capture_cycle(tmp_path, NOW, again, ("BTCUSDT",))
    assert ("classification", "ETHUSDT") not in second.calls
    assert "classification:ETHUSDT" in second.skipped


def test_an_existing_liquidity_snapshot_is_not_fetched_again(tmp_path: Path) -> None:
    path = tmp_path / "captures/session=2026-10-07/liquidity/symbol=BTCUSDT.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"spread_snapshots": 1, "spread_bps": ["5"]}))
    source = Source(_pre_answers(OPEN, NOW))
    report = run_capture_cycle(tmp_path, NOW, source, ("BTCUSDT",))
    assert ("book", "BTCUSDT") not in report.calls
    assert "book:BTCUSDT" in report.skipped
    assert not (tmp_path / "captures/session=2026-10-07/book/symbol=BTCUSDT.json").exists()
    with pytest.raises(EvaluationError, match="conflicting observation"):
        store_session_book(tmp_path, OPEN, "BTCUSDT", BookShot(Decimal("5"), NOW, NOW))


def test_a_provider_failure_is_not_stored_as_absence(tmp_path: Path) -> None:
    source = Source(
        {},
        fail={
            (OPEN, "universe", None),
            (OPEN, "ticker", None),
            (OPEN, "peg", "USDCUSDT"),
            (OPEN, "regime", "btc_dominance"),
            (OPEN, "regime", "stablecoin_supply"),
            (OPEN, "classification", "BTCUSDT"),
            (OPEN, "book", "BTCUSDT"),
        },
    )
    report = run_capture_cycle(tmp_path, NOW, source, ("BTCUSDT",))
    assert "universe" in report.retries
    assert "book:BTCUSDT" in report.retries
    captures = tmp_path / "captures"
    assert not captures.exists() or list(captures.rglob("absences/*")) == []
    assert report.readiness is None
    assert report.finalized is False


def test_post_close_keeps_raw_hours_and_omits_a_contradictory_count(tmp_path: Path) -> None:
    answers = _pre_answers(OPEN, NOW)
    answers[(OPEN, "daily_bars", "BTCUSDT")] = _days(OPEN, Decimal("4800"))
    answers[(OPEN, "hour_bars", "BTCUSDT")] = _hours(OPEN, (Decimal("100"),) * 24)
    source = Source(answers)
    report = run_capture_cycle(tmp_path, AFTER, source, ("BTCUSDT",), requested=OPEN)
    assert report.cycle.pre_close is None
    assert ("daily_bars", "BTCUSDT") in report.calls
    assert ("book", "BTCUSDT") not in report.calls
    assert report.spike_counts == (("BTCUSDT", None),)
    document = json.loads(
        (tmp_path / "captures/session=2026-10-07/hours/symbol=BTCUSDT.json").read_text()
    )
    assert document["spike_candle_count"] is None
    assert len(document["hours"]) == 24
    assert (tmp_path / "captures/session=2026-10-07/final/symbol=BTCUSDT.json").is_file()


def test_post_close_stores_a_derived_count_when_the_hours_reconcile(tmp_path: Path) -> None:
    answers: Answers = {
        (OPEN, "daily_bars", "BTCUSDT"): _days(OPEN, Decimal("4800")),
        (OPEN, "hour_bars", "BTCUSDT"): _hours(OPEN, (Decimal("4800"),) + (Decimal("0"),) * 23),
    }
    report = run_capture_cycle(tmp_path, AFTER, Source(answers), ("BTCUSDT",), requested=OPEN)
    assert report.spike_counts == (("BTCUSDT", 1),)
    document = json.loads(
        (tmp_path / "captures/session=2026-10-07/hours/symbol=BTCUSDT.json").read_text()
    )
    assert document["spike_candle_count"] == 1
    retry = Source(answers)
    run_capture_cycle(tmp_path, AFTER, retry, ("BTCUSDT",), requested=OPEN)
    assert retry.calls == []


def test_an_evidence_block_finalizes_without_downstream_bars(tmp_path: Path) -> None:
    early = run_capture_cycle(tmp_path, NOW, Source(_pre_answers(OPEN, NOW)), ("BTCUSDT",))
    assert early.readiness is None
    assert not (tmp_path / "sessions").exists()
    ready = run_capture_cycle(
        tmp_path,
        AFTER,
        Source({}, fail={(OPEN, "daily_bars", "BTCUSDT"), (OPEN, "hour_bars", "BTCUSDT")}),
        ("BTCUSDT",),
        requested=OPEN,
    )
    assert ready.readiness is not None
    assert ready.readiness.ready is True
    assert ready.finalized is True
    notes = ready.readiness.decision_notes
    assert any(note.startswith("evidence_blocked:BTCUSDT:missing_") for note in notes)
    assert not any(note.startswith("policy_rejected:BTCUSDT:") for note in notes)
    universe = json.loads((tmp_path / "sessions/date=2026-10-07/universe.json").read_text())
    assert universe["symbols"] == ["BTCUSDT"]
    assert not (tmp_path / "captures/session=2026-10-07/final/symbol=BTCUSDT.json").exists()
    manifest = tmp_path / "sessions/date=2026-10-07/manifest.json"
    body = manifest.read_bytes()
    held = run_capture_cycle(tmp_path, AFTER, Source({}), ("BTCUSDT",), requested=OPEN)
    assert held.calls == ()
    assert held.finalized is True
    assert manifest.read_bytes() == body
    from cip.evaluation.populate import populate_session
    from cip.evaluation.prospect import load_closed

    again = populate_session(
        tmp_path,
        OPEN,
        AFTER,
        load_closed(tmp_path, OPEN),
        git_sha="0" * 40,
        weights_present=False,
    )
    assert again.manifest is not None
    assert again.readiness.ready is True


def test_a_missing_classification_or_lane_file_is_not_ready(tmp_path: Path) -> None:
    from cip.evaluation.capture_cycle import _symbol_ready
    from cip.evaluation.stage import load_universe

    policy = load_universe()
    assert _symbol_ready(tmp_path, OPEN, "BTCUSDT", policy) is False
    _closed_passer(tmp_path)
    lane = tmp_path / "captures/session=2026-10-07/lane/symbol=BTCUSDT.json"
    lane.unlink()
    assert _symbol_ready(tmp_path, OPEN, "BTCUSDT", policy) is False


def test_a_lane_passer_still_waits_for_downstream_evidence(tmp_path: Path) -> None:
    _closed_passer(tmp_path)
    waiting = run_capture_cycle(
        tmp_path,
        AFTER,
        Source({}, fail={(OPEN, "daily_bars", "BTCUSDT"), (OPEN, "hour_bars", "BTCUSDT")}),
        ("BTCUSDT",),
        requested=OPEN,
    )
    assert waiting.finalized is False
    answers: Answers = {
        (OPEN, "daily_bars", "BTCUSDT"): _days(OPEN, Decimal("4800")),
        (OPEN, "hour_bars", "BTCUSDT"): _hours(OPEN, (Decimal("4800"),) + (Decimal("0"),) * 23),
    }
    run_capture_cycle(tmp_path, AFTER, Source(answers), ("BTCUSDT",), requested=OPEN)
    store_packet(tmp_path, OPEN, _packet(OPEN, NOW))
    store_session_book(tmp_path, OPEN, "BTCUSDT", BookShot(Decimal("5"), NOW, NOW))
    ready = run_capture_cycle(tmp_path, AFTER, Source({}), ("BTCUSDT",), requested=OPEN)
    assert ready.finalized is True
    assert ready.readiness is not None
    assert ready.readiness.ready is True


def _closed_passer(root: Path) -> None:
    from cip.evaluation.classification import store_classification
    from cip.evaluation.eligibility import Lane
    from cip.evaluation.lane_fields import LaneFields, store_lane

    run_capture_cycle(root, NOW, Source(_pre_answers(OPEN, NOW)), ("BTCUSDT",))
    folder = root / "captures/session=2026-10-07/classification"
    for path in folder.glob("symbol=*.json"):
        path.unlink()
    item = SymbolClassification(
        symbol="BTCUSDT",
        base_asset="BTC",
        eur_stable=False,
        fan_token=False,
        monitoring_tag=False,
        delisting=False,
        deposits_suspended=False,
        withdrawals_suspended=False,
        pending_migration=False,
        coin_id="bitcoin",
        mapping="unambiguous",
        market_cap_usd=Decimal("1000000000"),
        market_cap_source_timestamp=NOW,
        cmc_id=None,
        cmc_market_cap_usd=None,
        cmc_source_timestamp=None,
    )
    store_classification(root, OPEN, item, NOW)
    store_lane(
        root,
        OPEN,
        LaneFields(
            symbol="BTCUSDT",
            base_asset="BTC",
            lane=Lane.NORMAL,
            coin_id="bitcoin",
            circulating_ratio=Decimal("0.8"),
            history_days=(OPEN - date(2017, 8, 17)).days,
            history_open=datetime(2017, 8, 17, tzinfo=UTC),
            market_cap_rank=1,
            fdv_to_market_cap=Decimal("1.2"),
            unlock_state="unavailable",
            market_source_timestamp=NOW,
            market_captured_at=NOW,
            history_captured_at=NOW,
            unlock_provider=None,
            unlock_provider_id=None,
            unlock_source_timestamp=None,
        ),
    )


def test_a_finalized_session_is_not_captured_again(tmp_path: Path) -> None:
    manifest = tmp_path / "sessions/date=2026-10-07/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    source = Source(_pre_answers(OPEN, NOW))
    report = run_capture_cycle(tmp_path, AFTER, source, ("BTCUSDT",), requested=OPEN)
    assert source.calls == []
    assert report.finalized is True
    assert report.readiness is None


def test_rejected_observations_leave_the_stored_book_unchanged(tmp_path: Path) -> None:
    later = BookShot(Decimal("5"), AFTER, NOW)
    with pytest.raises(EvaluationError, match="capture is after the close"):
        store_session_book(tmp_path, OPEN, "BTCUSDT", later)
    with pytest.raises(EvaluationError, match="capture is after the close"):
        store_session_book(tmp_path, OPEN, "BTCUSDT", BookShot(Decimal("5"), NOW, AFTER))
    store_session_book(
        tmp_path,
        OPEN,
        "BTCUSDT",
        BookShot(
            Decimal("5"),
            None,
            NOW,
            ((Decimal("1"), Decimal("2")),),
            ((Decimal("3"), Decimal("4")),),
        ),
    )
    stored = json.loads(
        (tmp_path / "captures/session=2026-10-07/book/symbol=BTCUSDT.json").read_text()
    )
    assert stored["bids"] == [["1", "2"]]
    assert stored["asks"] == [["3", "4"]]
    listed = tmp_path / "captures/session=2026-10-07/liquidity/symbol=ETHUSDT.json"
    listed.parent.mkdir(parents=True)
    listed.write_text(json.dumps({"spread_bps": ["4"]}))
    assert book_is_stored(tmp_path, OPEN, "ETHUSDT")
    empty = tmp_path / "captures/session=2026-10-07/liquidity/symbol=SOLUSDT.json"
    empty.write_text(json.dumps({"spread_snapshots": 0, "spread_bps": []}))
    assert book_is_stored(tmp_path, OPEN, "SOLUSDT") is False
    broken = tmp_path / "captures/session=2026-10-07/liquidity/symbol=XRPUSDT.json"
    broken.write_text("[]")
    assert book_is_stored(tmp_path, OPEN, "XRPUSDT") is False


def test_unusable_provider_payloads_are_refused(tmp_path: Path) -> None:
    def run(
        name: str,
        answers: Answers,
        symbols: tuple[str, ...] = ("BTCUSDT",),
    ) -> CycleReport:
        return run_capture_cycle(tmp_path / name, NOW, Source(answers), symbols)

    base = _pre_answers(OPEN, NOW)
    broken = dict(base)
    broken[(OPEN, "universe", None)] = "nope"
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run("universe", broken)
    broken = dict(base)
    broken[(OPEN, "ticker", None)] = ("BTCUSDT", NOW, NOW)
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run("ticker", broken)
    broken = dict(base)
    broken[(OPEN, "ticker", None)] = (["BTCUSDT"], NOW, AFTER)
    with pytest.raises(EvaluationError, match="capture is after the close"):
        run("ticker-late", broken)
    broken = dict(base)
    broken[(OPEN, "ticker", None)] = (["BTCUSDT"], AFTER, NOW)
    with pytest.raises(EvaluationError, match="capture is after the close"):
        run("ticker-source", broken)
    run("ticker-undated", {**base, (OPEN, "ticker", None): (["BTCUSDT"], None, NOW)}, ())
    broken = dict(base)
    broken[(OPEN, "peg", "USDCUSDT")] = ((Decimal("1"),), NOW, AFTER)
    with pytest.raises(EvaluationError, match="capture is after the close"):
        run("peg-late", broken)
    broken = dict(base)
    broken[(OPEN, "peg", "USDCUSDT")] = ((Decimal("1"),), AFTER, NOW)
    with pytest.raises(EvaluationError, match="capture is after the close"):
        run("peg-source", broken)
    broken = dict(base)
    broken[(OPEN, "peg", "USDCUSDT")] = (("1",), None, NOW)
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run("peg-text", broken)
    run("peg-undated", {**base, (OPEN, "peg", "USDCUSDT"): ((Decimal("1"),), None, NOW)}, ())
    broken = dict(base)
    broken[(OPEN, "regime", "btc_dominance")] = "nope"
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run("regime", broken)
    broken = dict(base)
    broken[(OPEN, "classification", "BTCUSDT")] = "nope"
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run("classification", broken)
    broken = dict(base)
    broken[(OPEN, "classification", "BTCUSDT")] = (
        SymbolClassification(
            symbol="ETHUSDT",
            base_asset="ETH",
            eur_stable=None,
            fan_token=None,
            monitoring_tag=None,
            delisting=None,
            deposits_suspended=None,
            withdrawals_suspended=None,
            pending_migration=None,
            coin_id=None,
            mapping="unmapped",
            market_cap_usd=None,
            market_cap_source_timestamp=None,
            cmc_id=None,
            cmc_market_cap_usd=None,
            cmc_source_timestamp=None,
        ),
        NOW,
    )
    with pytest.raises(EvaluationError, match="conflicting observation"):
        run("mismatch", broken)
    broken = dict(base)
    broken[(OPEN, "book", "BTCUSDT")] = "depth"
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run("book", broken)
    broken = dict(base)
    broken[(OPEN, "peg", "USDCUSDT")] = "nope"
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run("peg-shape", broken)
    broken = dict(base)
    broken[(OPEN, "classification", "BTCUSDT")] = ("BTCUSDT", NOW)
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run("classification-shape", broken)
    bars = dict(base)
    bars[(OPEN, "daily_bars", "BTCUSDT")] = "nope"
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run_capture_cycle(tmp_path / "bars", AFTER, Source(bars), ("BTCUSDT",), requested=OPEN)


def test_post_close_retries_and_refuses_a_malformed_hour(tmp_path: Path) -> None:
    failed = Source({}, fail={(OPEN, "daily_bars", "BTCUSDT")})
    report = run_capture_cycle(tmp_path, AFTER, failed, ("BTCUSDT",), requested=OPEN)
    assert report.retries == ("daily_bars:BTCUSDT",)
    hours_fail: Answers = {
        (OPEN, "daily_bars", "BTCUSDT"): _days(OPEN, Decimal("4800")),
    }
    source = Source(hours_fail, fail={(OPEN, "hour_bars", "BTCUSDT")})
    report = run_capture_cycle(tmp_path, AFTER, source, ("BTCUSDT",), requested=OPEN)
    assert "hour_bars:BTCUSDT" in report.retries
    bad = dict(hours_fail)
    bad[(OPEN, "hour_bars", "BTCUSDT")] = (HourQuote(1, 2, Decimal("1")),)
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run_capture_cycle(tmp_path / "malformed", AFTER, Source(bad), ("BTCUSDT",), requested=OPEN)
    typed = dict(hours_fail)
    typed[(OPEN, "hour_bars", "BTCUSDT")] = ("nope",)
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run_capture_cycle(tmp_path / "typed", AFTER, Source(typed), ("BTCUSDT",), requested=OPEN)
    hours = tmp_path / "malformed/captures/session=2026-10-07/hours/symbol=BTCUSDT.json"
    assert not hours.exists()
    short: Answers = {
        (OPEN, "daily_bars", "BTCUSDT"): _days(OPEN, Decimal("4800")),
        (OPEN, "hour_bars", "BTCUSDT"): _hours(OPEN, (Decimal("1"),) * 23),
    }
    first = run_capture_cycle(
        tmp_path / "short", AFTER, Source(short), ("BTCUSDT",), requested=OPEN
    )
    assert "hour_bars:BTCUSDT" in first.retries
    assert not (tmp_path / "short/captures/session=2026-10-07/hours").exists()
    again = Source(short)
    run_capture_cycle(tmp_path / "short", AFTER, again, ("BTCUSDT",), requested=OPEN)
    assert ("hour_bars", "BTCUSDT", OPEN) in again.calls
    full = _hours(OPEN, (Decimal("4800"),) + (Decimal("0"),) * 23)
    extra_open = full[-1].open_ms + _HOUR_MS
    extra = (*full, HourQuote(extra_open, extra_open + _HOUR_MS - 1, Decimal("9")))
    kept: Answers = {
        (OPEN, "daily_bars", "BTCUSDT"): _days(OPEN, Decimal("4800")),
        (OPEN, "hour_bars", "BTCUSDT"): extra,
    }
    report = run_capture_cycle(
        tmp_path / "extra", AFTER, Source(kept), ("BTCUSDT",), requested=OPEN
    )
    document = json.loads(
        (tmp_path / "extra/captures/session=2026-10-07/hours/symbol=BTCUSDT.json").read_text()
    )
    assert report.spike_counts == (("BTCUSDT", 1),)
    assert len(document["hours"]) == 24
    duplicate = (*full, full[0])
    refused: Answers = {
        (OPEN, "daily_bars", "BTCUSDT"): _days(OPEN, Decimal("4800")),
        (OPEN, "hour_bars", "BTCUSDT"): duplicate,
    }
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run_capture_cycle(
            tmp_path / "duplicate", AFTER, Source(refused), ("BTCUSDT",), requested=OPEN
        )
    negative = (*full[:-1], HourQuote(full[-1].open_ms, full[-1].close_ms, Decimal("-1")))
    bad_quote: Answers = {
        (OPEN, "daily_bars", "BTCUSDT"): _days(OPEN, Decimal("4800")),
        (OPEN, "hour_bars", "BTCUSDT"): negative,
    }
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run_capture_cycle(
            tmp_path / "negative", AFTER, Source(bad_quote), ("BTCUSDT",), requested=OPEN
        )


def test_a_duplicate_daily_date_is_not_turned_into_a_spike_count() -> None:
    repeated = (_bar(OPEN, Decimal("1")), _bar(OPEN, Decimal("1")))
    with pytest.raises(EvaluationError, match="capture is unusable"):
        _spike(OPEN, AFTER, repeated, ())


def test_spike_reads_the_policy_path_the_workload_sets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.yaml"
    monkeypatch.setenv("POLICY_PATH", str(missing))
    assert _policy_path() == missing
    with pytest.raises(PolicyError, match="cannot read policy file"):
        _spike(OPEN, AFTER, (), ())
    monkeypatch.setenv(
        "POLICY_PATH",
        str(Path(__file__).resolve().parents[3] / "policies" / "investment-policy.yaml"),
    )
    assert _spike(OPEN, AFTER, (), ()) is None


def test_readiness_waits_for_each_pre_close_file(tmp_path: Path) -> None:
    session = tmp_path / "captures/session=2026-10-07"
    (session).mkdir(parents=True)
    (session / "universe.json").write_text("{}")
    (session / "ticker.json").write_text("{}")
    report = run_capture_cycle(
        tmp_path,
        AFTER,
        Source({}, fail={(OPEN, "daily_bars", "BTCUSDT")}),
        (),
        requested=OPEN,
    )
    assert report.readiness is None
    (session / "peg").mkdir()
    (session / "peg/symbol=USDCUSDT.json").write_text("{}")
    report = run_capture_cycle(
        tmp_path,
        AFTER,
        Source({}, fail={(OPEN, "daily_bars", "BTCUSDT")}),
        (),
        requested=OPEN,
    )
    assert report.readiness is None
    (session / "regime").mkdir()
    (session / "regime/btc_dominance.json").write_text("{}")
    (session / "regime/stablecoin_supply.json").write_text("{}")
    (session / "universe.json").write_text(json.dumps({"symbols": "BTCUSDT"}))
    report = run_capture_cycle(
        tmp_path,
        AFTER,
        Source({}, fail={(OPEN, "daily_bars", "BTCUSDT")}),
        (),
        requested=OPEN,
    )
    assert report.readiness is None
    cycle = capture_cycle(NOW)
    _post_close(tmp_path, cycle, Source({}), (), [], [], [], [])
    _one_bar(tmp_path, cycle, Source({}), "BTCUSDT", [], [], [], [])
    assert _present(tmp_path, OPEN, "book", "BTCUSDT") is False


def test_a_stored_bar_file_must_still_parse(tmp_path: Path) -> None:
    answers: Answers = {
        (OPEN, "daily_bars", "BTCUSDT"): _days(OPEN, Decimal("4800")),
        (OPEN, "hour_bars", "BTCUSDT"): _hours(OPEN, (Decimal("4800"),) + (Decimal("0"),) * 23),
    }
    root = tmp_path / "bars"
    run_capture_cycle(root, AFTER, Source(answers), ("BTCUSDT",), requested=OPEN)
    final = root / "captures/session=2026-10-07/final/symbol=BTCUSDT.json"
    final.write_text(json.dumps({"bars": ["nope"]}))
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run_capture_cycle(root, AFTER, Source({}), ("BTCUSDT",), requested=OPEN)
    final.write_text("[]")
    with pytest.raises(EvaluationError, match="capture is unusable"):
        run_capture_cycle(root, AFTER, Source({}), ("BTCUSDT",), requested=OPEN)
    manifest = tmp_path / "sessions/date=2026-10-07/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    source = Source(_pre_answers(OPEN, NOW))
    report = run_capture_cycle(tmp_path, NOW, source, ("BTCUSDT",))
    assert source.calls == []
    assert report.skipped == ("finalized:2026-10-07",)
    assert report.finalized is False
    text = Path(cycle_module.__file__).read_text()
    assert "run_daily_scan" not in text
    assert "prod_decisions_started_at" not in text
    assert "score_v2_shadow_started_at" not in text
    assert "prod_shadow_started_at" not in text


def test_lane_evidence_runs_after_classification_and_before_the_book(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        "POLICY_PATH",
        str(Path(__file__).resolve().parents[3] / "policies" / "investment-policy.yaml"),
    )
    root = tmp_path / "lanes"
    session = "2026-10-07"
    folder = root / f"captures/session={session}"
    for name in ("ticker.json", "peg/symbol=USDCUSDT.json"):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    for series in ("btc_dominance", "stablecoin_supply"):
        path = folder / "regime" / f"{series}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (folder / "universe.json").write_text(json.dumps({"symbols": ["BTCUSDT", "LOWUSDT"]}))
    from cip.evaluation.classification import store_classification
    from cip.evaluation.classification_bundle import store_component

    stamp = datetime(2026, 10, 7, 18, tzinfo=UTC)
    store_classification(root, OPEN, _stored("BTCUSDT", "BTC", Decimal("1000000000")), NOW)
    store_classification(root, OPEN, _stored("LOWUSDT", "LOW", Decimal("1000")), NOW)
    store_component(
        root,
        OPEN,
        "markets",
        [
            {
                "circulating_supply": "80",
                "fully_diluted_valuation": "500",
                "id": "bitcoin",
                "last_updated": "2026-10-07T18:00:00Z",
                "market_cap": "400",
                "market_cap_rank": 4,
                "total_supply": "100",
            }
        ],
        NOW,
    )
    answers: Answers = {
        (OPEN, "book", "BTCUSDT"): BookShot(Decimal("1"), stamp, NOW),
        (OPEN, "listing", "BTCUSDT"): [[1_502_928_000_000]],
    }
    source = Source(answers)
    report = run_capture_cycle(root, NOW, source, ("BTCUSDT",))
    assert ("listing", "BTCUSDT", OPEN) in source.calls
    assert ("listing", "LOWUSDT", OPEN) not in source.calls
    assert ("book", "BTCUSDT", OPEN) in source.calls
    assert report.retries == ()
    assert (folder / "lane/symbol=BTCUSDT.json").is_file()
    assert (folder / "candidates/symbol=BTCUSDT.json").is_file()
    assert not (folder / "lane/symbol=LOWUSDT.json").exists()


def test_a_listing_failure_is_a_retry_and_the_book_still_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        "POLICY_PATH",
        str(Path(__file__).resolve().parents[3] / "policies" / "investment-policy.yaml"),
    )
    root = tmp_path / "retry"
    folder = root / "captures/session=2026-10-07"
    for name in ("ticker.json", "peg/symbol=USDCUSDT.json"):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    for series in ("btc_dominance", "stablecoin_supply"):
        path = folder / "regime" / f"{series}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (folder / "universe.json").write_text(json.dumps({"symbols": ["BTCUSDT"]}))
    from cip.evaluation.classification import store_classification

    store_classification(root, OPEN, _stored("BTCUSDT", "BTC", Decimal("1000000000")), NOW)
    source = Source(
        {(OPEN, "book", "BTCUSDT"): BookShot(Decimal("1"), None, NOW)},
        fail={(OPEN, "listing", "BTCUSDT")},
    )
    report = run_capture_cycle(root, NOW, source, ("BTCUSDT",))
    assert report.retries == ("listing:BTCUSDT",)
    assert ("book", "BTCUSDT", OPEN) in source.calls
    assert not (folder / "lane/symbol=BTCUSDT.json").exists()


def test_a_lane_conflict_is_reported_after_the_book(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        "POLICY_PATH",
        str(Path(__file__).resolve().parents[3] / "policies" / "investment-policy.yaml"),
    )
    root = tmp_path / "conflict"
    folder = root / "captures/session=2026-10-07"
    _skip_files(folder)
    (folder / "universe.json").write_text(json.dumps({"symbols": ["BTCUSDT"]}))
    from cip.evaluation.classification import store_classification
    from cip.evaluation.classification_bundle import store_component

    store_classification(root, OPEN, _stored("BTCUSDT", "BTC", Decimal("1000000000")), NOW)
    store_component(
        root,
        OPEN,
        "markets",
        [
            {
                "circulating_supply": "80",
                "fully_diluted_valuation": "500",
                "id": "bitcoin",
                "last_updated": "2026-10-08T00:00:01Z",
                "market_cap": "400",
                "market_cap_rank": 4,
                "total_supply": "100",
            }
        ],
        NOW,
    )
    source = Source({(OPEN, "book", "BTCUSDT"): BookShot(Decimal("1"), None, NOW)})
    with pytest.raises(EvaluationError, match="after the close"):
        run_capture_cycle(root, NOW, source, ("BTCUSDT",))
    assert ("book", "BTCUSDT", OPEN) in source.calls
    assert ("listing", "BTCUSDT", OPEN) not in source.calls


def _skip_files(folder: Path) -> None:
    for name in ("ticker.json", "peg/symbol=USDCUSDT.json", "regime/btc_dominance.json"):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    path = folder / "regime/stablecoin_supply.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


def _stored(symbol: str, base: str, cap: Decimal) -> SymbolClassification:
    return SymbolClassification(
        symbol=symbol,
        base_asset=base,
        eur_stable=False,
        fan_token=False,
        monitoring_tag=False,
        delisting=False,
        deposits_suspended=False,
        withdrawals_suspended=False,
        pending_migration=False,
        coin_id="bitcoin",
        mapping="unambiguous",
        market_cap_usd=cap,
        market_cap_source_timestamp=datetime(2026, 10, 7, 18, tzinfo=UTC),
        cmc_id=None,
        cmc_market_cap_usd=None,
        cmc_source_timestamp=None,
    )
