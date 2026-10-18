import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from moto import mock_aws

from cip.adapters.market import Depth, DepthLevel, ExchangeInfo, Kline, Ticker24h
from cip.domain.errors import (
    EvaluationError,
    ExchangeBannedError,
    ExchangeGeoBlockedError,
    MarketDataError,
    RateLimited,
    RecorderError,
)
from cip.evaluation.capture_cycle import CycleReport, capture_cycle
from cip.evaluation.session import SessionReadiness, session_close
from cip.handlers import session_capture
from cip.handlers.session_capture import EvidenceSource, TemporaryFailure
from cip.recorders.observation import Observation


@pytest.fixture(autouse=True)
def _no_catalog_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_capture, "_sleep", lambda _seconds: None)


def test_pause_delegates_to_the_sleeper(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []
    monkeypatch.setattr(session_capture, "_sleep", seen.append)
    session_capture._pause(0)
    assert seen == [0]


OPEN = date(2026, 10, 7)
NOW = datetime(2026, 10, 7, 19, tzinfo=UTC)
AFTER = datetime(2026, 10, 8, 1, tzinfo=UTC)
BUCKET = "cip-prod-data-test"
REPO = Path(__file__).parents[3]
POLICY = REPO / "policies" / "investment-policy.yaml"
BASE = "https://data-api.binance.vision"


class _Context:
    function_name = "cip-test-session-capture"
    memory_limit_in_mb = 512
    invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:cip-test"
    aws_request_id = "req-1"


class _Spot:
    def __init__(self) -> None:
        self.depth_calls = 0
        self.info_calls = 0
        self.fail: Exception | None = None
        self.empty_book = False
        self.peg: tuple[Kline, ...] = (_hour(OPEN, 0, "1.0001"),)

    def exchange_info(self) -> ExchangeInfo:
        self.info_calls += 1
        if self.fail is not None:
            raise self.fail
        return ExchangeInfo.model_validate(
            {
                "symbols": [
                    {
                        "baseAsset": "BTC",
                        "quoteAsset": "USDT",
                        "status": "TRADING",
                        "symbol": "BTCUSDT",
                    }
                ]
            }
        )

    def ticker_24hr(self) -> tuple[Ticker24h, ...]:
        return (
            Ticker24h.model_validate(
                {"lastPrice": "100", "quoteVolume": "10", "symbol": "BTCUSDT"}
            ),
        )

    def klines(self, symbol: str, *, interval: str, limit: int) -> tuple[Kline, ...]:
        del symbol, interval, limit
        return self.peg

    def depth(self, symbol: str, *, limit: int) -> Depth:
        del symbol, limit
        self.depth_calls += 1
        if self.empty_book:
            return Depth(last_update_id=1, bids=(), asks=())
        return Depth(
            last_update_id=1,
            bids=(DepthLevel(Decimal("100"), Decimal("2")),),
            asks=(DepthLevel(Decimal("101"), Decimal("3")),),
        )


def _hour(day: date, hour: int, close: str, *, quote: str = "1") -> Kline:
    open_ms = int(datetime(day.year, day.month, day.day, hour, tzinfo=UTC).timestamp()) * 1000
    return Kline(
        open_ms,
        Decimal("1"),
        Decimal("1"),
        Decimal("1"),
        Decimal(close),
        Decimal("1"),
        open_ms + 3_600_000 - 1,
        Decimal(quote),
        1,
    )


def _daily_row(day: date, quote: str = "100") -> list[object]:
    open_ms = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * 1000
    return [open_ms, "1", "2", "1", "1", "1", open_ms + 86_400_000 - 1, quote, 1, "0.4", "0.4"]


def _source(spot: _Spot | None = None, *, now: datetime = NOW) -> tuple[EvidenceSource, _Spot]:
    market = spot or _Spot()

    def reader(url: str, params: dict[str, str] | None) -> object:
        del params
        if url.endswith("/global"):
            return {"data": {"market_cap_percentage": {"btc": "54.2"}, "updated_at": 1_759_000_000}}
        if "stablecoincharts" in url:
            return [{"date": 1_759_000_000, "totalCirculatingUSD": {"peggedUSD": "10"}}]
        if params_of(url):
            return []
        catalog = _catalog(url)
        if catalog is not None:
            return catalog
        raise AssertionError(url)

    return (
        EvidenceSource(market, now, Decimal("0.02"), BASE, reader),
        market,
    )


def params_of(url: str) -> bool:
    return url.endswith("/klines")


def _catalog(url: str) -> object | None:
    if url.endswith("get-all-asset"):
        return {
            "data": [
                {
                    "assetCode": "BTC",
                    "delisted": False,
                    "preDelist": False,
                    "swapTag": "no",
                    "tags": ["fan_token"],
                }
            ],
            "success": True,
        }
    if url.endswith("getNetworkCoinAll"):
        return {
            "data": [{"coin": "BTC", "depositAllEnable": True, "withdrawAllEnable": False}],
            "success": True,
        }
    if "exchanges/binance/tickers" in url:
        return {
            "tickers": [
                {
                    "base": "BTC",
                    "coin_id": "bitcoin",
                    "market": {"identifier": "binance"},
                    "target": "USDT",
                }
            ]
        }
    if "category=eur-stablecoin" in url:
        return [{"id": "eurite"}]
    if "ids=" in url:
        return [{"id": "bitcoin", "last_updated": "2026-10-07T18:00:00Z", "market_cap": 1}]
    if url.endswith("symbol/list"):
        return {"data": [{"cmcUniqueId": 1, "symbol": "BTCUSDT"}]}
    return None


def _bucket(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setenv("DATA_BUCKET", BUCKET)
    monkeypatch.setenv("CAPTURE_SYMBOLS", "BTCUSDT")
    client = __import__("boto3").client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=BUCKET)
    return client


def test_pre_close_stores_one_book_and_the_retry_does_not_take_another(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with mock_aws():
        client = _bucket(monkeypatch)
        client.put_object(
            Bucket=BUCKET, Key="captures/session=2026-10-06/universe.json", Body=b"sealed"
        )
        client.put_object(Bucket=BUCKET, Key="decisions/keep.json", Body=b"clock")
        root = tmp_path / "capture"
        source, _spot = _source()
        first = session_capture.capture(
            {"session": "2026-10-09", "date": "2026-10-06"},
            _Context(),
            now=NOW,
            source=source,
            client=client,
            root=root,
        )
        book = "captures/session=2026-10-07/book/symbol=BTCUSDT.json"
        assert first["open_session"] == "2026-10-07"
        assert first["pre_close"] == "2026-10-07"
        assert first["post_close"] is None
        assert first["finalized"] is False
        assert first["ready"] is None
        assert "classification:BTCUSDT" not in first["retries"]
        assert book in first["written"]
        assert "captures/session=2026-10-07/classification/symbol=BTCUSDT.json" in first["written"]
        assert "GET /api/v3/depth BTCUSDT" in first["provider_calls"]
        assert any("get-all-asset" in call for call in first["provider_calls"])
        stored = json.loads(client.get_object(Bucket=BUCKET, Key=book)["Body"].read())
        assert stored["spread_snapshots"] == 1
        assert len(stored["spread_bps"]) == 1
        assert (
            client.get_object(Bucket=BUCKET, Key="captures/session=2026-10-06/universe.json")[
                "Body"
            ].read()
            == b"sealed"
        )
        second_source, second_spot = _source()
        second = session_capture.capture(
            {}, _Context(), now=NOW, source=second_source, client=client, root=root
        )
        assert second_spot.depth_calls == 0
        assert second_spot.info_calls == 0
        assert "book:BTCUSDT" in second["skipped"]
        assert "classification:BTCUSDT" in second["skipped"]
        assert second["provider_calls"] == []
        assert book in second["unchanged"]
        assert book not in second["written"]
        names = [item["Key"] for item in client.list_objects_v2(Bucket=BUCKET)["Contents"]]
        assert names.count(book) == 1
        assert not any("absences/" in name for name in names)
        kept = client.get_object(Bucket=BUCKET, Key="decisions/keep.json")["Body"].read()
        assert kept == b"clock"


def test_a_banned_host_stores_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with mock_aws():
        client = _bucket(monkeypatch)
        spot = _Spot()
        spot.fail = ExchangeBannedError("Binance returned 418")
        source, _market = _source(spot)
        with pytest.raises(ExchangeBannedError):
            session_capture.capture(
                {}, _Context(), now=NOW, source=source, client=client, root=tmp_path
            )
        assert "Contents" not in client.list_objects_v2(Bucket=BUCKET)


def test_post_close_keeps_an_incomplete_hour_grid_unstored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with mock_aws():
        client = _bucket(monkeypatch)
        rows = [_daily_row(OPEN - timedelta(days=offset), "2400") for offset in range(30, -1, -1)]
        hours = [_hour(OPEN, hour, "1", quote="1") for hour in range(23)]

        def reader(url: str, params: dict[str, str] | None) -> object:
            if params and params.get("interval") == "1d":
                return [*rows, _daily_row(OPEN + timedelta(days=1), "9")]
            if params and params.get("interval") == "1h":
                return [
                    [
                        candle.open_time,
                        "1",
                        "1",
                        "1",
                        "1",
                        "1",
                        candle.close_time,
                        format(candle.quote_volume, "f"),
                        1,
                    ]
                    for candle in hours
                ]
            if url.endswith("/global"):
                return {
                    "data": {"market_cap_percentage": {"btc": "54.2"}, "updated_at": 1_759_000_000}
                }
            catalog = _catalog(url)
            if catalog is not None:
                return catalog
            return [{"date": 1_759_000_000, "totalCirculatingUSD": {"peggedUSD": "10"}}]

        source = EvidenceSource(_Spot(), AFTER, Decimal("0.02"), BASE, reader)
        result = session_capture.capture(
            {}, _Context(), now=AFTER, source=source, client=client, root=tmp_path
        )
        assert result["pre_close"] == "2026-10-08"
        assert result["post_close"] == "2026-10-07"
        assert "hour_bars:BTCUSDT" in result["retries"]
        names = [item["Key"] for item in client.list_objects_v2(Bucket=BUCKET)["Contents"]]
        assert not any(name.endswith("hours/symbol=BTCUSDT.json") for name in names)
        assert any(name.endswith("final/symbol=BTCUSDT.json") for name in names)


def test_a_sealed_clock_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with mock_aws():
        client = _bucket(monkeypatch)
        source, spot = _source()
        result = session_capture.capture(
            {},
            _Context(),
            now=datetime(2026, 10, 6, 12, tzinfo=UTC),
            source=source,
            client=client,
            root=tmp_path,
        )
        assert spot.info_calls == 0
        assert result["pre_close"] is None
        assert result["post_close"] is None
        assert result["written"] == []
        assert result["provider_calls"] == []


def test_a_manifest_blocks_another_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with mock_aws():
        client = _bucket(monkeypatch)
        key = "sessions/date=2026-10-07/manifest.json"
        client.put_object(Bucket=BUCKET, Key=key, Body=b'{"sealed":true}')
        spot = _Spot()
        source, _market = _source(spot)
        result = session_capture.capture(
            {}, _Context(), now=NOW, source=source, client=client, root=tmp_path
        )
        assert spot.info_calls == 0
        assert result["finalized"] is False
        assert client.get_object(Bucket=BUCKET, Key=key)["Body"].read() == b'{"sealed":true}'


def test_readiness_is_reported_when_the_cycle_returns_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with mock_aws():
        client = _bucket(monkeypatch)
        readiness = SessionReadiness(
            ready=False,
            blocks=("classification_missing",),
            decision_notes=(),
            score_weights="absent",
        )
        cycle = capture_cycle(NOW)
        report = CycleReport(
            cycle, (("universe", None),), ("book:BTCUSDT",), (), (), readiness, False
        )

        def run(*_args: object, **_kwargs: object) -> CycleReport:
            return report

        monkeypatch.setattr(session_capture, "run_capture_cycle", run)
        source, _market = _source()
        result = session_capture.capture(
            {}, _Context(), now=NOW, source=source, client=client, root=tmp_path
        )
        assert result["ready"] is False
        assert result["blocks"] == ["classification_missing"]
        assert result["calls"] == [{"kind": "universe", "symbol": None}]


def test_the_clock_is_used_when_the_caller_does_not_supply_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with mock_aws():
        _bucket(monkeypatch)
        monkeypatch.setattr(session_capture, "_clock", lambda: NOW)
        monkeypatch.setenv("CAPTURE_ROOT", str(tmp_path / "default-root"))
        source, _market = _source()
        result = session_capture.capture({"date": "2026-10-09"}, _Context(), source=source)
        assert result["open_session"] == "2026-10-07"


def test_owned_clients_close(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with mock_aws():
        client = _bucket(monkeypatch)
        source, _market = _source()
        closed = {"count": 0}

        class _Door:
            def close(self) -> None:
                closed["count"] += 1

        monkeypatch.setattr(
            session_capture, "_open_source", lambda _now, _remaining: (source, (_Door(),))
        )
        session_capture.capture({}, _Context(), now=NOW, client=client, root=tmp_path)
        assert closed["count"] == 1


def test_public_reader_uses_the_policy_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLICY_PATH", str(POLICY))
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "coingecko" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "market_cap_percentage": {"btc": 54.2},
                        "updated_at": 1_759_000_000,
                    }
                },
            )
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def client(**kwargs: object) -> httpx.Client:
        assert kwargs["follow_redirects"] is False
        return real_client(transport=transport, timeout=10.0, follow_redirects=False)

    class _Market:
        def __init__(self, base_url: str) -> None:
            assert base_url == BASE

        def close(self) -> None:
            seen.append("closed")

    monkeypatch.setattr(session_capture.httpx, "Client", client)
    monkeypatch.setattr(session_capture, "BinanceMarketClient", _Market)
    source, owned = session_capture._open_source(NOW)
    try:
        with pytest.raises(TemporaryFailure):
            source.fetch(OPEN, "regime", "stablecoin_supply")
    finally:
        for item in owned:
            item.close()
    assert any("coingecko" not in item for item in seen)
    assert seen[-1] == "closed"


def _full_ticker_page() -> dict[str, object]:
    return {
        "tickers": [
            {
                "base": "BTC",
                "coin_id": "bitcoin",
                "market": {"identifier": "binance"},
                "target": "USDT",
            }
        ]
        * 100
    }


def test_retry_after_then_a_longer_fallback_finishes_one_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []
    pages: dict[str, int] = {}

    def reader(url: str, params: dict[str, str] | None) -> object:
        del params
        if "tickers?page=" in url:
            pages[url] = pages.get(url, 0) + 1
            page = url.rsplit("=", 1)[-1]
            if page != "6":
                return _full_ticker_page()
            if pages[url] == 1:
                raise RateLimited("25")
            if pages[url] == 2:
                raise RateLimited("later")
            return {"tickers": []}
        catalog = _catalog(url)
        if catalog is not None:
            return catalog
        raise AssertionError(url)

    source = EvidenceSource(_Spot(), NOW, Decimal("0.02"), BASE, reader)
    monkeypatch.setattr(session_capture, "_pause", waits.append)
    item, _captured = source.fetch(OPEN, "classification", "BTCUSDT")
    assert item.symbol == "BTCUSDT"
    assert pages["https://api.coingecko.com/api/v3/exchanges/binance/tickers?page=6"] == 3
    assert all(count == 1 for page, count in pages.items() if not page.endswith("=6"))
    assert 25.0 in waits
    assert 45.0 in waits
    again = source.fetch(OPEN, "classification", "BTCUSDT")
    assert again[0].symbol == "BTCUSDT"
    assert pages["https://api.coingecko.com/api/v3/exchanges/binance/tickers?page=6"] == 3
    outcomes = [row["outcome"] for row in source.catalog_trace]
    assert "succeeded" in outcomes
    assert "abandoned" not in outcomes
    page = "https://api.coingecko.com/api/v3/exchanges/binance/tickers?page=1"
    stored = source._pages[page]
    assert source._get(page) is stored
    assert source.calls.count(f"GET {page}") == 1


def test_a_429_stops_when_the_execution_budget_cannot_cover_the_wait() -> None:
    def reader(url: str, params: dict[str, str] | None) -> object:
        del params
        if "tickers?page=" in url:
            raise RateLimited("120")
        catalog = _catalog(url)
        if catalog is not None:
            return catalog
        raise AssertionError(url)

    source = EvidenceSource(_Spot(), NOW, Decimal("0.02"), BASE, reader, remaining=lambda: 20_000)
    with pytest.raises(TemporaryFailure, match="status 429"):
        source.fetch(OPEN, "classification", "BTCUSDT")
    assert source.catalog_trace[-1]["outcome"] == "abandoned"
    recorded = len(source.calls)
    with pytest.raises(TemporaryFailure, match="classification catalog is incomplete"):
        source.fetch(OPEN, "classification", "ETHUSDT")
    assert len(source.calls) == recorded


def test_a_tiny_budget_does_not_start_a_coingecko_page() -> None:
    def reader(url: str, params: dict[str, str] | None) -> object:
        del params
        if "coingecko.com" in url:
            raise AssertionError("the request must not start")
        catalog = _catalog(url)
        if catalog is not None:
            return catalog
        raise AssertionError(url)

    source = EvidenceSource(_Spot(), NOW, Decimal("0.02"), BASE, reader, remaining=lambda: 1_000)
    with pytest.raises(TemporaryFailure, match="status 429"):
        source.fetch(OPEN, "classification", "BTCUSDT")
    assert not any("coingecko" in call for call in source.calls)


def test_catalog_budget_uses_the_lambda_clock() -> None:
    assert session_capture._budget(object())() == 180_000
    clock = type("Clock", (), {"get_remaining_time_in_millis": lambda self: 90_000})
    assert session_capture._budget(clock())() == 90_000
    blank = type("Blank", (), {"get_remaining_time_in_millis": lambda self: "soon"})
    assert session_capture._budget(blank())() == 180_000
    flag = type("Flag", (), {"get_remaining_time_in_millis": lambda self: True})
    assert session_capture._budget(flag())() == 180_000
    assert session_capture._catalog_delay(0, "-1") == 30.0
    assert session_capture._catalog_delay(0, "inf") == 30.0
    assert session_capture._catalog_delay(1, "nan") == 45.0
    assert session_capture._catalog_delay(0, "0") == 0.0
    assert session_capture._catalog_delay(3, "10") is None


def test_a_throttled_later_source_keeps_the_ticker_artifact_and_the_book(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def reader(url: str, params: dict[str, str] | None) -> object:
        del params
        if url.endswith("/global"):
            return {"data": {"market_cap_percentage": {"btc": "54.2"}, "updated_at": 1_759_000_000}}
        if "stablecoincharts" in url:
            return [{"date": 1_759_000_000, "totalCirculatingUSD": {"peggedUSD": "10"}}]
        if "eur-stablecoin" in url:
            raise RateLimited("60")
        catalog = _catalog(url)
        if catalog is not None:
            return catalog
        raise AssertionError(url)

    with mock_aws():
        client = _bucket(monkeypatch)
        source = EvidenceSource(_Spot(), NOW, Decimal("0.02"), BASE, reader)
        first = session_capture.capture(
            {}, _Context(), now=NOW, source=source, client=client, root=tmp_path
        )
        ticker = "captures/session=2026-10-07/classification_sources/tickers.json"
        assert ticker in first["written"]
        assert not any("/classification/symbol=" in key for key in first["written"])
        assert "classification:BTCUSDT" in first["retries"]
        assert "captures/session=2026-10-07/book/symbol=BTCUSDT.json" in first["written"]
        resumed = EvidenceSource(_Spot(), NOW, Decimal("0.02"), BASE, reader)
        second = session_capture.capture(
            {}, _Context(), now=NOW, source=resumed, client=client, root=tmp_path
        )
        assert not any("tickers?page=" in call for call in second["provider_calls"])
        assert "classification:BTCUSDT" in second["retries"]


def test_ticker_pages_survive_a_budget_stop_and_the_next_capture_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requested: list[int] = []

    def reader(url: str, params: dict[str, str] | None) -> object:
        del params
        if url.endswith("/global"):
            return {"data": {"market_cap_percentage": {"btc": "54.2"}, "updated_at": 1_759_000_000}}
        if "stablecoincharts" in url:
            return [{"date": 1_759_000_000, "totalCirculatingUSD": {"peggedUSD": "10"}}]
        if "tickers?page=" in url:
            page = int(url.rsplit("=", 1)[-1])
            requested.append(page)
            if page >= 6:
                raise RateLimited("60")
            return _full_ticker_page()
        catalog = _catalog(url)
        if catalog is not None:
            return catalog
        raise AssertionError(url)

    with mock_aws():
        client = _bucket(monkeypatch)
        source = EvidenceSource(
            _Spot(), NOW, Decimal("0.02"), BASE, reader, remaining=lambda: 20_000
        )
        first = session_capture.capture(
            {}, _Context(), now=NOW, source=source, client=client, root=tmp_path
        )
        prefix = "captures/session=2026-10-07/classification_sources/tickers/"
        assert requested == [1, 2, 3, 4, 5, 6]
        assert [f"{prefix}page={page}.json" for page in range(1, 6)] == [
            key for key in first["written"] if key.startswith(prefix)
        ]
        assert (
            "captures/session=2026-10-07/classification_sources/tickers.json"
            not in (first["written"])
        )
        assert not any("/classification/symbol=" in key for key in first["written"])
        assert "captures/session=2026-10-07/book/symbol=BTCUSDT.json" in first["written"]
        assert source.catalog_trace[-1]["outcome"] == "abandoned"
        requested.clear()
        resumed = EvidenceSource(
            _Spot(), NOW, Decimal("0.02"), BASE, reader, remaining=lambda: 20_000
        )
        session_capture.capture(
            {}, _Context(), now=NOW, source=resumed, client=client, root=tmp_path
        )
        assert requested == [6]


def test_a_coingecko_429_is_retried_before_the_catalog_is_abandoned() -> None:
    calls = {"n": 0}

    def reader(url: str, params: dict[str, str] | None) -> object:
        del params
        if "tickers?page=" in url:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RecorderError("status 429")
        catalog = _catalog(url)
        if catalog is not None:
            return catalog
        raise AssertionError(url)

    source = EvidenceSource(_Spot(), NOW, Decimal("0.02"), BASE, reader)
    item, _captured = source.fetch(OPEN, "classification", "BTCUSDT")
    assert item.symbol == "BTCUSDT"
    assert calls["n"] == 3

    def always(url: str, params: dict[str, str] | None) -> object:
        del url, params
        raise RecorderError("status 429")

    blocked = EvidenceSource(_Spot(), NOW, Decimal("0.02"), BASE, always)
    with pytest.raises(TemporaryFailure):
        blocked.fetch(OPEN, "classification", "BTCUSDT")
    recorded = len(blocked.calls)
    with pytest.raises(TemporaryFailure, match="classification catalog is incomplete"):
        blocked.fetch(OPEN, "classification", "BTCUSDT")
    assert len(blocked.calls) == recorded


def test_a_non_429_catalog_error_is_not_retried() -> None:
    def reader(url: str, params: dict[str, str] | None) -> object:
        del params
        if url.endswith("get-all-asset"):
            raise RecorderError("status 500")
        catalog = _catalog(url)
        if catalog is not None:
            return catalog
        raise AssertionError(url)

    source = EvidenceSource(_Spot(), NOW, Decimal("0.02"), BASE, reader)
    with pytest.raises(TemporaryFailure, match="status 500"):
        source.fetch(OPEN, "classification", "BTCUSDT")
    assert sum(call.endswith("get-all-asset") for call in source.calls) == 1


def test_classification_retries_an_incomplete_catalog_and_an_unknown_base() -> None:
    full = {
        "tickers": [
            {
                "base": "BTC",
                "coin_id": "bitcoin",
                "market": {"identifier": "binance"},
                "target": "USDT",
            }
        ]
        * 100
    }

    def reader(url: str, params: dict[str, str] | None) -> object:
        del params
        if "tickers?page=" in url:
            return full
        catalog = _catalog(url)
        if catalog is not None:
            return catalog
        raise AssertionError(url)

    source = EvidenceSource(_Spot(), NOW, Decimal("0.02"), BASE, reader)
    with pytest.raises(TemporaryFailure, match="classification catalog is incomplete"):
        source.fetch(OPEN, "classification", "BTCUSDT")
    calls_after_failure = len(source.calls)
    with pytest.raises(TemporaryFailure, match="classification catalog is incomplete"):
        source.fetch(OPEN, "classification", "BTCUSDT")
    assert len(source.calls) == calls_after_failure
    with pytest.raises(EvaluationError, match="unusable"):
        source.fetch(OPEN, "classification", "bad symbol")
    missing = EvidenceSource(_Spot(), NOW, Decimal("0.02"), BASE, _source()[0]._reader)
    with pytest.raises(TemporaryFailure, match="classification base is unresolved"):
        missing.fetch(OPEN, "classification", "ETHUSDT")
    again = missing.fetch(OPEN, "classification", "BTCUSDT")
    assert again[0].symbol == "BTCUSDT"
    assert missing.calls.count("GET /api/v3/exchangeInfo") == 1


def test_classification_and_a_ticker_at_the_close_make_no_provider_call() -> None:
    late, _spot = _source(now=session_close(OPEN))
    with pytest.raises(TemporaryFailure, match="after the close"):
        late.fetch(OPEN, "classification", "BTCUSDT")
    with pytest.raises(TemporaryFailure, match="after the close"):
        late.fetch(OPEN, "ticker", None)
    assert late.calls == []


def test_provider_failures_stay_retryable_and_a_ban_does_not() -> None:
    source, spot = _source()
    spot.fail = ExchangeGeoBlockedError("451")
    with pytest.raises(TemporaryFailure):
        source.fetch(OPEN, "universe", None)
    banned, market = _source()
    market.fail = ExchangeBannedError("418")
    with pytest.raises(ExchangeBannedError):
        banned.fetch(OPEN, "universe", None)

    def broken(url: str, params: dict[str, str] | None) -> object:
        del url, params
        raise httpx.ConnectError("down")

    flaky = EvidenceSource(_Spot(), NOW, Decimal("0.02"), BASE, broken)
    with pytest.raises(TemporaryFailure):
        flaky.fetch(OPEN, "regime", "btc_dominance")

    def banned(url: str, params: dict[str, str] | None) -> object:
        del url, params
        raise RecorderError("status 418")

    def rejected(url: str, params: dict[str, str] | None) -> object:
        del url, params
        raise RecorderError("status 500")

    klines = EvidenceSource(_Spot(), AFTER, Decimal("0.02"), BASE, banned)
    with pytest.raises(ExchangeBannedError):
        klines.fetch(OPEN, "daily_bars", "BTCUSDT")
    retryable = EvidenceSource(_Spot(), AFTER, Decimal("0.02"), BASE, rejected)
    with pytest.raises(TemporaryFailure):
        retryable.fetch(OPEN, "hour_bars", "BTCUSDT")


def test_an_empty_book_and_a_closed_peg_hour_are_not_stored() -> None:
    spot = _Spot()
    spot.empty_book = True
    source, _market = _source(spot)
    with pytest.raises(TemporaryFailure, match="book is empty"):
        source.fetch(OPEN, "book", "BTCUSDT")
    spot.peg = (_hour(OPEN + timedelta(days=1), 0, "1"),)
    with pytest.raises(TemporaryFailure, match="peg hour is still open"):
        source.fetch(OPEN, "peg", "USDCUSDT")
    with pytest.raises(EvaluationError, match="unusable"):
        source.fetch(OPEN, "peg", "ETHUSDT")
    with pytest.raises(EvaluationError, match="unusable"):
        source.fetch(OPEN, "book", "bad symbol")


def test_regime_after_the_close_is_not_returned() -> None:
    def reader(url: str, params: dict[str, str] | None) -> object:
        del params
        stamp = int(session_close(OPEN).timestamp())
        if url.endswith("/global"):
            return {"data": {"market_cap_percentage": {"btc": "1"}, "updated_at": stamp}}
        return [{"date": stamp, "totalCirculatingUSD": {"peggedUSD": "1"}}]

    source = EvidenceSource(_Spot(), NOW, Decimal("0.02"), BASE, reader)
    with pytest.raises(TemporaryFailure, match="after the close"):
        source.fetch(OPEN, "regime", "btc_dominance")
    with pytest.raises(TemporaryFailure, match="after the close"):
        source.fetch(OPEN, "regime", "stablecoin_supply")
    with pytest.raises(EvaluationError, match="regime"):
        source.fetch(OPEN, "regime", "funding")
    with pytest.raises(TemporaryFailure, match="not collected"):
        source.fetch(OPEN, "funding", None)
    undated = Observation(
        series="btc_dominance",
        provider="coingecko",
        source_timestamp=None,
        observed_at=NOW,
        symbol=None,
        values=(("btc_dominance", Decimal("1")),),
        units=(("btc_dominance", "percent"),),
    )

    def parse(payload: object, *, observed_at: datetime) -> Observation:
        del payload, observed_at
        return undated

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(session_capture, "parse_btc_dominance", parse)
    try:
        kept = source.fetch(OPEN, "regime", "btc_dominance")
    finally:
        monkeypatch.undo()
    assert kept.observation.source_timestamp is None


def test_daily_bars_drop_a_later_day_and_retry_when_the_session_bar_is_missing() -> None:
    def reader(url: str, params: dict[str, str] | None) -> object:
        del url
        assert params is not None
        if params["interval"] == "1d":
            return [_daily_row(OPEN + timedelta(days=1))]
        raise AssertionError(params["interval"])

    source = EvidenceSource(_Spot(), AFTER, Decimal("0.02"), BASE, reader)
    with pytest.raises(TemporaryFailure, match="session bar is missing"):
        source.fetch(OPEN, "daily_bars", "BTCUSDT")


def test_hour_bars_keep_quote_volume_and_a_bad_payload_retries() -> None:
    candle = _hour(OPEN, 0, "1", quote="4")

    def reader(url: str, params: dict[str, str] | None) -> object:
        del url, params
        return [
            [candle.open_time, "1", "1", "1", "1", "1", candle.close_time, "4", 1],
        ]

    source = EvidenceSource(_Spot(), AFTER, Decimal("0.02"), BASE, reader)
    hours = source.fetch(OPEN, "hour_bars", "BTCUSDT")
    assert isinstance(hours, tuple)
    assert hours[0].quote_volume == Decimal("4")

    def broken(url: str, params: dict[str, str] | None) -> object:
        del url, params
        return {"no": "rows"}

    bad = EvidenceSource(_Spot(), AFTER, Decimal("0.02"), BASE, broken)
    with pytest.raises(TemporaryFailure):
        bad.fetch(OPEN, "hour_bars", "BTCUSDT")


def test_prefixes_skip_the_sealed_session_and_a_naive_clock() -> None:
    assert session_capture.evidence_prefixes(NOW) == (
        "captures/session=2026-10-07/",
        "sessions/date=2026-10-07/",
    )
    assert "2026-10-06" not in "".join(session_capture.evidence_prefixes(AFTER))
    assert "2026-10-07" in "".join(session_capture.evidence_prefixes(AFTER))
    assert session_capture.evidence_prefixes(datetime(2026, 10, 6, 12, tzinfo=UTC)) == ()
    assert session_capture.evidence_prefixes(datetime(2026, 10, 5, 12, tzinfo=UTC)) == ()
    with pytest.raises(EvaluationError, match="timezone-aware"):
        session_capture.evidence_prefixes(datetime(2026, 10, 7, 19))  # noqa: DTZ001
    with pytest.raises(MarketDataError, match="https"):
        EvidenceSource(_Spot(), NOW, Decimal("0.02"), "http://example", lambda _url, _params: {})


def test_symbols_default_to_bitcoin_and_reject_a_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAPTURE_SYMBOLS", raising=False)
    assert session_capture.capture_symbols() == ("BTCUSDT",)
    monkeypatch.setenv("CAPTURE_SYMBOLS", "BTCUSDT, ETHUSDT")
    assert session_capture.capture_symbols() == ("BTCUSDT", "ETHUSDT")
    monkeypatch.setenv("CAPTURE_SYMBOLS", "")
    assert session_capture.capture_symbols() == ()
    monkeypatch.setenv("CAPTURE_SYMBOLS", "../BTC")
    with pytest.raises(EvaluationError, match="unusable"):
        session_capture.capture_symbols()


def test_upload_refuses_a_conflict_and_a_sealed_key(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    changed = root / "captures" / "session=2026-10-07"
    changed.mkdir(parents=True)
    (changed / "universe.json").write_bytes(b"new")
    puts: list[str] = []

    class _Store:
        def put_object(self, **kwargs: object) -> None:
            puts.append(str(kwargs["Key"]))

    with pytest.raises(EvaluationError, match="conflicting"):
        session_capture._upload(
            _Store(),
            BUCKET,
            root,
            {"captures/session=2026-10-07/universe.json": b"old"},
        )
    assert puts == []
    sealed = root / "captures" / "session=2026-10-06"
    sealed.mkdir()
    (sealed / "universe.json").write_bytes(b"no")
    with pytest.raises(EvaluationError, match="sealed"):
        session_capture._upload(_Store(), BUCKET, root, {})
    assert puts == []


def test_download_skips_a_sealed_key_and_rejects_a_non_byte_body(tmp_path: Path) -> None:
    class _Pages:
        def __init__(self) -> None:
            self.pages = [
                {
                    "Contents": [
                        {"Key": "captures/session=2026-10-06/universe.json"},
                        {"Key": "captures/session=2026-10-07/universe.json"},
                    ],
                    "IsTruncated": True,
                    "NextContinuationToken": "next",
                },
                {"IsTruncated": False},
            ]

        def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
            if self.pages:
                return self.pages.pop(0)
            return {"IsTruncated": False}

        def get_object(self, **_kwargs: object) -> dict[str, object]:
            return {"Body": BytesIO(b"kept")}

    root = tmp_path / "down"
    root.mkdir()
    stored = session_capture._download(_Pages(), BUCKET, root, NOW)
    assert stored == {"captures/session=2026-10-07/universe.json": b"kept"}
    assert not (root / "captures" / "session=2026-10-06").exists()

    class _Text:
        def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
            return {"Contents": [{"Key": "captures/session=2026-10-07/ticker.json"}]}

        def get_object(self, **_kwargs: object) -> dict[str, object]:
            return {"Body": _TextBody()}

    class _TextBody:
        def read(self) -> str:
            return "text"

    with pytest.raises(EvaluationError, match="unusable"):
        session_capture._download(_Text(), BUCKET, root, NOW)


def test_download_reads_earlier_books_and_skips_a_later_session(tmp_path: Path) -> None:
    eth = "captures/session=2026-10-07/book/symbol=ETHUSDT.json"
    btc = "captures/session=2026-10-07/book/symbol=BTCUSDT.json"

    class _Books:
        def __init__(self) -> None:
            self.catalog = 0

        def list_objects_v2(self, **kwargs: object) -> dict[str, object]:
            prefix = str(kwargs.get("Prefix"))
            if kwargs.get("Delimiter") == "/":
                self.catalog += 1
                if self.catalog == 1:
                    return {
                        "CommonPrefixes": [
                            {"Prefix": "captures/session=2026-10-05/"},
                            {"Prefix": "captures/session=nope/"},
                            {"Prefix": "captures/session=2026-10-06/"},
                        ],
                        "IsTruncated": True,
                        "NextContinuationToken": "more",
                    }
                return {
                    "CommonPrefixes": [
                        {"Prefix": "captures/session=2026-10-07/"},
                        {"Prefix": "captures/session=2026-10-09/"},
                        {"Prefix": "captures/not-a-session/"},
                    ],
                    "IsTruncated": False,
                }
            if prefix == "captures/session=2026-10-08/":
                return {"Contents": [{"Key": "captures/session=2026-10-08/universe.json"}]}
            if prefix == "captures/session=2026-10-07/":
                return {"Contents": [{"Key": eth}]}
            if prefix == "captures/session=2026-10-07/book/":
                return {
                    "Contents": [{"Key": eth}, {"Key": btc}, {"Key": "other/symbol=BTCUSDT.json"}]
                }
            return {"Contents": []}

        def get_object(self, **kwargs: object) -> dict[str, object]:
            return {"Body": BytesIO(str(kwargs["Key"]).encode())}

    root = tmp_path / "books"
    root.mkdir()
    stored = session_capture._download(_Books(), BUCKET, root, AFTER)
    assert stored[eth] == eth.encode()
    assert stored[btc] == btc.encode()
    assert list(stored).count(eth) == 1
    assert "captures/session=2026-10-09/book/symbol=BTCUSDT.json" not in stored
    assert "captures/session=2026-10-06/book/symbol=BTCUSDT.json" not in stored
    assert not (root / "captures/session=2026-10-05").exists()


def test_a_stray_file_is_not_uploaded_and_storage_root_has_a_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "stray"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "local.txt").write_text("no")
    same = root / "captures" / "session=2026-10-07"
    same.mkdir(parents=True)
    (same / "universe.json").write_bytes(b"same")
    puts: list[str] = []

    class _Store:
        def put_object(self, **kwargs: object) -> None:
            puts.append(str(kwargs["Key"]))

    written, unchanged = session_capture._upload(
        _Store(), BUCKET, root, {"captures/session=2026-10-07/universe.json": b"same"}
    )
    assert written == ()
    assert unchanged == ("captures/session=2026-10-07/universe.json",)
    assert puts == []
    assert session_capture._clock().tzinfo is UTC
    monkeypatch.delenv("CAPTURE_ROOT", raising=False)
    assert session_capture._storage_root() == Path("/tmp/cip-session-capture")  # noqa: S108
    monkeypatch.setenv("CAPTURE_ROOT", str(tmp_path))
    assert session_capture._storage_root() == tmp_path
    session_capture._reset(tmp_path / "fresh")
    session_capture._reset(tmp_path / "fresh")
    assert (tmp_path / "fresh").is_dir()


def test_the_workload_is_capture_only() -> None:
    handler = (REPO / "src/cip/handlers/session_capture.py").read_text()
    terraform = (REPO / "terraform/modules/workload/capture.tf").read_text()
    for name in (
        "run_daily_scan",
        "prod_shadow_started_at",
        "prod_decisions_started_at",
        "score_v2_shadow_started_at",
        "BINANCE_API_KEY",
        "fapi.binance",
    ):
        assert name not in handler
        assert name not in terraform
    assert 'handler       = "cip.handlers.session_capture.capture"' in terraform
    assert 'schedule_expression = "rate(1 hour)"' in terraform
    assert "state               = var.capture_schedule_enabled" in terraform
    assert "CAPTURE_SYMBOLS" in terraform
    assert '"BTCUSDT"' in terraform
    assert "dynamodb" not in terraform
    assert "secretsmanager" not in terraform
    assert "ssm:" not in terraform
    assert "s3:DeleteObject" in terraform
    assert "run_daily_scan" not in (REPO / "terraform/environments/prod/main.tf").read_text()


def test_listing_open_is_the_earliest_daily_candle_inside_the_session() -> None:
    seen: dict[str, str] = {}

    def reader(_url: str, params: dict[str, str] | None) -> object:
        assert params is not None
        seen.update(params)
        return [[1_502_928_000_000]]

    source = EvidenceSource(_Spot(), NOW, Decimal("0.02"), BASE, reader)
    assert source.fetch(OPEN, "listing", "BTCUSDT") == [[1_502_928_000_000]]
    close_ms = int(session_close(OPEN).timestamp() * 1000)
    assert seen["startTime"] == "0"
    assert seen["endTime"] == str(close_ms - 1)
    assert seen["interval"] == "1d"
    assert seen["limit"] == "1"
    assert seen["symbol"] == "BTCUSDT"
    assert "GET /api/v3/klines BTCUSDT listing" in source.calls


def test_listing_after_the_close_does_not_call_the_provider() -> None:
    def reader(_url: str, _params: dict[str, str] | None) -> object:
        raise AssertionError("listing was requested")

    source = EvidenceSource(_Spot(), AFTER, Decimal("0.02"), BASE, reader)
    with pytest.raises(TemporaryFailure, match="after the close"):
        source.fetch(OPEN, "listing", "BTCUSDT")
