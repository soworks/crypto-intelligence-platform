import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from cip.domain.errors import EvaluationError, RecorderError
from cip.evaluation import classification_bundle
from cip.evaluation.classification_bundle import ensure_bundle, load_component, store_component
from cip.evaluation.classification_producer import (
    CatalogIncomplete,
    classify_stored,
    fetch_component,
    fetch_page,
)

OPEN = date(2026, 10, 8)
NOW = datetime(2026, 10, 8, 16, 24, tzinfo=UTC)
PAIRS = (("BTCUSDT", "BTC"), ("ETHUSDT", "ETH"))
_TICKER = {
    "base": "BTC",
    "coin_id": "bitcoin",
    "market": {"identifier": "binance"},
    "target": "USDT",
}


def _payload(url: str) -> object:
    if url.endswith("get-all-asset"):
        return {"data": [{"assetCode": "BTC", "tags": []}], "success": True}
    if url.endswith("getNetworkCoinAll"):
        return {"data": [{"coin": "BTC"}], "success": True}
    if "tickers?page=" in url:
        return {"tickers": [_TICKER]}
    if "eur-stablecoin" in url:
        return [{"id": "eurite"}]
    if url.endswith("symbol/list"):
        return {"data": [{"cmcUniqueId": 1, "symbol": "BTCUSDT"}]}
    if "ids=" in url:
        return [{"id": "bitcoin", "last_updated": "2026-10-08T16:00:00Z", "market_cap": 2}]
    raise AssertionError(url)


def _bundle(root: Path, blocked: set[str] | None = None) -> list[str]:
    seen: list[str] = []
    blocked = set() if blocked is None else blocked

    def get(url: str) -> object:
        seen.append(url)
        if any(part in url for part in blocked):
            raise RecorderError("status 429")
        return _payload(url)

    ensure_bundle(root, OPEN, NOW, PAIRS, get)
    return seen


def test_a_later_throttle_keeps_the_ticker_catalog_and_publishes_nothing(tmp_path: Path) -> None:
    seen = _bundle(tmp_path, {"eur-stablecoin"})
    assert any("tickers?page=1" in url for url in seen)
    assert (tmp_path / "captures/session=2026-10-08/classification_sources/tickers.json").is_file()
    assert not (tmp_path / "captures/session=2026-10-08/classification").exists()


def test_the_next_invocation_does_not_download_the_finished_ticker_catalog(tmp_path: Path) -> None:
    _bundle(tmp_path, {"eur-stablecoin"})
    seen = _bundle(tmp_path, set())
    assert not any("tickers" in url for url in seen)
    assert any("eur-stablecoin" in url for url in seen)
    names = sorted(
        path.name
        for path in (tmp_path / "captures/session=2026-10-08/classification").glob("*.json")
    )
    assert names == []


def test_invocations_complete_independent_sources_then_publish_once(tmp_path: Path) -> None:
    _bundle(tmp_path, {"eur-stablecoin"})
    _bundle(tmp_path, {"ids="})
    calls: list[str] = []

    def get(url: str) -> object:
        calls.append(url)
        return _payload(url)

    published = ensure_bundle(tmp_path, OPEN, NOW, PAIRS, get)
    assert published is not None
    assert [item.symbol for item in published] == ["BTCUSDT", "ETHUSDT"]
    assert not any("tickers" in url or "eur-stablecoin" in url for url in calls)
    assert any("ids=" in url for url in calls)
    again = ensure_bundle(tmp_path, OPEN, NOW, PAIRS, get)
    assert again is not None
    assert [item.symbol for item in again] == ["BTCUSDT", "ETHUSDT"]
    assert calls == [url for url in calls if "ids=" in url]


def test_a_second_complete_read_makes_no_provider_call(tmp_path: Path) -> None:
    assert ensure_bundle(tmp_path, OPEN, NOW, PAIRS, _payload) is not None

    def get(url: str) -> object:
        raise AssertionError(url)

    again = ensure_bundle(tmp_path, OPEN, NOW, PAIRS, get)
    assert again is not None
    assert len(again) == 2


def test_a_conflicting_source_fails_closed_and_the_first_body_remains(tmp_path: Path) -> None:
    store_component(tmp_path, OPEN, "eur", [{"id": "eurite"}], NOW)
    path = tmp_path / "captures/session=2026-10-08/classification_sources/eur.json"
    before = path.read_bytes()
    with pytest.raises(EvaluationError, match="conflicting observation"):
        store_component(tmp_path, OPEN, "eur", [{"id": "other"}], NOW)
    assert path.read_bytes() == before
    store_component(tmp_path, OPEN, "eur", [{"id": "eurite"}], NOW)


def test_a_source_outside_the_session_is_rejected(tmp_path: Path) -> None:
    after = datetime(2026, 10, 9, 0, 0, tzinfo=UTC)
    before = datetime(2026, 10, 7, 23, tzinfo=UTC)
    with pytest.raises(EvaluationError, match="outside the session"):
        store_component(tmp_path, OPEN, "eur", [{"id": "eurite"}], after)
    with pytest.raises(EvaluationError, match="outside the session"):
        store_component(tmp_path, OPEN, "eur", [{"id": "eurite"}], before)
    store_component(tmp_path, OPEN, "eur", [{"id": "eurite"}], NOW)
    document = json.loads(
        (tmp_path / "captures/session=2026-10-08/classification_sources/eur.json").read_text()
    )
    document["captured_at"] = "2026-10-09T00:00:00Z"
    path = tmp_path / "captures/session=2026-10-08/classification_sources/eur.json"
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="outside the session"):
        load_component(tmp_path, OPEN, "eur")


def test_a_finalized_or_sealed_session_cannot_gain_a_source(tmp_path: Path) -> None:
    manifest = tmp_path / "sessions/date=2026-10-08/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    with pytest.raises(EvaluationError, match="finalized session is sealed"):
        store_component(tmp_path, OPEN, "eur", [{"id": "eurite"}], NOW)
    with pytest.raises(EvaluationError, match="sealed session stays sealed"):
        store_component(
            tmp_path, date(2026, 10, 6), "eur", [], datetime(2026, 10, 6, 1, tzinfo=UTC)
        )


def test_a_damaged_source_file_fails_closed(tmp_path: Path) -> None:
    store_component(tmp_path, OPEN, "eur", [{"id": "eurite"}], NOW)
    path = tmp_path / "captures/session=2026-10-08/classification_sources/eur.json"
    document = json.loads(path.read_text())
    document["content_sha256"] = "0" * 64
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="unusable"):
        load_component(tmp_path, OPEN, "eur")
    path.write_text(json.dumps(["nope"]))
    with pytest.raises(EvaluationError, match="unusable"):
        load_component(tmp_path, OPEN, "eur")
    path.write_text(
        json.dumps(
            {
                "captured_at": "yesterday",
                "complete": True,
                "component": "eur",
                "content_sha256": "x",
                "payload": [],
                "session": OPEN.isoformat(),
            }
        )
    )
    with pytest.raises(EvaluationError, match="unusable"):
        load_component(tmp_path, OPEN, "eur")
    document = {
        "captured_at": "2026-10-08T16:24:00+00:00",
        "complete": False,
        "component": "eur",
        "content_sha256": "abc",
        "payload": [],
        "session": "2026-10-07",
    }
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="outside the session"):
        load_component(tmp_path, OPEN, "eur")


def test_an_unknown_source_and_a_short_market_catalog_fail(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="unknown"):
        fetch_component("funding", _payload, PAIRS, {})
    with pytest.raises(CatalogIncomplete):
        fetch_component("markets", _payload, PAIRS, {})
    naive = NOW.replace(tzinfo=None)
    with pytest.raises(EvaluationError, match="timezone-aware"):
        store_component(tmp_path, OPEN, "eur", [], naive)
    later = NOW + timedelta(minutes=1)
    store_component(tmp_path, OPEN, "assets", {"data": []}, NOW)
    with pytest.raises(EvaluationError, match="conflicting observation"):
        store_component(tmp_path, OPEN, "assets", {"data": [1]}, later)
    with pytest.raises(EvaluationError, match="unknown"):
        store_component(tmp_path, OPEN, "funding", [], NOW)
    with pytest.raises(EvaluationError, match="unusable"):
        store_component(tmp_path, OPEN, "eur", "raw", NOW)
    stored = store_component(tmp_path, OPEN, "coins", {"data": [Decimal("1.5")]}, NOW)
    assert stored == {"data": ["1.5"]}
    with pytest.raises(TypeError):
        store_component(tmp_path, OPEN, "marketing", {"data": [object()]}, NOW)
    store_component(tmp_path, OPEN, "eur", [{"id": "eurite"}], NOW)

    path = tmp_path / "captures/session=2026-10-08/classification_sources/eur.json"
    document = json.loads(path.read_text())
    document["complete"] = False
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="incomplete"):
        load_component(tmp_path, OPEN, "eur")
    document["complete"] = True
    document["captured_at"] = 1
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="unusable"):
        load_component(tmp_path, OPEN, "eur")
    document["captured_at"] = "not-a-time"
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="unusable"):
        load_component(tmp_path, OPEN, "eur")
    document["captured_at"] = "2026-10-08T16:24:00"
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="timezone-aware"):
        load_component(tmp_path, OPEN, "eur")

    def markets(url: str) -> object:
        if "ids=" in url:
            return {"id": "bitcoin"}
        return _payload(url)

    with pytest.raises(CatalogIncomplete):
        fetch_component("markets", markets, PAIRS, {"tickers": [_TICKER]})
    with pytest.raises(EvaluationError, match="unusable"):
        fetch_component("marketing", lambda _url: {"data": {}}, PAIRS, {})
    with pytest.raises(EvaluationError, match="incomplete"):
        classify_stored(PAIRS, {})
    with pytest.raises(EvaluationError, match="incomplete"):
        classify_stored(PAIRS, {"tickers": [], "eur": [], "markets": {}})
    with pytest.raises(EvaluationError, match="unusable"):
        classify_stored(PAIRS, {"tickers": [], "eur": [], "markets": [], "marketing": {}})
    with pytest.raises(CatalogIncomplete):
        fetch_component(
            "markets",
            lambda _url: [{"id": "other", "last_updated": "2026-10-08T16:00:00Z", "market_cap": 1}],
            PAIRS,
            {"tickers": [_TICKER]},
        )
    path = tmp_path / "captures/session=2026-10-08/classification_sources/coins.json"
    document = json.loads(path.read_text())
    document["provenance"] = "other"
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="unusable"):
        load_component(tmp_path, OPEN, "coins")


def test_a_racing_writer_cannot_replace_a_finished_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def occupy(source: str, destination: str) -> None:
        del source
        Path(destination).write_bytes(b"other")
        raise FileExistsError

    monkeypatch.setattr(classification_bundle.os, "link", occupy)
    with pytest.raises(EvaluationError, match="conflicting observation"):
        store_component(tmp_path, OPEN, "eur", [{"id": "eurite"}], NOW)

    def same(source: str, destination: str) -> None:
        Path(destination).write_bytes(Path(source).read_bytes())
        raise FileExistsError

    monkeypatch.setattr(classification_bundle.os, "link", same)
    stored = store_component(tmp_path, OPEN, "assets", {"data": []}, NOW)
    assert stored == {"data": []}


def _ticker_rows(count: int, page: int) -> list[dict[str, object]]:
    return [{**_TICKER, "volume": f"{page}-{index}"} for index in range(count)]


def _ticker_page(url: str) -> int:
    return int(url.rsplit("=", 1)[-1])


def _sources(root: Path) -> Path:
    return root / "captures/session=2026-10-08/classification_sources"


def test_pages_survive_a_throttle_and_the_next_invocation_resumes(tmp_path: Path) -> None:
    first: list[int] = []

    def opening(url: str) -> object:
        if "tickers?page=" in url:
            page = _ticker_page(url)
            first.append(page)
            if page == 6:
                raise RecorderError("status 429")
            return {"tickers": _ticker_rows(100, page)}
        raise AssertionError(url)

    assert ensure_bundle(tmp_path, OPEN, NOW, PAIRS, opening) is None
    assert first == [1, 2, 3, 4, 5, 6]
    folder = _sources(tmp_path) / "tickers"
    assert sorted(path.name for path in folder.glob("page=*.json")) == [
        "page=1.json",
        "page=2.json",
        "page=3.json",
        "page=4.json",
        "page=5.json",
    ]
    assert not (_sources(tmp_path) / "tickers.json").exists()
    assert not (tmp_path / "captures/session=2026-10-08/classification").exists()

    second: list[int] = []

    def resumed(url: str) -> object:
        if "tickers?page=" in url:
            page = _ticker_page(url)
            second.append(page)
            if page == 6:
                raise RecorderError("status 429")
            return {"tickers": _ticker_rows(100, page)}
        raise AssertionError(url)

    later = NOW + timedelta(hours=1)
    kept = (folder / "page=5.json").read_bytes()
    assert ensure_bundle(tmp_path, OPEN, later, PAIRS, resumed) is None
    assert second == [6]
    assert (folder / "page=5.json").read_bytes() == kept


def test_hourly_invocations_complete_one_ticker_source_then_publish(tmp_path: Path) -> None:
    hold: list[object] = [6]

    def get(url: str) -> object:
        if "tickers?page=" in url:
            page = _ticker_page(url)
            if page == hold[0]:
                raise RecorderError("status 429")
            if page < 14:
                return {"tickers": _ticker_rows(100, page)}
            return {"tickers": [_TICKER]}
        if hold[0] == "assets" and url.endswith("get-all-asset"):
            raise RecorderError("status 429")
        return _payload(url)

    assert ensure_bundle(tmp_path, OPEN, NOW, PAIRS, get) is None
    hold[0] = 11
    assert ensure_bundle(tmp_path, OPEN, NOW + timedelta(hours=1), PAIRS, get) is None
    hold[0] = "assets"
    assert ensure_bundle(tmp_path, OPEN, NOW + timedelta(hours=2), PAIRS, get) is None
    manifest = _sources(tmp_path) / "tickers.json"
    document = json.loads(manifest.read_text())
    assert document["session"] == OPEN.isoformat()
    assert document["provenance"] == "coingecko:binance-tickers"
    assert [part["page"] for part in document["pages"]] == list(range(1, 15))
    assert document["query"] == {"page_size": 100}
    assert document["retrieval_earliest"] < document["retrieval_latest"]
    assert "payload" not in document
    assert len({part["content_sha256"] for part in document["pages"]}) == 14
    before = manifest.read_bytes()
    seen: list[str] = []

    def finish(url: str) -> object:
        seen.append(url)
        return _payload(url)

    published = ensure_bundle(tmp_path, OPEN, NOW + timedelta(hours=3), PAIRS, finish)
    assert published is not None
    assert [item.symbol for item in published] == ["BTCUSDT", "ETHUSDT"]
    assert not any("tickers?page=" in url for url in seen)
    assert any(url.endswith("get-all-asset") for url in seen)
    assert any("eur-stablecoin" in url for url in seen)
    assert any("ids=" in url for url in seen)
    assert manifest.read_bytes() == before
    with pytest.raises(EvaluationError, match="conflicting observation"):
        classification_bundle._complete_pages(
            tmp_path, OPEN, "tickers", 14, NOW + timedelta(hours=4)
        )
    assert manifest.read_bytes() == before


def test_a_page_conflict_fails_closed_and_the_same_page_is_idempotent(tmp_path: Path) -> None:
    rows = _ticker_rows(1, 1)
    classification_bundle._store_page(tmp_path, OPEN, "tickers", 1, rows, NOW)
    path = _sources(tmp_path) / "tickers/page=1.json"
    before = path.read_bytes()
    classification_bundle._store_page(tmp_path, OPEN, "tickers", 1, rows, NOW)
    assert path.read_bytes() == before
    with pytest.raises(EvaluationError, match="conflicting observation"):
        classification_bundle._store_page(tmp_path, OPEN, "tickers", 1, _ticker_rows(1, 2), NOW)
    assert path.read_bytes() == before


def test_a_malformed_or_full_terminal_page_does_not_complete_the_source(tmp_path: Path) -> None:
    def malformed(url: str) -> object:
        if "tickers?page=" not in url:
            raise AssertionError(url)
        page = _ticker_page(url)
        if page < 3:
            return {"tickers": _ticker_rows(100, page)}
        return {"tickers": "nope"}

    assert ensure_bundle(tmp_path, OPEN, NOW, PAIRS, malformed) is None
    assert (_sources(tmp_path) / "tickers/page=2.json").is_file()
    assert not (_sources(tmp_path) / "tickers/page=3.json").exists()
    assert not (_sources(tmp_path) / "tickers.json").exists()

    fresh = tmp_path / "over"

    def oversized(url: str) -> object:
        if "tickers?page=" in url:
            return {"tickers": _ticker_rows(101, 1)}
        raise AssertionError(url)

    assert ensure_bundle(fresh, OPEN, NOW, PAIRS, oversized) is None
    assert not (_sources(fresh) / "tickers").exists()

    short = tmp_path / "short-prefix"
    classification_bundle._store_page(short, OPEN, "tickers", 1, _ticker_rows(50, 1), NOW)
    classification_bundle._store_page(short, OPEN, "tickers", 2, _ticker_rows(1, 2), NOW)
    with pytest.raises(EvaluationError, match="incomplete"):
        classification_bundle._complete_pages(short, OPEN, "tickers", 2, NOW)
    assert not (_sources(short) / "tickers.json").exists()
    full = tmp_path / "full-terminal"
    classification_bundle._store_page(full, OPEN, "tickers", 1, _ticker_rows(100, 1), NOW)
    with pytest.raises(EvaluationError, match="incomplete"):
        classification_bundle._complete_pages(full, OPEN, "tickers", 1, NOW)
    assert not (_sources(full) / "tickers.json").exists()


def test_a_fragment_outside_the_session_cannot_rebuild_the_catalog(tmp_path: Path) -> None:
    payload = classification_bundle._freeze([_TICKER])
    body = {
        "captured_at": "2026-10-07T23:00:00Z",
        "component": "tickers",
        "content_sha256": classification_bundle._hash(payload),
        "page": 1,
        "page_size": 100,
        "payload": payload,
        "provenance": "coingecko:binance-tickers",
        "query": {"page": 1},
        "session": OPEN.isoformat(),
    }
    path = classification_bundle._page_path(tmp_path, OPEN, "tickers", 1)
    path.parent.mkdir(parents=True)
    path.write_bytes(classification_bundle._json(body))
    with pytest.raises(EvaluationError, match="outside the session"):
        ensure_bundle(tmp_path, OPEN, NOW, PAIRS, _payload)


def test_page_acquisition_rejects_an_unknown_page_and_stops_at_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(EvaluationError, match="unknown"):
        fetch_page("assets", _payload, 1)
    with pytest.raises(EvaluationError, match="unknown"):
        fetch_page("tickers", _payload, 0)

    def bare_list(url: str) -> object:
        del url
        return _ticker_rows(1, 1)

    rows, size = fetch_page("tickers", bare_list, 1)
    assert size == 100
    assert rows == _ticker_rows(1, 1)
    monkeypatch.setattr(classification_bundle, "_MAX_PAGES", 1)

    def full(url: str) -> object:
        if "tickers?page=" in url:
            return {"tickers": _ticker_rows(100, 1)}
        raise AssertionError(url)

    assert ensure_bundle(tmp_path, OPEN, NOW, PAIRS, full) is None
    assert not (_sources(tmp_path) / "tickers.json").exists()
    assert (_sources(tmp_path) / "tickers/page=1.json").is_file()


def _write_page(root: Path, document: dict[str, object]) -> None:
    path = classification_bundle._page_path(root, OPEN, "tickers", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def test_a_damaged_page_or_manifest_fails_closed(tmp_path: Path) -> None:
    classification_bundle._store_page(tmp_path, OPEN, "tickers", 1, _ticker_rows(101, 1), NOW)
    with pytest.raises(EvaluationError, match="unusable"):
        ensure_bundle(tmp_path, OPEN, NOW, PAIRS, _payload)
    path = classification_bundle._page_path(tmp_path, OPEN, "tickers", 1)
    document = json.loads(path.read_text())
    document["payload"] = {"row": 1}
    document["content_sha256"] = classification_bundle._hash(document["payload"])
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="unusable"):
        classification_bundle._load_page(tmp_path, OPEN, "tickers", 1)
    with pytest.raises(EvaluationError, match="unusable"):
        classification_bundle._complete_pages(tmp_path, OPEN, "tickers", 1, NOW)
    path.write_text("[]")
    with pytest.raises(EvaluationError, match="unusable"):
        classification_bundle._load_page(tmp_path, OPEN, "tickers", 1)

    valid = {
        "captured_at": "2026-10-08T16:24:00Z",
        "component": "tickers",
        "content_sha256": classification_bundle._hash([_TICKER]),
        "page": 1,
        "payload": [_TICKER],
        "provenance": "coingecko:binance-tickers",
        "session": OPEN.isoformat(),
    }
    _write_page(tmp_path, {**valid, "session": "2026-10-07"})
    with pytest.raises(EvaluationError, match="outside the session"):
        classification_bundle._load_page(tmp_path, OPEN, "tickers", 1)
    _write_page(tmp_path, {**valid, "component": "eur"})
    with pytest.raises(EvaluationError, match="outside the session"):
        classification_bundle._load_page(tmp_path, OPEN, "tickers", 1)
    _write_page(tmp_path, {**valid, "page": 2})
    with pytest.raises(EvaluationError, match="unusable"):
        classification_bundle._load_page(tmp_path, OPEN, "tickers", 1)
    _write_page(tmp_path, {**valid, "provenance": "other"})
    with pytest.raises(EvaluationError, match="unusable"):
        classification_bundle._load_page(tmp_path, OPEN, "tickers", 1)
    _write_page(tmp_path, {**valid, "content_sha256": 1})
    with pytest.raises(EvaluationError, match="unusable"):
        classification_bundle._load_page(tmp_path, OPEN, "tickers", 1)
    _write_page(tmp_path, {**valid, "content_sha256": "0" * 64})
    with pytest.raises(EvaluationError, match="unusable"):
        classification_bundle._load_page(tmp_path, OPEN, "tickers", 1)
    path.unlink()
    assert ensure_bundle(tmp_path, OPEN, NOW, PAIRS, _payload) is not None
    manifest_path = _sources(tmp_path) / "tickers.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["session"] = "2026-10-07"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationError, match="outside the session"):
        load_component(tmp_path, OPEN, "tickers")
    manifest["session"] = OPEN.isoformat()
    manifest["component"] = "eur"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationError, match="outside the session"):
        load_component(tmp_path, OPEN, "tickers")
    manifest["component"] = "tickers"
    manifest["provenance"] = "other"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationError, match="unusable"):
        load_component(tmp_path, OPEN, "tickers")
    manifest["provenance"] = "coingecko:binance-tickers"
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationError, match="unusable"):
        load_component(tmp_path, OPEN, "tickers")
    manifest["complete"] = True
    manifest["pages"] = []
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationError, match="incomplete"):
        load_component(tmp_path, OPEN, "tickers")
    manifest["pages"] = ["nope"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationError, match="incomplete"):
        load_component(tmp_path, OPEN, "tickers")
    manifest["pages"] = [{"page": 9}]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationError, match="incomplete"):
        load_component(tmp_path, OPEN, "tickers")
    page = classification_bundle._page_path(tmp_path, OPEN, "tickers", 1)
    page.unlink()
    manifest["pages"] = [{"content_sha256": "abc", "page": 1}]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationError, match="unusable"):
        load_component(tmp_path, OPEN, "tickers")
    classification_bundle._store_page(tmp_path, OPEN, "tickers", 1, [_TICKER], NOW)
    with pytest.raises(EvaluationError, match="unusable"):
        load_component(tmp_path, OPEN, "tickers")
    stored = json.loads(page.read_text())
    manifest["pages"] = [{"content_sha256": stored["content_sha256"], "page": 1}]
    manifest["aggregate_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationError, match="unusable"):
        load_component(tmp_path, OPEN, "tickers")
    with pytest.raises(EvaluationError, match="incomplete"):
        classification_bundle._rows_from_manifest(
            tmp_path, OPEN, "tickers", {**manifest, "pages": "nope"}
        )
