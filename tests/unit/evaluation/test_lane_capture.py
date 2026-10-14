import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from cip.domain.errors import EvaluationError
from cip.domain.policy import load_policy
from cip.evaluation.classification import SymbolClassification, store_classification
from cip.evaluation.classification_bundle import store_component
from cip.evaluation.lane_capture import ListingUnavailable, capture_lanes
from cip.evaluation.lane_fields import history_days

SESSION = date(2026, 10, 7)
NOW = datetime(2026, 10, 7, 19, tzinfo=UTC)
STAMP = datetime(2026, 10, 7, 18, tzinfo=UTC)
OPEN_MS = 1_502_928_000_000
POLICY = Path(__file__).resolve().parents[3] / "policies" / "investment-policy.yaml"
UNIVERSE = load_policy(POLICY).policy.hypotheses.universe


def test_a_normal_survivor_stores_lane_evidence_and_a_packet(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    _markets(tmp_path)
    seen: list[str] = []

    def fetch(symbol: str) -> object:
        seen.append(symbol)
        return [[OPEN_MS, "1"]]

    assert capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, fetch) == ()
    assert seen == ["BTCUSDT"]
    lane = json.loads(_lane(tmp_path, "BTCUSDT").read_text())
    packet = json.loads(_packet(tmp_path, "BTCUSDT").read_text())
    assert lane["market_cap_rank"] == 4
    assert lane["circulating_ratio"] == "0.8"
    assert lane["fdv_to_market_cap"] == "1.25"
    assert lane["history_days"] == history_days(OPEN_MS, SESSION)
    assert lane["unlock_state"] == "unavailable"
    assert lane["unlock_provider"] is None
    assert lane["market_source_timestamp"] == "2026-10-07T18:00:00Z"
    assert packet["candidate"]["facts"]["history_days"] == lane["history_days"]
    assert packet["candidate"]["facts"]["unlock_schedule_known"] is None
    assert packet["source"] == "coingecko:markets"


def test_a_second_acquisition_does_not_fetch_or_rewrite(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    _markets(tmp_path)
    calls = {"count": 0}

    def fetch(_symbol: str) -> object:
        calls["count"] += 1
        return [[OPEN_MS, "1"]]

    capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, fetch)
    before = _lane(tmp_path, "BTCUSDT").read_bytes()
    packet = _packet(tmp_path, "BTCUSDT").read_bytes()
    later = NOW + timedelta(hours=1)
    assert capture_lanes(tmp_path, SESSION, later, ("BTCUSDT",), UNIVERSE, fetch) == ()
    assert calls["count"] == 1
    assert _lane(tmp_path, "BTCUSDT").read_bytes() == before
    assert _packet(tmp_path, "BTCUSDT").read_bytes() == packet


def test_a_missing_packet_is_restored_from_the_stored_lane(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    _markets(tmp_path)
    capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, lambda _symbol: [[OPEN_MS]])
    _packet(tmp_path, "BTCUSDT").unlink()
    assert capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden) == ()
    assert _packet(tmp_path, "BTCUSDT").is_file()


def test_high_risk_stays_blocked_without_an_unlock_source(tmp_path: Path) -> None:
    _store(tmp_path, _item("SOLUSDT", "SOL", Decimal("100000000"), coin_id="solana"))
    _markets(tmp_path, asset_id="solana", rank=400)
    capture_lanes(tmp_path, SESSION, NOW, ("SOLUSDT",), UNIVERSE, lambda _symbol: [[OPEN_MS]])
    lane = json.loads(_lane(tmp_path, "SOLUSDT").read_text())
    assert lane["lane"] == "high_risk"
    assert lane["market_cap_rank"] is None
    assert lane["fdv_to_market_cap"] is None
    assert lane["unlock_state"] == "unavailable"
    assert lane["unlock_source_timestamp"] is None
    assert not _packet(tmp_path, "SOLUSDT").exists()


def test_an_earlier_rejection_is_not_looked_up(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000")))
    _store(tmp_path, _item("BNBUSDT", "BNB", Decimal("1000000000"), fan_token=True))
    _markets(tmp_path)
    assert capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT", "BNBUSDT"), UNIVERSE, _forbidden) == ()
    assert not _lane(tmp_path, "BTCUSDT").exists()
    assert not _lane(tmp_path, "BNBUSDT").exists()


def test_a_missing_classification_waits_instead_of_fetching(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    assert capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT", "ETHUSDT"), UNIVERSE, _forbidden) == ()


def test_an_unavailable_listing_stays_unstored_and_the_next_symbol_continues(
    tmp_path: Path,
) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    _store(tmp_path, _item("ETHUSDT", "ETH", Decimal("1000000000"), coin_id="ethereum"))
    _markets(tmp_path)

    def fetch(symbol: str) -> object:
        if symbol == "BTCUSDT":
            raise ListingUnavailable("status 429")
        return [[OPEN_MS]]

    pending = capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT", "ETHUSDT"), UNIVERSE, fetch)
    assert pending == ("BTCUSDT",)
    assert not _lane(tmp_path, "BTCUSDT").exists()
    assert _lane(tmp_path, "ETHUSDT").is_file()


def test_an_empty_listing_stays_missing_history(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    _markets(tmp_path)
    assert capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, lambda _symbol: []) == ()
    lane = json.loads(_lane(tmp_path, "BTCUSDT").read_text())
    assert lane["history_days"] is None
    assert lane["history_open"] is None
    assert not _packet(tmp_path, "BTCUSDT").exists()
    assert capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden) == ()


def test_a_missing_market_row_stays_missing(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    pending = capture_lanes(
        tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, lambda _symbol: [[OPEN_MS]]
    )
    assert pending == ()
    lane = json.loads(_lane(tmp_path, "BTCUSDT").read_text())
    assert lane["circulating_ratio"] is None
    assert lane["market_source_timestamp"] is None
    assert not _packet(tmp_path, "BTCUSDT").exists()


def test_a_rank_rejection_stores_evidence_without_a_packet(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    _markets(tmp_path, rank=500)
    capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, lambda _symbol: [[OPEN_MS]])
    assert _lane(tmp_path, "BTCUSDT").is_file()
    assert not _packet(tmp_path, "BTCUSDT").exists()


def test_a_conflicting_lane_file_fails_closed(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    _markets(tmp_path)

    def fetch(_symbol: str) -> object:
        _lane(tmp_path, "BTCUSDT").parent.mkdir(parents=True, exist_ok=True)
        _lane(tmp_path, "BTCUSDT").write_bytes(b'{"symbol":"OTHER"}')
        return [[OPEN_MS]]

    with pytest.raises(EvaluationError, match="different payload"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, fetch)


def test_a_market_timestamp_after_the_close_is_not_stored(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    _markets(tmp_path, updated="2026-10-08T00:00:01Z")
    with pytest.raises(EvaluationError, match="after the close"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)
    assert not _lane(tmp_path, "BTCUSDT").exists()


def test_a_retrieval_after_the_close_does_not_fill_the_session(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    later = datetime(2026, 10, 8, 1, tzinfo=UTC)
    with pytest.raises(EvaluationError, match="after the close"):
        capture_lanes(tmp_path, SESSION, later, ("BTCUSDT",), UNIVERSE, _forbidden)


def test_a_malformed_listing_does_not_block_the_next_symbol(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    _store(tmp_path, _item("ETHUSDT", "ETH", Decimal("1000000000"), coin_id="ethereum"))
    _markets(tmp_path)

    def fetch(symbol: str) -> object:
        if symbol == "BTCUSDT":
            return [["not-a-candle"]]
        return [[OPEN_MS]]

    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT", "ETHUSDT"), UNIVERSE, fetch)
    assert not _lane(tmp_path, "BTCUSDT").exists()
    assert _lane(tmp_path, "ETHUSDT").is_file()


def test_a_listing_after_the_session_is_refused(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    _markets(tmp_path)
    future = int(datetime(2026, 10, 8, tzinfo=UTC).timestamp() * 1000)
    with pytest.raises(EvaluationError, match="history starts after the session"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, lambda _symbol: [[future]])
    assert not _lane(tmp_path, "BTCUSDT").exists()


def test_a_sealed_or_finalized_session_is_not_fetched(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="sealed session stays sealed"):
        capture_lanes(tmp_path, date(2026, 10, 6), NOW, ("BTCUSDT",), UNIVERSE, _forbidden)
    manifest = tmp_path / "sessions" / "date=2026-10-07" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    with pytest.raises(EvaluationError, match="finalized session is sealed"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)


def test_a_naive_clock_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="timezone-aware UTC"):
        capture_lanes(
            tmp_path,
            SESSION,
            datetime(2026, 10, 7, 19),  # noqa: DTZ001
            ("BTCUSDT",),
            UNIVERSE,
            _forbidden,
        )


def test_an_unusable_classification_or_market_catalog_fails_closed(tmp_path: Path) -> None:
    path = _classification_path(tmp_path, "BTCUSDT")
    path.parent.mkdir(parents=True)
    path.write_text("[]")
    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)
    path.write_text(json.dumps({"mapping": "sideways"}))
    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)
    path.unlink()
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000"), coin_id=None))
    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, lambda _symbol: [[OPEN_MS]])
    markets = _markets_path(tmp_path)
    markets.parent.mkdir(parents=True, exist_ok=True)
    markets.write_text("[]")
    path.unlink()
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)
    markets.write_text(json.dumps({"captured_at": None, "payload": []}))
    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)
    markets.write_text(json.dumps({"captured_at": "2026-10-08T01:00:00Z", "payload": []}))
    with pytest.raises(EvaluationError, match="after the close"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)


def test_a_second_malformed_listing_keeps_the_first_error(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    _store(tmp_path, _item("ETHUSDT", "ETH", Decimal("1000000000"), coin_id="ethereum"))
    _markets(tmp_path)
    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(
            tmp_path,
            SESSION,
            NOW,
            ("BTCUSDT", "ETHUSDT"),
            UNIVERSE,
            lambda _symbol: [["bad"]],
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("unlock_state", "missing"),
        ("symbol", ""),
        ("history_days", -1),
        ("history_days", True),
        ("history_captured_at", 1),
        ("history_captured_at", "yesterday"),
        ("history_captured_at", "2026-10-07T19:00:00"),
        ("history_captured_at", "2026-10-07T19:00:00+01:00"),
        ("unlock_provider", ""),
    ],
)
def test_a_reused_lane_document_must_parse(tmp_path: Path, key: str, value: object) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    _markets(tmp_path)
    capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, lambda _symbol: [[OPEN_MS]])
    document = json.loads(_lane(tmp_path, "BTCUSDT").read_text())
    document[key] = value
    _lane(tmp_path, "BTCUSDT").write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)


def test_a_classification_document_must_parse(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    path = _classification_path(tmp_path, "BTCUSDT")
    document = json.loads(path.read_text())
    document["eur_stable"] = "yes"
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)
    document["eur_stable"] = None
    document["market_cap_usd"] = 1
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)
    document["market_cap_usd"] = "nope"
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)
    document["market_cap_usd"] = "Infinity"
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)


def test_a_corrupt_lane_file_is_not_reused(tmp_path: Path) -> None:
    _store(tmp_path, _item("BTCUSDT", "BTC", Decimal("1000000000")))
    _markets(tmp_path)
    path = _lane(tmp_path, "BTCUSDT")
    path.parent.mkdir(parents=True)
    path.write_text("[]")
    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)
    path.write_text(json.dumps({"lane": "sideways", "unlock_state": "unavailable"}))
    with pytest.raises(EvaluationError, match="unusable"):
        capture_lanes(tmp_path, SESSION, NOW, ("BTCUSDT",), UNIVERSE, _forbidden)


def _forbidden(_symbol: str) -> object:
    raise AssertionError("listing was requested")


def _item(
    symbol: str,
    base: str,
    cap: Decimal | None,
    *,
    fan_token: bool = False,
    coin_id: str | None = "bitcoin",
) -> SymbolClassification:
    return SymbolClassification(
        symbol=symbol,
        base_asset=base,
        eur_stable=False,
        fan_token=fan_token,
        monitoring_tag=False,
        delisting=False,
        deposits_suspended=False,
        withdrawals_suspended=False,
        pending_migration=False,
        coin_id=coin_id,
        mapping="unambiguous" if coin_id is not None else "unmapped",
        market_cap_usd=cap,
        market_cap_source_timestamp=None if cap is None else STAMP,
        cmc_id=None,
        cmc_market_cap_usd=None,
        cmc_source_timestamp=None,
    )


def _store(root: Path, item: SymbolClassification) -> None:
    store_classification(root, SESSION, item, NOW)


def _markets(
    root: Path,
    *,
    asset_id: str = "bitcoin",
    rank: int = 4,
    updated: str = "2026-10-07T18:00:00Z",
) -> None:
    store_component(
        root,
        SESSION,
        "markets",
        [
            {
                "circulating_supply": "80",
                "fully_diluted_valuation": "500",
                "id": asset_id,
                "last_updated": updated,
                "market_cap": "400",
                "market_cap_rank": rank,
                "total_supply": "100",
            }
        ],
        NOW,
    )


def _lane(root: Path, symbol: str) -> Path:
    return root / "captures" / f"session={SESSION.isoformat()}" / "lane" / f"symbol={symbol}.json"


def _packet(root: Path, symbol: str) -> Path:
    folder = root / "captures" / f"session={SESSION.isoformat()}" / "candidates"
    return folder / f"symbol={symbol}.json"


def _classification_path(root: Path, symbol: str) -> Path:
    return (
        root
        / "captures"
        / f"session={SESSION.isoformat()}"
        / "classification"
        / f"symbol={symbol}.json"
    )


def _markets_path(root: Path) -> Path:
    return (
        root
        / "captures"
        / f"session={SESSION.isoformat()}"
        / "classification_sources"
        / "markets.json"
    )
