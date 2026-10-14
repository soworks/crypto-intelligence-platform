import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from cip.domain.errors import EvaluationError
from cip.domain.policy import load_policy
from cip.evaluation.classification import SymbolClassification, store_classification
from cip.evaluation.eligibility import Lane
from cip.evaluation.lane_fields import LaneFields, store_lane
from cip.evaluation.stage import load_symbol, needs_lane, settle

SESSION = date(2026, 10, 7)
NOW = datetime(2026, 10, 7, 19, tzinfo=UTC)
OPEN = datetime(2017, 8, 17, tzinfo=UTC)
UNIVERSE = load_policy(
    Path(__file__).resolve().parents[3] / "policies" / "investment-policy.yaml"
).policy.hypotheses.universe


def test_an_exclusion_policy_rejection_is_terminal() -> None:
    item = _item(fan_token=True)
    settled = settle(item, None, UNIVERSE)
    assert settled is not None
    assert settled.kind == "policy_rejected"
    assert "fan_token" in settled.reason_codes
    assert needs_lane(item, UNIVERSE) is False


def test_a_missing_classification_flag_stays_evidence_blocked() -> None:
    item = _item(eur_stable=None)
    settled = settle(item, None, UNIVERSE)
    assert settled is not None
    assert settled.kind == "evidence_blocked"
    assert settled.reason_codes == ("missing_eur_stable_classification",)


def test_a_cap_below_the_floor_does_not_need_a_lane_file() -> None:
    item = _item(cap=Decimal("1000"))
    settled = settle(item, None, UNIVERSE)
    assert settled is not None
    assert settled.kind == "policy_rejected"
    assert settled.reason_codes == ("below_market_cap",)
    assert needs_lane(item, UNIVERSE) is False


def test_a_missing_cap_stays_evidence_blocked() -> None:
    item = _item(cap=None)
    settled = settle(item, None, UNIVERSE)
    assert settled is not None
    assert settled.kind == "evidence_blocked"
    assert settled.reason_codes == ("missing_market_cap",)


def test_a_lane_passer_is_not_terminal_and_a_missing_lane_file_stays_open() -> None:
    item = _item(cap=Decimal("1000000000"))
    assert needs_lane(item, UNIVERSE) is True
    assert settle(item, None, UNIVERSE) is None
    assert settle(item, _lane(item), UNIVERSE) is None


def test_a_lane_policy_rejection_is_terminal_without_later_inputs() -> None:
    item = _item(cap=Decimal("1000000000"))
    lane = _lane(item, history_days=1)
    settled = settle(item, lane, UNIVERSE)
    assert settled is not None
    assert settled.kind == "policy_rejected"
    assert "history_below_minimum" in settled.reason_codes


def test_a_missing_unlock_schedule_stays_evidence_blocked() -> None:
    item = _item(symbol="SOLUSDT", base="SOL", cap=Decimal("100000000"))
    lane = _lane(item, lane=Lane.HIGH_RISK, rank=None, fdv=None)
    settled = settle(item, lane, UNIVERSE)
    assert settled is not None
    assert settled.kind == "evidence_blocked"
    assert settled.reason_codes == ("missing_unlock_schedule",)


def test_loading_requires_the_symbol_in_the_file(tmp_path: Path) -> None:
    assert load_symbol(tmp_path, SESSION, "BTCUSDT") is None
    store_classification(tmp_path, SESSION, _item(), NOW)
    loaded = load_symbol(tmp_path, SESSION, "BTCUSDT")
    assert loaded is not None
    assert loaded[1] is None
    path = tmp_path / "captures/session=2026-10-07/classification/symbol=BTCUSDT.json"
    document = json.loads(path.read_text())
    document["symbol"] = "ETHUSDT"
    path.write_text(json.dumps(document))
    with pytest.raises(EvaluationError, match="unusable"):
        load_symbol(tmp_path, SESSION, "BTCUSDT")
    document["symbol"] = "BTCUSDT"
    path.write_text(json.dumps(document))
    store_lane(tmp_path, SESSION, _lane(_item(), symbol="ETHUSDT"))
    moved = tmp_path / "captures/session=2026-10-07/lane/symbol=ETHUSDT.json"
    moved.rename(tmp_path / "captures/session=2026-10-07/lane/symbol=BTCUSDT.json")
    with pytest.raises(EvaluationError, match="unusable"):
        load_symbol(tmp_path, SESSION, "BTCUSDT")


def _item(
    *,
    symbol: str = "BTCUSDT",
    base: str = "BTC",
    cap: Decimal | None = Decimal("1000000000"),
    fan_token: bool | None = False,
    eur_stable: bool | None = False,
) -> SymbolClassification:
    return SymbolClassification(
        symbol=symbol,
        base_asset=base,
        eur_stable=eur_stable,
        fan_token=fan_token,
        monitoring_tag=False,
        delisting=False,
        deposits_suspended=False,
        withdrawals_suspended=False,
        pending_migration=False,
        coin_id="bitcoin",
        mapping="unambiguous",
        market_cap_usd=cap,
        market_cap_source_timestamp=None if cap is None else NOW,
        cmc_id=None,
        cmc_market_cap_usd=None,
        cmc_source_timestamp=None,
    )


def _lane(
    item: SymbolClassification,
    *,
    symbol: str | None = None,
    lane: Lane = Lane.NORMAL,
    history_days: int | None = None,
    rank: int | None = 1,
    fdv: Decimal | None = Decimal("1.2"),
) -> LaneFields:
    if history_days is None:
        history_open = OPEN
        days = (SESSION - OPEN.date()).days
    else:
        opened = SESSION - timedelta(days=history_days)
        history_open = datetime(opened.year, opened.month, opened.day, tzinfo=UTC)
        days = history_days
    return LaneFields(
        symbol=item.symbol if symbol is None else symbol,
        base_asset=item.base_asset,
        lane=lane,
        coin_id="bitcoin",
        circulating_ratio=Decimal("0.8"),
        history_days=days,
        history_open=history_open,
        market_cap_rank=rank,
        fdv_to_market_cap=fdv,
        unlock_state="unavailable",
        market_source_timestamp=NOW,
        market_captured_at=NOW,
        history_captured_at=NOW,
        unlock_provider=None,
        unlock_provider_id=None,
        unlock_source_timestamp=None,
    )
