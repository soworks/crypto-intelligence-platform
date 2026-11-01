from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from cip.backtest.contracts import Eligibility
from cip.domain.errors import EvaluationError
from cip.domain.policy import load_policy
from cip.evaluation.decision import (
    Cohort,
    DecisionRecord,
    Disposition,
    SourceStamp,
)
from cip.evaluation.eligibility import (
    CandidateFacts,
    EligibilityDecision,
    Lane,
    PolicyEligibility,
    assess,
)

_POLICY = Path(__file__).parents[3] / "policies" / "investment-policy.yaml"
UNIVERSE = load_policy(_POLICY).policy.hypotheses.universe
DAY = date(2026, 10, 4)


def _money(value: float) -> Decimal:
    return Decimal(str(value))


def _normal(**overrides: object) -> CandidateFacts:
    lane = UNIVERSE.normal
    values: dict[str, object] = {
        "symbol": "SOLUSDT",
        "base_asset": "SOL",
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


def _high(**overrides: object) -> CandidateFacts:
    lane = UNIVERSE.high_risk
    values: dict[str, object] = {
        "market_cap_usd": _money(UNIVERSE.normal.minimum_market_cap_usd) - 1,
        "market_cap_rank": None,
        "circulating_ratio": _money(lane.minimum_circulating_ratio),
        "fdv_to_market_cap": None,
        "history_days": lane.minimum_history_days,
        "unlock_schedule_known": True,
    }
    values.update(overrides)
    return _normal(**values)


def test_the_normal_lane_floor_is_eligible_and_not_an_order() -> None:
    floor = format(_money(UNIVERSE.normal.minimum_market_cap_usd), "f")
    result = assess(_normal(unlock_schedule_known=False, market_cap_usd=floor), UNIVERSE)
    assert result.eligible is True
    assert result.lane is Lane.NORMAL
    assert result.reason_codes == ("normal_lane",)
    assert "disposition" not in EligibilityDecision.model_fields


def test_the_boundary_below_the_normal_floor_uses_the_high_risk_gates() -> None:
    result = assess(
        _high(
            circulating_ratio=Decimal("0.30"),
            market_cap_rank=10_000,
            fdv_to_market_cap=Decimal("9"),
        ),
        UNIVERSE,
    )
    assert result == EligibilityDecision(
        eligible=True, lane=Lane.HIGH_RISK, reason_codes=("high_risk_lane",)
    )
    refused = assess(_normal(circulating_ratio=Decimal("0.30")), UNIVERSE)
    assert refused.reason_codes == ("circulating_ratio_below_minimum",)
    assert refused.lane is Lane.NORMAL


def test_a_cap_below_the_high_risk_floor_is_refused() -> None:
    cap = _money(UNIVERSE.high_risk.minimum_market_cap_usd) - 1
    result = assess(_high(market_cap_usd=cap), UNIVERSE)
    assert result.eligible is False
    assert result.lane is None
    assert result.reason_codes == ("below_market_cap",)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("eur_stable", None, "missing_eur_stable_classification"),
        ("eur_stable", True, "eur_stable"),
        ("fan_token", None, "missing_fan_token_classification"),
        ("fan_token", True, "fan_token"),
        ("status", None, "missing_trading_status"),
        ("status", "HALT", "non_trading"),
        ("monitoring_tag", None, "missing_monitoring_tag"),
        ("monitoring_tag", True, "monitoring_tag"),
        ("delisting", None, "missing_delisting_status"),
        ("delisting", True, "delisting"),
        ("deposits_suspended", None, "missing_deposit_status"),
        ("deposits_suspended", True, "deposits_suspended"),
        ("withdrawals_suspended", None, "missing_withdrawal_status"),
        ("withdrawals_suspended", True, "withdrawals_suspended"),
        ("pending_migration", None, "missing_migration_status"),
        ("pending_migration", True, "pending_migration"),
    ],
)
def test_a_missing_or_excluded_flag_is_refused(field: str, value: object, code: str) -> None:
    result = assess(_normal(**{field: value}), UNIVERSE)
    assert result.eligible is False
    assert result.reason_codes == (code,)


def test_listed_bases_are_excluded_without_becoming_a_buy() -> None:
    stable = UNIVERSE.exclusions.stablecoin_symbols[0]
    wrapped = UNIVERSE.exclusions.wrapped_symbols[0]
    stable_result = assess(
        _normal(symbol=f"{stable}USDT", base_asset=stable, status="HALT"), UNIVERSE
    )
    assert stable_result.reason_codes == ("stablecoin", "non_trading")
    wrapped_result = assess(_normal(symbol=f"{wrapped}USDT", base_asset=wrapped), UNIVERSE)
    assert wrapped_result.reason_codes == ("wrapped",)
    record = DecisionRecord(
        cohort=Cohort.SHADOW,
        symbol=f"{stable}USDT",
        evaluated_at=datetime(2026, 10, 4, tzinfo=UTC),
        policy_version="policy-sha",
        git_sha="a" * 40,
        disposition=Disposition.INELIGIBLE,
        reason_codes=stable_result.reason_codes,
        features={},
        score=None,
        score_components=None,
        rank=None,
        regime=None,
        raw_regime=None,
        sources=(
            SourceStamp(
                name="exchangeInfo",
                observed_at=datetime(2026, 10, 4, tzinfo=UTC),
                provenance="exchangeInfo",
            ),
        ),
    )
    assert record.disposition is Disposition.INELIGIBLE
    assert record.score is None


def test_normal_lane_thresholds_fail_closed() -> None:
    normal = UNIVERSE.normal
    assert assess(_normal(market_cap_usd=None), UNIVERSE).reason_codes == ("missing_market_cap",)
    assert assess(_normal(market_cap_rank=None), UNIVERSE).reason_codes == (
        "missing_market_cap_rank",
    )
    assert assess(
        _normal(market_cap_rank=normal.market_cap_rank_ceiling + 1), UNIVERSE
    ).reason_codes == ("market_cap_rank_above_ceiling",)
    assert assess(_normal(circulating_ratio=None), UNIVERSE).reason_codes == (
        "missing_circulating_ratio",
    )
    assert assess(_normal(fdv_to_market_cap=None), UNIVERSE).reason_codes == (
        "missing_fdv_to_market_cap",
    )
    assert assess(
        _normal(fdv_to_market_cap=_money(normal.maximum_fdv_to_market_cap) + 1), UNIVERSE
    ).reason_codes == ("fdv_to_market_cap_above_maximum",)
    assert assess(_normal(history_days=None), UNIVERSE).reason_codes == ("missing_history",)
    assert assess(_normal(history_days=normal.minimum_history_days - 1), UNIVERSE).reason_codes == (
        "history_below_minimum",
    )


def test_high_risk_thresholds_fail_closed() -> None:
    high = UNIVERSE.high_risk
    assert assess(_high(circulating_ratio=None), UNIVERSE).reason_codes == (
        "missing_circulating_ratio",
    )
    assert assess(
        _high(circulating_ratio=_money(high.minimum_circulating_ratio) - Decimal("0.01")),
        UNIVERSE,
    ).reason_codes == ("circulating_ratio_below_minimum",)
    assert assess(_high(unlock_schedule_known=None), UNIVERSE).reason_codes == (
        "missing_unlock_schedule",
    )
    assert assess(_high(unlock_schedule_known=False), UNIVERSE).reason_codes == (
        "unlock_schedule_unknown",
    )
    assert assess(_high(history_days=None), UNIVERSE).reason_codes == ("missing_history",)
    assert assess(_high(history_days=high.minimum_history_days - 1), UNIVERSE).reason_codes == (
        "history_below_minimum",
    )


def test_a_classification_rejection_stops_before_lane_fields() -> None:
    result = assess(
        _normal(
            withdrawals_suspended=True,
            market_cap_rank=None,
            circulating_ratio=None,
            fdv_to_market_cap=None,
            history_days=None,
        ),
        UNIVERSE,
    )
    assert result.lane is Lane.NORMAL
    assert result.reason_codes == ("withdrawals_suspended",)


def test_a_fan_token_with_no_market_cap_keeps_both_reasons() -> None:
    result = assess(_normal(fan_token=True, market_cap_usd=None), UNIVERSE)
    assert result.lane is None
    assert result.reason_codes == ("fan_token", "missing_market_cap")


def test_listing_age_must_match_the_lane_history_gate() -> None:
    high = UNIVERSE.model_copy(
        update={
            "new_listing": UNIVERSE.new_listing.model_copy(
                update={"high_risk_lane_days": UNIVERSE.high_risk.minimum_history_days + 1}
            )
        }
    )
    with pytest.raises(EvaluationError, match="high-risk"):
        assess(_normal(), high)
    normal = UNIVERSE.model_copy(
        update={
            "new_listing": UNIVERSE.new_listing.model_copy(
                update={"normal_lane_days": UNIVERSE.normal.minimum_history_days + 1}
            )
        }
    )
    with pytest.raises(EvaluationError, match="normal"):
        assess(_normal(), normal)


def test_the_replay_contract_uses_the_same_assessment() -> None:
    facts = _normal()
    gate = PolicyEligibility(UNIVERSE, {(facts.symbol, DAY): facts})
    assert isinstance(gate, Eligibility)
    assert gate.eligible(facts.symbol, DAY) is True
    assert gate.explain(facts.symbol, DAY).reason_codes == ("normal_lane",)
    assert gate.explain("ETHUSDT", DAY).reason_codes == ("missing_candidate_facts",)
    assert gate.eligible("ETHUSDT", DAY) is False
    mismatched = PolicyEligibility(UNIVERSE, {("BTCUSDT", DAY): facts})
    assert mismatched.explain("BTCUSDT", DAY).reason_codes == ("symbol_mismatch",)
    assert assess(_normal(symbol="BTCUSDT"), UNIVERSE).reason_codes == ("base_asset_mismatch",)


def test_malformed_facts_and_decisions_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _normal(symbol="sol")
    with pytest.raises(ValidationError):
        _normal(status="")
    with pytest.raises(ValidationError):
        _normal(market_cap_usd=0.1 + 0.2)
    with pytest.raises(ValidationError):
        _normal(market_cap_usd=True)
    with pytest.raises(ValidationError):
        _normal(market_cap_usd="nope")
    with pytest.raises(ValidationError):
        _normal(market_cap_usd="NaN")
    with pytest.raises(ValidationError):
        _normal(market_cap_usd=Decimal("NaN"))
    with pytest.raises(ValidationError):
        _normal(market_cap_usd=object())
    with pytest.raises(ValidationError):
        _normal(market_cap_rank=True)
    with pytest.raises(ValidationError):
        _normal(market_cap_rank=1.5)
    with pytest.raises(ValidationError):
        _normal(eur_stable=1)
    with pytest.raises(ValidationError):
        _normal(market_cap_rank=0)
    with pytest.raises(ValidationError):
        _normal(history_days=-1)
    with pytest.raises(ValidationError):
        EligibilityDecision(eligible=True, lane=None, reason_codes=("normal_lane",))
    with pytest.raises(ValidationError):
        EligibilityDecision(eligible=True, lane=Lane.NORMAL, reason_codes=("stablecoin",))
    with pytest.raises(ValidationError):
        EligibilityDecision(eligible=False, lane=Lane.NORMAL, reason_codes=("normal_lane",))
    with pytest.raises(ValidationError):
        EligibilityDecision(eligible=False, lane=None, reason_codes=("Nope",))
    with pytest.raises(ValidationError):
        EligibilityDecision(eligible=False, lane=None, reason_codes=())
