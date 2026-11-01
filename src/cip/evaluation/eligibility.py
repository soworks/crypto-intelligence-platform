from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Annotated, Self, cast

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from cip.domain.errors import EvaluationError
from cip.domain.policy import UniverseHypotheses

_SYMBOL = re.compile(r"^[A-Z0-9]{1,20}$")
_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _finite_decimal(value: object) -> Decimal:
    if isinstance(value, (bool, float, int)):
        raise ValueError("decimal values are Decimal or decimal strings")
    if isinstance(value, str):
        try:
            parsed = Decimal(value)
        except InvalidOperation as error:
            raise ValueError("decimal values are Decimal or decimal strings") from error
        if not parsed.is_finite():
            raise ValueError("decimal values are finite")
        return parsed
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("decimal values are finite")
        return value
    raise ValueError("decimal values are Decimal or decimal strings")


def _optional_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    return _finite_decimal(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if type(value) is bool:
        return value
    raise ValueError("flags are booleans")


def _optional_int(value: object, *, minimum: int) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError("counts are integers")
    if value < minimum:
        raise ValueError(f"counts are at least {minimum}")
    return value


def _optional_rank(value: object) -> int | None:
    return _optional_int(value, minimum=1)


def _optional_history(value: object) -> int | None:
    return _optional_int(value, minimum=0)


def _threshold(value: float) -> Decimal:
    return Decimal(str(value))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Lane(StrEnum):
    NORMAL = "normal"
    HIGH_RISK = "high_risk"


class CandidateFacts(_Strict):
    """Point-in-time inputs for one symbol. Missing values stay missing."""

    symbol: str
    base_asset: str
    quote_asset: str
    status: str | None
    eur_stable: Annotated[bool | None, BeforeValidator(_optional_bool)]
    fan_token: Annotated[bool | None, BeforeValidator(_optional_bool)]
    monitoring_tag: Annotated[bool | None, BeforeValidator(_optional_bool)]
    delisting: Annotated[bool | None, BeforeValidator(_optional_bool)]
    deposits_suspended: Annotated[bool | None, BeforeValidator(_optional_bool)]
    withdrawals_suspended: Annotated[bool | None, BeforeValidator(_optional_bool)]
    pending_migration: Annotated[bool | None, BeforeValidator(_optional_bool)]
    market_cap_usd: Annotated[Decimal | None, BeforeValidator(_optional_decimal)]
    market_cap_rank: Annotated[int | None, BeforeValidator(_optional_rank)]
    circulating_ratio: Annotated[Decimal | None, BeforeValidator(_optional_decimal)]
    fdv_to_market_cap: Annotated[Decimal | None, BeforeValidator(_optional_decimal)]
    history_days: Annotated[int | None, BeforeValidator(_optional_history)]
    unlock_schedule_known: Annotated[bool | None, BeforeValidator(_optional_bool)]

    @field_validator("symbol", "base_asset", "quote_asset")
    @classmethod
    def _symbol(cls, value: str) -> str:
        if _SYMBOL.fullmatch(value) is None:
            raise ValueError("symbol must be 1 to 20 uppercase letters or digits")
        return value

    @field_validator("status")
    @classmethod
    def _status(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value == "":
            raise ValueError("status is empty")
        return value


class EligibilityDecision(_Strict):
    """Lane membership or the reasons it was refused. This is not an order."""

    eligible: bool
    lane: Lane | None
    reason_codes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _codes_match_the_lane(self) -> Self:
        if any(_CODE.fullmatch(code) is None for code in self.reason_codes):
            raise ValueError("reason codes are lowercase snake_case")
        lane_code = None if self.lane is None else f"{self.lane.value}_lane"
        if self.eligible:
            if lane_code is None or self.reason_codes != (lane_code,):
                raise ValueError("an eligible symbol carries its lane")
            return self
        if lane_code is not None and lane_code in self.reason_codes:
            raise ValueError("a rejection does not carry a lane code")
        return self


def assess(facts: CandidateFacts, universe: UniverseHypotheses) -> EligibilityDecision:
    """Apply exclusions and the non-liquidity lane gates. A pass is not a BUY."""
    _listing_age_agrees(universe)
    if facts.symbol != f"{facts.base_asset}{facts.quote_asset}":
        return _decision(False, None, ("base_asset_mismatch",))
    reasons = _exclusions(facts, universe)
    lane, lane_reasons = _lane(facts, universe)
    if reasons:
        if lane is None:
            reasons.extend(lane_reasons)
        return _decision(False, lane, tuple(reasons))
    reasons.extend(lane_reasons)
    if reasons:
        return _decision(False, lane, tuple(reasons))
    chosen = cast(Lane, lane)
    return _decision(True, chosen, (f"{chosen.value}_lane",))


class PolicyEligibility:
    """Replay adapter. `eligible` is the contract; `explain` keeps the reason codes."""

    def __init__(
        self,
        universe: UniverseHypotheses,
        facts: Mapping[tuple[str, date], CandidateFacts],
    ) -> None:
        self._universe = universe
        self._facts = dict(facts)

    def explain(self, symbol: str, as_of: date) -> EligibilityDecision:
        found = self._facts.get((symbol, as_of))
        if found is None:
            return _decision(False, None, ("missing_candidate_facts",))
        if found.symbol != symbol:
            return _decision(False, None, ("symbol_mismatch",))
        return assess(found, self._universe)

    def eligible(self, symbol: str, as_of: date) -> bool:
        return self.explain(symbol, as_of).eligible


def _decision(
    eligible: bool, lane: Lane | None, reason_codes: tuple[str, ...]
) -> EligibilityDecision:
    return EligibilityDecision(eligible=eligible, lane=lane, reason_codes=reason_codes)


def _listing_age_agrees(universe: UniverseHypotheses) -> None:
    listing = universe.new_listing
    if listing.high_risk_lane_days != universe.high_risk.minimum_history_days:
        raise EvaluationError("high-risk listing age disagrees with the lane history gate")
    if listing.normal_lane_days != universe.normal.minimum_history_days:
        raise EvaluationError("normal listing age disagrees with the lane history gate")


def _exclusions(facts: CandidateFacts, universe: UniverseHypotheses) -> list[str]:
    exclusions = universe.exclusions
    reasons: list[str] = []
    if facts.base_asset in exclusions.stablecoin_symbols:
        reasons.append("stablecoin")
    reasons.extend(
        _flag(
            facts.eur_stable,
            missing="missing_eur_stable_classification",
            rejected="eur_stable",
        )
    )
    if facts.base_asset in exclusions.wrapped_symbols:
        reasons.append("wrapped")
    reasons.extend(
        _flag(facts.fan_token, missing="missing_fan_token_classification", rejected="fan_token")
    )
    if facts.status is None:
        reasons.append("missing_trading_status")
    elif facts.status != "TRADING":
        reasons.append("non_trading")
    reasons.extend(
        _flag(facts.monitoring_tag, missing="missing_monitoring_tag", rejected="monitoring_tag")
    )
    reasons.extend(_flag(facts.delisting, missing="missing_delisting_status", rejected="delisting"))
    reasons.extend(
        _flag(
            facts.deposits_suspended,
            missing="missing_deposit_status",
            rejected="deposits_suspended",
        )
    )
    reasons.extend(
        _flag(
            facts.withdrawals_suspended,
            missing="missing_withdrawal_status",
            rejected="withdrawals_suspended",
        )
    )
    reasons.extend(
        _flag(
            facts.pending_migration,
            missing="missing_migration_status",
            rejected="pending_migration",
        )
    )
    return reasons


def _flag(value: bool | None, *, missing: str, rejected: str) -> tuple[str, ...]:
    if value is None:
        return (missing,)
    if value:
        return (rejected,)
    return ()


def _lane(facts: CandidateFacts, universe: UniverseHypotheses) -> tuple[Lane | None, list[str]]:
    if facts.market_cap_usd is None:
        return None, ["missing_market_cap"]
    normal_floor = _threshold(universe.normal.minimum_market_cap_usd)
    high_floor = _threshold(universe.high_risk.minimum_market_cap_usd)
    if facts.market_cap_usd >= normal_floor:
        return Lane.NORMAL, _normal_reasons(facts, universe)
    if facts.market_cap_usd >= high_floor:
        return Lane.HIGH_RISK, _high_reasons(facts, universe)
    return None, ["below_market_cap"]


def _normal_reasons(facts: CandidateFacts, universe: UniverseHypotheses) -> list[str]:
    lane = universe.normal
    reasons: list[str] = []
    if facts.market_cap_rank is None:
        reasons.append("missing_market_cap_rank")
    elif facts.market_cap_rank > lane.market_cap_rank_ceiling:
        reasons.append("market_cap_rank_above_ceiling")
    reasons.extend(
        _minimum(
            facts.circulating_ratio,
            lane.minimum_circulating_ratio,
            missing="missing_circulating_ratio",
            below="circulating_ratio_below_minimum",
        )
    )
    if facts.fdv_to_market_cap is None:
        reasons.append("missing_fdv_to_market_cap")
    elif facts.fdv_to_market_cap > _threshold(lane.maximum_fdv_to_market_cap):
        reasons.append("fdv_to_market_cap_above_maximum")
    reasons.extend(_history(facts.history_days, lane.minimum_history_days))
    return reasons


def _high_reasons(facts: CandidateFacts, universe: UniverseHypotheses) -> list[str]:
    lane = universe.high_risk
    reasons: list[str] = []
    reasons.extend(
        _minimum(
            facts.circulating_ratio,
            lane.minimum_circulating_ratio,
            missing="missing_circulating_ratio",
            below="circulating_ratio_below_minimum",
        )
    )
    if facts.unlock_schedule_known is None:
        reasons.append("missing_unlock_schedule")
    elif not facts.unlock_schedule_known:
        reasons.append("unlock_schedule_unknown")
    reasons.extend(_history(facts.history_days, lane.minimum_history_days))
    return reasons


def _minimum(value: Decimal | None, minimum: float, *, missing: str, below: str) -> tuple[str, ...]:
    if value is None:
        return (missing,)
    if value < _threshold(minimum):
        return (below,)
    return ()


def _history(value: int | None, minimum: int) -> tuple[str, ...]:
    if value is None:
        return ("missing_history",)
    if value < minimum:
        return ("history_below_minimum",)
    return ()
