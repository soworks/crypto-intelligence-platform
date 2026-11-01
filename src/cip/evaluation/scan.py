"""Daily scan over stored inputs. It does not call providers and it does not place orders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from cip.domain.errors import EvaluationError
from cip.domain.events import EventType, LedgerEvent
from cip.domain.policy import LoadedPolicy
from cip.evaluation.decision import (
    Cohort,
    DecisionRecord,
    Disposition,
    SourceStamp,
    decision_id,
)
from cip.evaluation.eligibility import CandidateFacts, _exclusions, assess
from cip.evaluation.features import Tokenomics, measure
from cip.evaluation.fundamentals import (
    CmcReading,
    CoinGeckoReading,
    UnlockReading,
    assess_fundamentals,
)
from cip.evaluation.liquidity import MarketSnapshot, assess_market
from cip.evaluation.regime import PriorSession, RegimeDecision, classify
from cip.evaluation.score import combine
from cip.evaluation.store import DecisionWriter, StoredDecision
from cip.history.bars import DailyBar
from cip.recorders.observation import Observation

_BTC = "BTCUSDT"
_Regime = Literal["RISK_ON", "NEUTRAL", "RISK_OFF"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UniverseSnapshot(_Strict):
    """Symbols discovered for one closed session. This is a stored document."""

    session: date
    symbols: tuple[str, ...]
    observed_at: datetime
    provenance: str = Field(min_length=1)


class ScanCandidate(_Strict):
    """Stored facts for one snapshot symbol. Missing packets stay missing."""

    facts: CandidateFacts
    market: MarketSnapshot
    coingecko: CoinGeckoReading
    cmc: CmcReading
    unlocks: UnlockReading
    tokenomics: Tokenomics | None


@dataclass(frozen=True)
class ScanResult:
    """Decisions written for one closed session, plus the ledger events that name them."""

    evaluated_at: datetime
    records: tuple[DecisionRecord, ...]
    stored: tuple[StoredDecision, ...]
    events: tuple[LedgerEvent, ...]


@dataclass(frozen=True)
class _Prepared:
    symbol: str
    disposition: Disposition
    reason_codes: tuple[str, ...]
    features: dict[str, Decimal]
    score: Decimal | None
    components: dict[str, Decimal] | None
    regime: _Regime | None
    raw_regime: _Regime | None
    sources: tuple[SourceStamp, ...]


def run_daily_scan(
    *,
    session: date,
    as_of: datetime,
    cohort: Cohort,
    policy: LoadedPolicy,
    git_sha: str,
    correlation_id: str,
    snapshot: UniverseSnapshot | None,
    candidates: Mapping[str, ScanCandidate],
    bars: Mapping[str, Sequence[DailyBar]],
    observations: Sequence[Observation],
    prior: Sequence[PriorSession],
    verified_ids: Mapping[str, str],
    weights: Mapping[str, Decimal] | None,
    writer: DecisionWriter,
    lineage: Sequence[SourceStamp] = (),
) -> ScanResult:
    """Evaluate every snapshot symbol and store one decision for each."""
    evaluated_at = _close(session, as_of)
    if not correlation_id:
        raise EvaluationError("correlation id is missing")
    if snapshot is None:
        raise EvaluationError("universe snapshot is missing")
    if snapshot.session != session:
        raise EvaluationError("universe snapshot is for a different session")
    if len(snapshot.symbols) != len(set(snapshot.symbols)):
        raise EvaluationError("universe snapshot repeats a symbol")
    for symbol in snapshot.symbols:
        candidate = candidates.get(symbol)
        if candidate is not None and candidate.facts.symbol != symbol:
            raise EvaluationError("candidate symbol does not match the snapshot")
    hypotheses = policy.policy.hypotheses
    regime = classify(
        as_of=session,
        btc_bars=bars.get(_BTC, ()),
        universe_bars={symbol: series for symbol, series in bars.items() if symbol != _BTC},
        observations=observations,
        prior=prior,
        hypotheses=hypotheses.regime,
    )
    prepared = tuple(
        _prepare(
            symbol,
            candidates.get(symbol),
            session,
            evaluated_at,
            snapshot,
            policy,
            bars,
            verified_ids,
            weights,
            regime,
        )
        for symbol in snapshot.symbols
    )
    ranks = _ranks(prepared)
    records = tuple(
        _record(
            item,
            ranks.get(item.symbol),
            cohort,
            evaluated_at,
            policy.version,
            git_sha,
            lineage,
        )
        for item in prepared
    )
    stored = tuple(writer.write(record) for record in records)
    events = tuple(
        _ledger_event(correlation_id, policy.version, evaluated_at, item) for item in stored
    )
    return ScanResult(evaluated_at, records, stored, events)


def _close(session: date, as_of: datetime) -> datetime:
    if as_of.tzinfo is None or as_of.utcoffset() != timedelta(0):
        raise EvaluationError("as_of must be timezone-aware UTC")
    evaluated_at = datetime.combine(session + timedelta(days=1), time.min, tzinfo=UTC)
    if as_of < evaluated_at:
        raise EvaluationError("session has not closed")
    return evaluated_at


def _prepare(
    symbol: str,
    candidate: ScanCandidate | None,
    session: date,
    evaluated_at: datetime,
    snapshot: UniverseSnapshot,
    policy: LoadedPolicy,
    bars: Mapping[str, Sequence[DailyBar]],
    verified_ids: Mapping[str, str],
    weights: Mapping[str, Decimal] | None,
    regime: RegimeDecision,
) -> _Prepared:
    sources = _sources(snapshot, candidate, evaluated_at)
    hypotheses = policy.policy.hypotheses
    features = measure(
        symbol=symbol,
        as_of=session,
        bars=bars.get(symbol, ()),
        btc_bars=bars.get(_BTC, ()),
        universe_bars={name: series for name, series in bars.items() if name != _BTC},
        atr_period=hypotheses.exits.atr_period_days,
        tokenomics=None if candidate is None else candidate.tokenomics,
    ).features
    if candidate is None:
        return _ineligible(symbol, ("missing_candidate",), features, regime, sources)
    market = assess(candidate.facts, hypotheses.universe)
    classification_closed = bool(_exclusions(candidate.facts, hypotheses.universe))
    if not classification_closed and (market.eligible or _market_observed(candidate.market)):
        market = assess_market(candidate.facts, candidate.market, hypotheses.universe)
    if not market.eligible:
        return _ineligible(symbol, market.reason_codes, features, regime, sources)
    fundamentals = assess_fundamentals(
        base_asset=candidate.facts.base_asset,
        verified_ids=verified_ids,
        coingecko=candidate.coingecko,
        cmc=candidate.cmc,
        unlocks=candidate.unlocks,
        as_of=evaluated_at,
        hypotheses=hypotheses.fundamentals,
    )
    if not fundamentals.accepted:
        return _ineligible(symbol, fundamentals.reason_codes, features, regime, sources)
    if regime.regime is None:
        return _ineligible(symbol, regime.reason_codes, features, regime, sources)
    disposition, reasons, score, components = _score(features, weights, regime)
    if score is None:
        return _ineligible(symbol, reasons, features, regime, sources)
    return _Prepared(
        symbol,
        disposition,
        reasons,
        features,
        score,
        components,
        regime.regime,
        regime.raw,
        sources,
    )


def _market_observed(market: MarketSnapshot) -> bool:
    """A clock alone is not a book. Null fields stay unobserved."""
    observed = market.model_dump()
    observed.pop("as_of")
    return any(value is not None for value in observed.values())


def _score(
    features: Mapping[str, Decimal],
    weights: Mapping[str, Decimal] | None,
    regime: RegimeDecision,
) -> tuple[Disposition, tuple[str, ...], Decimal | None, dict[str, Decimal] | None]:
    if not weights:
        return Disposition.INELIGIBLE, ("score_weights_not_frozen",), None, None
    try:
        result = combine(features, weights)
    except EvaluationError:
        return Disposition.INELIGIBLE, ("score_refused",), None, None
    disposition, reasons = _disposition(regime, result.score, features)
    return disposition, reasons, result.score, result.components


def _disposition(
    regime: RegimeDecision,
    score: Decimal,
    features: Mapping[str, Decimal],
) -> tuple[Disposition, tuple[str, ...]]:
    if regime.regime == "RISK_OFF":
        return Disposition.SCORED, ("risk_off",)
    if regime.require_rs_vs_btc_30d_positive:
        spread = features.get("rs_30d_skip_1")
        if spread is None:
            return Disposition.SCORED, ("missing_rs_30d",)
        if spread <= 0:
            return Disposition.SCORED, ("rs_30d_not_positive",)
    minimum = cast(int, regime.min_score)
    if score < Decimal(minimum):
        return Disposition.SCORED, ("below_min_score",)
    return Disposition.BUY, ("buy_recommendation",)


def _ineligible(
    symbol: str,
    reason_codes: tuple[str, ...],
    features: dict[str, Decimal],
    regime: RegimeDecision,
    sources: tuple[SourceStamp, ...],
) -> _Prepared:
    return _Prepared(
        symbol,
        Disposition.INELIGIBLE,
        reason_codes,
        features,
        None,
        None,
        regime.regime,
        regime.raw,
        sources,
    )


def _sources(
    snapshot: UniverseSnapshot, candidate: ScanCandidate | None, evaluated_at: datetime
) -> tuple[SourceStamp, ...]:
    stamps = [
        SourceStamp(
            name="universe_snapshot",
            observed_at=snapshot.observed_at,
            provenance=snapshot.provenance,
        ),
        SourceStamp(name="daily_bars", observed_at=evaluated_at, provenance="stored_daily_bars"),
    ]
    if candidate is not None:
        stamps.append(
            SourceStamp(
                name="market_snapshot",
                observed_at=candidate.market.as_of,
                provenance="stored_market_snapshot",
            )
        )
    return tuple(stamps)


def _ranks(items: Sequence[_Prepared]) -> dict[str, int]:
    scored: list[tuple[Decimal, str]] = []
    for item in items:
        if item.score is not None:
            scored.append((item.score, item.symbol))
    ordered = sorted(scored, key=lambda pair: (-pair[0], pair[1]))
    return {symbol: index for index, (_score, symbol) in enumerate(ordered, start=1)}


def _record(
    item: _Prepared,
    rank: int | None,
    cohort: Cohort,
    evaluated_at: datetime,
    policy_version: str,
    git_sha: str,
    lineage: Sequence[SourceStamp],
) -> DecisionRecord:
    return DecisionRecord(
        cohort=cohort,
        symbol=item.symbol,
        evaluated_at=evaluated_at,
        policy_version=policy_version,
        git_sha=git_sha,
        disposition=item.disposition,
        reason_codes=item.reason_codes,
        features=item.features,
        score=item.score,
        score_components=item.components,
        rank=rank,
        regime=item.regime,
        raw_regime=item.raw_regime,
        sources=item.sources + tuple(lineage),
    )


def _ledger_event(
    correlation_id: str, policy_version: str, evaluated_at: datetime, stored: StoredDecision
) -> LedgerEvent:
    record = stored.record
    return LedgerEvent(
        event_type=EventType.DECISION_RECORDED,
        correlation_id=correlation_id,
        policy_version=policy_version,
        asset=record.symbol,
        created_at=evaluated_at,
        idempotency_key=decision_id(record),
        payload={
            "decision_key": stored.key,
            "sha256": stored.sha256,
            "symbol": record.symbol,
            "disposition": record.disposition.value,
        },
    )
