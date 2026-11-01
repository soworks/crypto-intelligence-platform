"""Evaluate one finalized session. Inputs come from the freeze, not from a provider."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

from cip.domain.errors import EvaluationError
from cip.domain.policy import LoadedPolicy
from cip.evaluation.decision import Cohort, SourceStamp
from cip.evaluation.inputs import load_session
from cip.evaluation.populate import SessionManifest, _load_bars, _load_regime, replay_session
from cip.evaluation.scan import ScanResult, run_daily_scan
from cip.evaluation.store import DecisionWriter
from cip.evaluation.stored_candidates import assemble_stored_candidates


def scan_finalized_session(
    root: Path,
    session: date,
    *,
    cohort: Cohort,
    policy: LoadedPolicy,
    git_sha: str,
    correlation_id: str,
    writer: DecisionWriter,
    verified_ids: Mapping[str, str],
) -> ScanResult:
    """Store one decision per frozen symbol. Missing weights cannot become a BUY."""
    sealed = replay_session(root, session)
    manifest = sealed.manifest
    if manifest is None:
        raise EvaluationError("session is not finalized")
    if not sealed.readiness.ready:
        raise EvaluationError("session is not ready")
    if manifest.score_weights != "absent":
        raise EvaluationError("frozen score weights are not stored")
    recorded = load_session(root, session)
    observations, _failures = _load_regime(root, session)
    candidates = {
        symbol: candidate
        for symbol, candidate in recorded.candidates.items()
        if candidate is not None
    }
    if _classification_stored(root, session):
        candidates = {
            symbol: candidate
            for symbol, candidate in assemble_stored_candidates(
                root,
                session,
                recorded.snapshot,
                dict(recorded.candidates),
                policy.policy.hypotheses.universe,
            ).items()
            if candidate is not None
        }
    return run_daily_scan(
        session=session,
        as_of=manifest.finalized_at,
        cohort=cohort,
        policy=policy,
        git_sha=git_sha,
        correlation_id=correlation_id,
        snapshot=recorded.snapshot,
        candidates=candidates,
        bars=_load_bars(root, session),
        observations=observations,
        prior=(),
        verified_ids=verified_ids,
        weights=None,
        writer=writer,
        lineage=(_freeze_stamp(manifest),),
    )


def _classification_stored(root: Path, session: date) -> bool:
    folder = root / "sessions" / f"date={session.isoformat()}" / "classification"
    return folder.is_dir() and any(folder.glob("symbol=*.json"))


def _freeze_stamp(manifest: SessionManifest) -> SourceStamp:
    provenance = ";".join(
        (
            f"session={manifest.session_date.isoformat()}",
            f"manifest={manifest.input_manifest_version}",
            f"finalized_at={manifest.finalized_at.isoformat()}",
            f"git_sha={manifest.git_sha}",
            f"universe={manifest.universe_snapshot_sha256}",
            f"bars={manifest.bars_manifest_sha256}",
            f"candidates={manifest.candidate_manifest_sha256}",
            f"regime={manifest.regime_inputs_sha256}",
            f"liquidity={manifest.liquidity_inputs_sha256 or 'uncovered'}",
            f"hours={manifest.hour_grid_sha256 or 'uncovered'}",
            f"books={manifest.book_inputs_sha256 or 'uncovered'}",
            f"score_weights={manifest.score_weights}",
        )
    )
    return SourceStamp(
        name="session_freeze",
        observed_at=manifest.finalized_at,
        provenance=provenance,
    )
