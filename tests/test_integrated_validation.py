from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from crypto_quant_domain import (
    ArtifactEnvelope,
    ArtifactRef,
    canonical_bytes,
    canonical_sha256,
)
from crypto_quant_validation import (
    Holdout,
    NoReport,
    OosRule,
    PublishedValidationReport,
    ValidationPolicy,
    validate_candidate,
)

_PLATFORM_ROOT = Path(__file__).resolve().parents[2]
_RESEARCH_PATH = (
    _PLATFORM_ROOT / "research-platform/tests/test_integrated_research.py"
)
_RESEARCH_SPEC = importlib.util.spec_from_file_location(
    "platform_integrated_research", _RESEARCH_PATH
)
assert _RESEARCH_SPEC is not None and _RESEARCH_SPEC.loader is not None
_RESEARCH = importlib.util.module_from_spec(_RESEARCH_SPEC)
sys.modules[_RESEARCH_SPEC.name] = _RESEARCH
_RESEARCH_SPEC.loader.exec_module(_RESEARCH)

_RESERVED_AT = "2026-08-18T00:00:00.000000Z"
_HOLDOUT_START = "2026-03-01T00:00:00.000000Z"
_HOLDOUT_END = "2026-04-01T00:00:00.000000Z"


def _plain(value: object) -> Any:
    try:
        return json.loads(canonical_bytes(value))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("value is not canonical JSON") from error


def _policy(profile_ref: object) -> ValidationPolicy:
    return ValidationPolicy(
        accepted_backtest_grades=("development",),
        accepted_metric_profile_refs=(profile_ref,),
        holdout=Holdout(
            "market-bundle:oos",
            "cash-development-v1",
            _HOLDOUT_START,
            _HOLDOUT_END,
            "HOLDOUT",
            False,
        ),
        oos_rule=OosRule(
            profile_ref,
            "simple_period_return",
            "fraction",
            "gte",
            "0",
            1,
        ),
    )


def _payload(foundation, ref) -> dict[str, Any]:
    try:
        return json.loads(foundation.read(ref=ref).source_bytes)["payload"]
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
        pytest.fail(f"published Validation artifact is malformed: {error}")


def _artifact_types(foundation) -> list[str]:
    types: list[str] = []
    for entry in foundation.entries("validation.artifacts.v1"):
        try:
            types.append(json.loads(entry.payload)["artifact_type"])
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
            pytest.fail(f"Validation log entry is malformed: {error}")
    return types


def _runtime(tmp_path: Path, *, market: bool = True):
    foundation, ledger, provider, research_inputs = _RESEARCH._runtime(
        tmp_path / "research"
    )
    candidate = _RESEARCH.execute_experiment(
        research_inputs,
        foundation,
        ledger,
        provider,
    )
    assert type(candidate) is _RESEARCH.PublishedStrategyCandidate
    selected = _payload(foundation, candidate.strategy_candidate_ref)
    profile_ref = _plain(research_inputs.experiment_spec.metric_profile_refs[0])
    prepared = _RESEARCH._prepare_with(
        foundation,
        tmp_path / "research/publications",
        experiment_id="validation:oos:adverse",
        market=market,
    )
    provider._prepared["validation:oos"] = prepared
    return (
        foundation,
        ledger,
        provider,
        candidate.strategy_candidate_ref,
        selected,
        _policy(profile_ref),
    )


def _model_runtime(tmp_path: Path, *, missing_training_reservation: bool = False):
    foundation, ledger, builder, provider, research_inputs, artifact = (
        _RESEARCH._model_runtime(tmp_path / "research")
    )
    if missing_training_reservation:
        reserve = ledger.reserve

        def omit_training(record, producer_ref):
            if record.purpose == "model_training":
                return None
            return reserve(record, producer_ref)

        ledger.reserve = omit_training  # type: ignore[method-assign]
    candidate = _RESEARCH.execute_model_experiment(
        research_inputs,
        foundation,
        ledger,
        builder,
        provider,
    )
    assert type(candidate) is _RESEARCH.PublishedStrategyCandidate
    selected = _payload(foundation, candidate.strategy_candidate_ref)
    profile_ref = _plain(research_inputs.experiment_spec.metric_profile_refs[0])
    prepared = _RESEARCH.backtest.prepare_model_bound_cash_development_backtest(
        request_intent=_RESEARCH._BINDING_MODULE._intent("validation:oos:model"),
        provider_inputs=_RESEARCH._BINDING_MODULE._provider_inputs(),
        model_timeline=provider._timeline,
        expected_model_key=artifact.model_key,
        expected_artifact_ref_hash=artifact.artifact_ref_hash,
        artifact_reader=foundation,
        artifact_publisher=foundation,
        market_reader=_RESEARCH._BINDING_MODULE._market_reader(),
        publication_root=tmp_path / "research/publications",
    )
    provider._prepared["validation:oos:model"] = prepared
    return (
        foundation,
        ledger,
        provider,
        candidate.strategy_candidate_ref,
        selected,
        _policy(profile_ref),
    )


def test_real_model_candidate_preserves_rejected_oos_report_and_replay(
    tmp_path: Path,
) -> None:
    foundation, ledger, provider, candidate_ref, selected, policy = _model_runtime(
        tmp_path
    )
    runs = provider.run_calls

    first = validate_candidate(
        candidate_ref,
        policy,
        {"binding_key": "validation:oos:model"},
        _RESERVED_AT,
        foundation,
        ledger,
        provider,
    )
    second = validate_candidate(
        candidate_ref,
        policy,
        {"binding_key": "validation:oos:model"},
        _RESERVED_AT,
        foundation,
        ledger,
        provider,
    )

    assert type(first) is PublishedValidationReport
    assert second == first
    assert provider.run_calls == runs + 1
    assert _payload(foundation, first.validation_report_ref)["result"] == "rejected"
    assert "model_build_evidence_ref" in selected


def test_model_candidate_missing_training_reservation_produces_no_report(
    tmp_path: Path,
) -> None:
    foundation, ledger, provider, candidate_ref, _, policy = _model_runtime(
        tmp_path,
        missing_training_reservation=True,
    )

    result = validate_candidate(
        candidate_ref,
        policy,
        {"binding_key": "validation:oos:model"},
        _RESERVED_AT,
        foundation,
        ledger,
        provider,
    )

    assert type(result) is NoReport
    assert result.reason_code == "SAMPLE_RESERVATION_COVERAGE_MISSING"


def test_model_candidate_substituted_build_evidence_produces_no_report(
    tmp_path: Path,
) -> None:
    foundation, ledger, provider, _, selected, policy = _model_runtime(tmp_path)
    forged = dict(selected)
    forged["model_build_evidence_ref"] = ArtifactRef(
        "model_build_evidence", 1, "sha256:" + "0" * 64
    ).to_canonical_dict()
    envelope = ArtifactEnvelope.create("strategy_candidate", 2, forged)
    candidate_ref = foundation.put(envelope=envelope)
    foundation.append(
        "research.artifacts.v1",
        canonical_sha256(
            ("artifact-publication-v1", "research.artifacts.v1", candidate_ref)
        ),
        canonical_bytes(envelope),
    )

    result = validate_candidate(
        candidate_ref,
        policy,
        {"binding_key": "validation:oos:model"},
        _RESERVED_AT,
        foundation,
        ledger,
        provider,
    )

    assert type(result) is NoReport
    assert result.reason_code == "CANDIDATE_PROVENANCE_INVALID"


def test_real_candidate_publishes_rejected_oos_report_and_replays(tmp_path: Path) -> None:
    foundation, ledger, provider, candidate_ref, selected, policy = _runtime(tmp_path)
    research_runs = provider.run_calls
    reservations = len(foundation.entries("validation.sample-consumption.v1"))

    first = validate_candidate(
        candidate_ref,
        policy,
        {"binding_key": "validation:oos"},
        _RESERVED_AT,
        foundation,
        ledger,
        provider,
    )
    after_first_runs = provider.run_calls
    second = validate_candidate(
        candidate_ref,
        policy,
        {"binding_key": "validation:oos"},
        _RESERVED_AT,
        foundation,
        ledger,
        provider,
    )

    assert type(first) is PublishedValidationReport
    assert second == first
    assert provider.run_calls == after_first_runs == research_runs + 1
    assert provider.derive_calls == 4
    assert provider.reservations_before_run[-1] == reservations + 1
    report = _payload(foundation, first.validation_report_ref)
    assert report["result"] == "rejected"
    assert report["case_result_refs"]
    assert selected["selected_publication_ref"]
    plan = _payload(foundation, first.validation_plan_ref)
    assert plan["sample_consumption_snapshot_ref"]
    assert _artifact_types(foundation).count("validation_plan") == 1


def test_real_blocked_oos_remains_inconclusive(tmp_path: Path) -> None:
    foundation, ledger, provider, candidate_ref, _, policy = _runtime(
        tmp_path,
        market=False,
    )

    result = validate_candidate(
        candidate_ref,
        policy,
        {"binding_key": "validation:oos"},
        _RESERVED_AT,
        foundation,
        ledger,
        provider,
    )

    assert type(result) is PublishedValidationReport
    assert _payload(foundation, result.validation_report_ref)["result"] == (
        "inconclusive"
    )


@pytest.mark.parametrize("failure", ("tamper", "retention"))
def test_real_candidate_repository_failure_produces_no_report(
    tmp_path: Path,
    failure: str,
) -> None:
    foundation, ledger, provider, candidate_ref, _, policy = _runtime(tmp_path)
    provider._repository_failure = failure
    runs = provider.run_calls

    result = validate_candidate(
        candidate_ref,
        policy,
        {"binding_key": "validation:oos"},
        _RESERVED_AT,
        foundation,
        ledger,
        provider,
    )

    assert type(result) is NoReport
    assert result.reason_code in {
        "PORT_EVIDENCE_TAMPERED",
        "PORT_RETENTION_UNAVAILABLE",
    }
    assert provider.run_calls == runs
