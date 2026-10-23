from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from crypto_quant_research import verify_candidate_provenance
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


def _model_policy(profile_ref):
    _, _, reader = _RESEARCH._BINDING_MODULE._model_case("validation:oos:model", scope="oos")
    ordinary = _policy(profile_ref)
    return ValidationPolicy(
        accepted_backtest_grades=ordinary.accepted_backtest_grades,
        accepted_metric_profile_refs=ordinary.accepted_metric_profile_refs,
        holdout=Holdout(reader.bundle_ref.to_canonical_dict(), "model-evaluation-synthetic-oos-v1",
                        "1970-01-01T00:00:00.000004Z", "1970-01-01T00:00:00.000005Z", "HOLDOUT", False),
        oos_rule=ordinary.oos_rule,
    )


def _prepare_model_oos(foundation, publication_root, provider, artifact):
    # Inert configuration: the scoped provider materializes only after OOS reservation.
    return {
        "publication_root": str(publication_root),
        "model_key": artifact.model_key,
        "artifact_ref_hash": artifact.artifact_ref_hash,
    }


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
    prepared = _prepare_model_oos(foundation, tmp_path / "research/publications", provider, artifact)
    provider._prepared["validation:oos:model"] = prepared
    return (
        foundation,
        ledger,
        provider,
        candidate.strategy_candidate_ref,
        selected,
        _model_policy(profile_ref),
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
        candidate_provenance_verifier=verify_candidate_provenance,
    )
    oos_prepared = provider._prepared["validation:oos:model"]
    source_reads = provider.source_read_calls
    ledger_before_replay = foundation.entries("validation.sample-consumption.v1")
    second = validate_candidate(
        candidate_ref,
        policy,
        {"binding_key": "validation:oos:model"},
        _RESERVED_AT,
        foundation,
        ledger,
        provider,
        candidate_provenance_verifier=verify_candidate_provenance,
    )

    assert type(first) is PublishedValidationReport
    assert second == first
    assert provider._prepared["validation:oos:model"] is oos_prepared
    assert provider.source_read_calls == source_reads
    assert provider.oos_prepare_calls == 1
    assert foundation.entries("validation.sample-consumption.v1") == ledger_before_replay
    assert provider.run_calls == runs + 1
    assert _payload(foundation, first.validation_report_ref)["result"] == "rejected"
    assert "model_build_evidence_ref" in selected
    prepared = provider._prepared["validation:oos:model"]
    assert type(prepared) is not dict
    publications = [ref for ref in provider._refs.values()
                    if type(ref) is _RESEARCH.backtest.AnalysisArtifactRef]
    analyses = [provider.load_analysis(_plain(ref)) for ref in publications]
    assert analyses[-1]["trade_count"] == 1
    assert analyses[-1]["simple_period_return"] == "-0.1"


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
        candidate_provenance_verifier=verify_candidate_provenance,
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
        candidate_provenance_verifier=verify_candidate_provenance,
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
        candidate_provenance_verifier=verify_candidate_provenance,
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
        candidate_provenance_verifier=verify_candidate_provenance,
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
        candidate_provenance_verifier=verify_candidate_provenance,
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
        candidate_provenance_verifier=verify_candidate_provenance,
    )

    assert type(result) is NoReport
    assert result.reason_code in {
        "PORT_EVIDENCE_TAMPERED",
        "PORT_RETENTION_UNAVAILABLE",
    }
    assert provider.run_calls == runs


def test_model_experiment3_exact_graph_fields_fail_closed(tmp_path):
    from copy import deepcopy
    from dataclasses import replace
    from crypto_quant_validation.integration import assess_admission, build_validation_plan
    from crypto_quant_validation.runtime import _candidate_graph
    foundation, ledger, provider, ref, _, policy = _model_runtime(tmp_path)
    graph, _ = _candidate_graph(ref, foundation, provider, _RESERVED_AT,
                             candidate_provenance_verifier=verify_candidate_provenance)
    plan = build_validation_plan(_plain(ref), _plain(ledger.freeze_snapshot()), policy)
    experiment = graph.model_build.experiment_spec
    assert experiment.ref['schema_version'] == 3
    for mode in ('missing', 'extra', 'legacy'):
        payload = deepcopy(experiment.payload)
        changed_ref = dict(experiment.ref)
        if mode == 'missing':
            del payload['evaluation_slices']
        elif mode == 'extra':
            payload['extra'] = 1
        else:
            changed_ref['schema_version'] = 1
        mutated = replace(graph, model_build=replace(graph.model_build,
                          experiment_spec=replace(experiment, ref=changed_ref, payload=payload)))
        result = assess_admission(plan, mutated, None)
        assert result.outcome == 'FAILED'
        assert result.reason_codes == ('CANDIDATE_PROVENANCE_INVALID',)


def test_model_oos_reservation_failure_prevents_deferred_prepare(tmp_path, monkeypatch):
    foundation, ledger, provider, ref, _, policy = _model_runtime(tmp_path)
    original = ledger.reserve
    error = RuntimeError('OOS reservation failure')
    def reserve(record, producer):
        if record.purpose == 'validation':
            raise error
        return original(record, producer)
    ledger.reserve = reserve
    def forbidden(*args, **kwargs):
        pytest.fail('reservation failure must prevent preparation')
    monkeypatch.setattr(_RESEARCH.backtest, 'prepare_model_bound_cash_development_backtest', forbidden)
    with pytest.raises(RuntimeError) as caught:
        validate_candidate(ref, policy, {'binding_key': 'validation:oos:model'}, _RESERVED_AT,
                           foundation, ledger, provider, candidate_provenance_verifier=verify_candidate_provenance)
    assert caught.value is error
    assert type(provider._prepared['validation:oos:model']) is dict
