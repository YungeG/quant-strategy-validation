from __future__ import annotations

import ast
import json
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest
from crypto_quant_domain import (
    ArtifactEnvelope,
    ArtifactRef,
    canonical_bytes,
    canonical_sha256,
)
from crypto_quant_foundation import FoundationFailure, LocalFoundation
from crypto_quant_validation import (
    Holdout,
    NoReport,
    OosRule,
    PublishedValidationReport,
    SampleConsumptionLedger,
    SampleConsumptionRecord,
    ValidationPolicy,
    validate_candidate,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.support.backtest_consumer_port import (  # noqa: E402
    CONTRACT_V2_PATH,
    InMemoryBacktestConsumerPort,
)

ARTIFACT_LOG = "validation.artifacts.v1"
RESEARCH_ARTIFACT_LOG = "research.artifacts.v1"
RESEARCH_EXECUTION_LOG = "research.execution.v1"
SAMPLE_LOG = "validation.sample-consumption.v1"
RESERVED_AT = "2026-02-01T00:00:00.000000Z"
ACCEPTED_AT = "2026-02-02T00:00:00.000000Z"
DATA_START = "2026-01-01T00:00:00.000000Z"
DATA_END = "2026-02-01T00:00:00.000000Z"
HOLDOUT_START = "2026-03-01T00:00:00.000000Z"
HOLDOUT_END = "2026-04-01T00:00:00.000000Z"


def _event_id(log_name: str, ref: ArtifactRef) -> str:
    return canonical_sha256(("artifact-publication-v1", log_name, ref))


def _wire(ref: ArtifactRef) -> dict[str, object]:
    return ref.to_canonical_dict()


def _publish(
    foundation: LocalFoundation, log_name: str, artifact_type: str, payload: dict
) -> ArtifactRef:
    envelope = ArtifactEnvelope.create(artifact_type, 1, payload)
    ref = foundation.put(envelope=envelope)
    foundation.append(log_name, _event_id(log_name, ref), canonical_bytes(envelope))
    return ref


def _payload(foundation: LocalFoundation, ref: ArtifactRef) -> dict:
    return json.loads(foundation.read(ref=ref).source_bytes)["payload"]


def _artifact_payloads(foundation: LocalFoundation) -> list[dict]:
    return [json.loads(entry.payload) for entry in foundation.entries(ARTIFACT_LOG)]


def _record(producer_ref: ArtifactRef, purpose: str) -> SampleConsumptionRecord:
    return SampleConsumptionRecord(
        "eth-usdt-v1",
        DATA_START if purpose != "validation" else HOLDOUT_START,
        DATA_END if purpose != "validation" else HOLDOUT_END,
        purpose,
        canonical_sha256(("sample-consumer-v1", producer_ref)),
        RESERVED_AT,
    )


def _policy(
    metric_profile_ref: object,
    *,
    grade: str = "development",
) -> ValidationPolicy:
    return ValidationPolicy(
        accepted_backtest_grades=(grade,),
        accepted_metric_profile_refs=(metric_profile_ref,),
        holdout=Holdout(
            "market-bundle:oos",
            "eth-usdt-v1",
            HOLDOUT_START,
            HOLDOUT_END,
            "HOLDOUT",
            False,
        ),
        oos_rule=OosRule(
            metric_profile_ref,
            "simple_period_return",
            "fraction",
            "gte",
            "0",
            1,
        ),
    )


class RecordingPort(InMemoryBacktestConsumerPort):
    def __init__(self, foundation: LocalFoundation) -> None:
        super().__init__()
        self._foundation = foundation
        self.run_requests: list[dict[str, object]] = []
        self.run_sample_counts: list[int] = []
        self.run_artifact_types: list[list[str]] = []
        self.derive_calls = 0

    def run(self, request_spec: dict) -> dict:
        self.run_requests.append(deepcopy(request_spec))
        self.run_sample_counts.append(len(self._foundation.entries(SAMPLE_LOG)))
        self.run_artifact_types.append(
            [item["artifact_type"] for item in _artifact_payloads(self._foundation)]
        )
        return super().run(request_spec)

    def derive(self, completed_ref: dict, metric_profile_ref: dict) -> dict:
        self.derive_calls += 1
        return super().derive(completed_ref, metric_profile_ref)


class DecisionGradePort(RecordingPort):
    def __init__(self, foundation: LocalFoundation) -> None:
        InMemoryBacktestConsumerPort.__init__(self, contract_path=CONTRACT_V2_PATH)
        self._foundation = foundation
        self.run_requests = []
        self.run_sample_counts = []
        self.run_artifact_types = []
        self.derive_calls = 0
        self.completed_v3_calls: list[object] = []
        self.analysis_v2_calls: list[object] = []

    def load_completed(self, ref: dict) -> dict:
        raise AssertionError("V2 completion must not use load_completed")

    def load_analysis(self, ref: dict) -> dict:
        raise AssertionError("V2 analysis must not use load_analysis")

    def load_completed_v3(self, ref: dict) -> dict:
        self.completed_v3_calls.append(deepcopy(ref))
        return super().load_completed_v3(ref)

    def load_analysis_v2(self, ref: dict) -> dict:
        self.analysis_v2_calls.append(deepcopy(ref))
        return super().load_analysis_v2(ref)


class OosAnalysisPort(RecordingPort):
    def __init__(
        self, foundation: LocalFoundation, mutate: Callable[[dict], None]
    ) -> None:
        super().__init__(foundation)
        self._mutate = mutate
        self._analysis_loads = 0

    def load_analysis(self, analysis_ref: dict) -> dict:
        self._analysis_loads += 1
        record = super().load_analysis(analysis_ref)
        if self._analysis_loads >= 2:
            self._mutate(record)
        return record


def _candidate(
    foundation: LocalFoundation,
    ledger: SampleConsumptionLedger,
    port: InMemoryBacktestConsumerPort,
    *,
    case_id: str = "adverse_completed",
    grade: str = "development",
) -> ArtifactRef:
    fixture = port.case(case_id)
    profile_ref = fixture["derive"]["metric_profile_ref"]
    completed = fixture.get("completed_v3", fixture.get("completed"))
    analysis = fixture.get("analysis_v2", fixture.get("analysis"))
    if type(completed) is not dict or type(analysis) is not dict:
        raise ValueError("fixture must contain one completed and analysis view")
    publication_ref = completed["publication_ref"]
    analysis_ref = analysis["analysis_ref"]

    experiment_ref = _publish(
        foundation, RESEARCH_ARTIFACT_LOG, "experiment_spec", {"fixture": "one"}
    )
    trial_ref = _publish(
        foundation,
        RESEARCH_ARTIFACT_LOG,
        "trial_declaration",
        {
            "experiment_ref": _wire(experiment_ref),
            "parameter_values": [["threshold", "0.1"], ["window", "10"]],
            "data_slice": {
                "market_bundle_ref": "market-bundle:development",
                "dataset_revision": "eth-usdt-v1",
                "interval_start": DATA_START,
                "interval_end": DATA_END,
            },
            "scenario_ref": "scenario:base",
            "seed": 1,
            "backtest_template_ref": "template:one",
            "model_input_bindings": [],
        },
    )
    analysis_task_ref = _publish(
        foundation,
        RESEARCH_ARTIFACT_LOG,
        "analysis_task",
        {
            "experiment_ref": _wire(experiment_ref),
            "trial_declaration_ref": _wire(trial_ref),
            "metric_profile_ref": profile_ref,
        },
    )
    selection_policy_ref = _publish(
        foundation,
        RESEARCH_ARTIFACT_LOG,
        "selection_policy",
        {
            "metric_profile_ref": profile_ref,
            "eligible_trial_statuses": ["COMPLETED"],
            "accepted_backtest_grades": [grade],
            "hard_filters": [],
            "ordering": ["simple_period_return:descending"],
            "max_selections": 1,
            "tie_break": "trial_declaration_ref_ascending",
        },
    )
    selection_ref = _publish(
        foundation,
        RESEARCH_ARTIFACT_LOG,
        "selection_declaration",
        {
            "experiment_ref": _wire(experiment_ref),
            "selection_policy_ref": _wire(selection_policy_ref),
            "universe_kind": "candidate_trial_declarations_v1",
            "declared_by_ref": "actor:researcher",
        },
    )
    trial_spec_ref = _publish(
        foundation,
        RESEARCH_ARTIFACT_LOG,
        "backtest_trial_spec",
        {
            "trial_declaration_ref": _wire(trial_ref),
            "resolved_model_refs": [],
            "backtest_request_ref": {"type": "backtest_request_ref", "id": "fixture"},
        },
    )
    trial_outcome_ref = _publish(
        foundation,
        RESEARCH_EXECUTION_LOG,
        "task_outcome",
        {
            "task_ref": {"kind": "TRIAL", "task_artifact_ref": _wire(trial_ref)},
            "state": "COMPLETED",
            "witness": {
                "trial_completed_publication": {"publication_ref": publication_ref}
            },
        },
    )
    analysis_outcome_ref = _publish(
        foundation,
        RESEARCH_EXECUTION_LOG,
        "task_outcome",
        {
            "task_ref": {
                "kind": "ANALYSIS",
                "task_artifact_ref": _wire(analysis_task_ref),
            },
            "state": "COMPLETED",
            "witness": {
                "analysis_derivation": {
                    "analysis_ref": analysis_ref,
                    "source_publication_ref": publication_ref,
                }
            },
        },
    )
    manifest_ref = _publish(
        foundation,
        RESEARCH_EXECUTION_LOG,
        "experiment_execution_manifest",
        {
            "experiment_ref": _wire(experiment_ref),
            "task_outcome_refs": [
                _wire(trial_outcome_ref),
                _wire(analysis_outcome_ref),
            ],
        },
    )
    family_ref = _publish(
        foundation,
        RESEARCH_ARTIFACT_LOG,
        "candidate_family",
        {
            "experiment_ref": _wire(experiment_ref),
            "execution_manifest_ref": _wire(manifest_ref),
        },
    )
    candidate_ref = _publish(
        foundation,
        RESEARCH_ARTIFACT_LOG,
        "strategy_candidate",
        {
            "candidate_family_ref": _wire(family_ref),
            "selection_declaration_ref": _wire(selection_ref),
            "selected_trial_declaration_ref": _wire(trial_ref),
            "selected_trial_spec_ref": _wire(trial_spec_ref),
            "selected_publication_ref": publication_ref,
            "selected_analysis_ref": analysis_ref,
            "selection_rank": 1,
            "validated": False,
        },
    )
    ledger.reserve(_record(trial_ref, "discovery"), trial_ref)
    ledger.reserve(_record(selection_ref, "selection"), selection_ref)
    return candidate_ref


def _setup(
    tmp_path: Path,
    port_type: type[RecordingPort] = RecordingPort,
    mutate: Callable[[dict], None] | None = None,
) -> tuple[
    LocalFoundation,
    SampleConsumptionLedger,
    RecordingPort,
    ArtifactRef,
    ValidationPolicy,
]:
    foundation = LocalFoundation(tmp_path, clock=lambda: ACCEPTED_AT)
    ledger = SampleConsumptionLedger(foundation)
    port = (
        port_type(foundation, mutate)  # type: ignore[call-arg]
        if mutate is not None
        else port_type(foundation)
    )
    candidate_ref = _candidate(foundation, ledger, port)
    metric_profile_ref = port.case("adverse_completed")["derive"]["metric_profile_ref"]
    return foundation, ledger, port, candidate_ref, _policy(metric_profile_ref)


def _oos_result_payload(foundation: LocalFoundation) -> dict:
    return [
        envelope["payload"]
        for envelope in _artifact_payloads(foundation)
        if envelope["artifact_type"] == "validation_case_result"
    ][-1]


def test_adverse_fixture_publishes_rejected_report_after_snapshot_plan_and_reservation(
    tmp_path: Path,
) -> None:
    foundation, ledger, port, candidate_ref, policy = _setup(tmp_path)

    result = validate_candidate(
        candidate_ref,
        policy,
        {"fixture_case": "adverse_completed"},
        RESERVED_AT,
        foundation,
        ledger,
        port,
    )

    assert type(result) is PublishedValidationReport
    report = _payload(foundation, result.validation_report_ref)
    plan = _payload(foundation, result.validation_plan_ref)
    assert report["result"] == "rejected"
    assert report["validation_plan_ref"] == _wire(result.validation_plan_ref)
    assert plan["candidate_ref"] == _wire(candidate_ref)
    assert port.run_sample_counts == [3]
    context_ref = json.loads(port.run_requests[0]["experiment_id"])
    assert context_ref["artifact_type"] == "validation_case"
    assert foundation.read(
        ref=ArtifactRef(
            context_ref["artifact_type"],
            context_ref["schema_version"],
            context_ref["content_hash"],
        )
    ).envelope.artifact_type == "validation_case"
    assert port.derive_calls == 1
    assert port.run_artifact_types == [
        [
            "sample_consumption_ledger_snapshot",
            "validation_plan",
            "sample_integrity_assessment",
            "validation_case",
            "validation_case_result",
            "validation_case",
        ]
    ]
    assert [item["artifact_type"] for item in _artifact_payloads(foundation)] == [
        "sample_consumption_ledger_snapshot",
        "validation_plan",
        "sample_integrity_assessment",
        "validation_case",
        "validation_case_result",
        "validation_case",
        "validation_case_result",
        "validation_report",
    ]
    assert _oos_result_payload(foundation)["outcome"] == "FAIL"


def test_decision_grade_fixture_publishes_supported_report_and_replays(
    tmp_path: Path,
) -> None:
    foundation = LocalFoundation(tmp_path, clock=lambda: ACCEPTED_AT)
    ledger = SampleConsumptionLedger(foundation)
    port = DecisionGradePort(foundation)
    candidate_ref = _candidate(
        foundation,
        ledger,
        port,
        case_id="decision_grade_completed_v3",
        grade="decision_grade",
    )
    profile = port.case("decision_grade_completed_v3")["derive"][
        "metric_profile_ref"
    ]
    policy = _policy(profile, grade="decision_grade")

    result = validate_candidate(
        candidate_ref,
        policy,
        {"fixture_case": "decision_grade_completed_v3"},
        RESERVED_AT,
        foundation,
        ledger,
        port,
    )

    assert type(result) is PublishedValidationReport
    report = _payload(foundation, result.validation_report_ref)
    plan = _payload(foundation, result.validation_plan_ref)
    assert report["result"] == "supported"
    assert plan["accepted_backtest_grades"] == ["decision_grade"]
    assert port.completed_v3_calls
    assert port.analysis_v2_calls
    assert port.derive_calls == 1
    oos = _oos_result_payload(foundation)
    assert oos["outcome"] == "PASS"
    evidence = oos["evidence"]
    assert set(evidence) == {
        "publication_ref",
        "analysis_ref",
        "metric_profile_ref",
        "source_execution_result_hash",
        "result_grade",
        "metric_key",
        "metric_value",
        "trade_count",
    }
    assert evidence["result_grade"] == "decision_grade"
    assert "rebuild_verification_ref" not in evidence
    assert "proof_publication_manifest_ref" not in evidence

    before = (
        port.derive_calls,
        len(port.run_requests),
        len(foundation.entries(ARTIFACT_LOG)),
        len(foundation.entries(SAMPLE_LOG)),
    )
    replay = validate_candidate(
        candidate_ref,
        policy,
        {"fixture_case": "decision_grade_completed_v3"},
        RESERVED_AT,
        foundation,
        ledger,
        port,
    )
    assert replay == result
    assert before == (
        port.derive_calls,
        len(port.run_requests),
        len(foundation.entries(ARTIFACT_LOG)),
        len(foundation.entries(SAMPLE_LOG)),
    )


def test_decision_grade_v2_failure_produces_no_report_or_v1_fallback(
    tmp_path: Path,
) -> None:
    foundation = LocalFoundation(tmp_path, clock=lambda: ACCEPTED_AT)
    ledger = SampleConsumptionLedger(foundation)
    port = DecisionGradePort(foundation)
    candidate_ref = _candidate(
        foundation,
        ledger,
        port,
        case_id="decision_grade_completed_v3",
        grade="decision_grade",
    )
    case = port.case("decision_grade_completed_v3")
    port.inject_failures(
        case["completed_v3"]["publication_ref"],
        "PORT_STATIC_PROOF_MISMATCH",
    )
    policy = _policy(case["derive"]["metric_profile_ref"], grade="decision_grade")

    result = validate_candidate(
        candidate_ref,
        policy,
        {"fixture_case": "decision_grade_completed_v3"},
        RESERVED_AT,
        foundation,
        ledger,
        port,
    )

    assert result == NoReport(None, "PORT_STATIC_PROOF_MISMATCH")
    assert foundation.entries(ARTIFACT_LOG) == ()
    assert port.run_requests == []


@pytest.mark.parametrize(
    ("case_id", "outcome", "report_result", "reason_code"),
    (
        ("terminal_blocked", "BLOCKED", "inconclusive", None),
        ("terminal_cancelled", "INCONCLUSIVE", "inconclusive", None),
        ("terminal_failed", "FAILED", None, "BACKTEST_TERMINAL_FAILED"),
        ("provider_failure", "FAILED", None, "PORT_RETENTION_UNAVAILABLE"),
    ),
)
def test_terminal_and_provider_failures_never_derive_and_only_failures_have_no_report(
    tmp_path: Path,
    case_id: str,
    outcome: str,
    report_result: str | None,
    reason_code: str | None,
) -> None:
    foundation, ledger, port, candidate_ref, policy = _setup(tmp_path)

    result = validate_candidate(
        candidate_ref,
        policy,
        {"fixture_case": case_id},
        RESERVED_AT,
        foundation,
        ledger,
        port,
    )

    assert port.run_sample_counts == [3]
    assert port.derive_calls == 0
    assert _oos_result_payload(foundation)["outcome"] == outcome
    if report_result is None:
        assert type(result) is NoReport
        assert result.reason_code == reason_code
        assert not any(
            item["artifact_type"] == "validation_report"
            for item in _artifact_payloads(foundation)
        )
    else:
        assert type(result) is PublishedValidationReport
        assert (
            _payload(foundation, result.validation_report_ref)["result"]
            == report_result
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda record: record.pop("simple_period_return"),
        lambda record: record.__setitem__("trade_count", 0),
    ),
    ids=("missing_metric", "insufficient_trades"),
)
def test_valid_inconclusive_oos_evidence_is_not_zero_filled(
    tmp_path: Path, mutate: Callable[[dict], None]
) -> None:
    foundation, ledger, port, candidate_ref, policy = _setup(
        tmp_path, OosAnalysisPort, mutate
    )

    result = validate_candidate(
        candidate_ref,
        policy,
        {"fixture_case": "adverse_completed"},
        RESERVED_AT,
        foundation,
        ledger,
        port,
    )

    assert type(result) is PublishedValidationReport
    assert (
        _payload(foundation, result.validation_report_ref)["result"] == "inconclusive"
    )
    oos = _oos_result_payload(foundation)
    assert oos["outcome"] == "INCONCLUSIVE"
    assert (
        oos["evidence"]["metric_value"] is None or oos["evidence"]["trade_count"] == 0
    )


def test_snapshot_excludes_later_equal_time_holdout_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, port, candidate_ref, policy = _setup(tmp_path)
    original_assess = ledger.assess_holdout
    late_case = ArtifactRef("validation_case", 1, "sha256:" + "f" * 64)

    def assess_after_late_reservation(
        snapshot_ref: ArtifactRef, holdout: Holdout
    ) -> ArtifactRef:
        ledger.reserve(_record(late_case, "validation"), late_case)
        return original_assess(snapshot_ref, holdout)

    monkeypatch.setattr(ledger, "assess_holdout", assess_after_late_reservation)
    result = validate_candidate(
        candidate_ref,
        policy,
        {"fixture_case": "adverse_completed"},
        RESERVED_AT,
        foundation,
        ledger,
        port,
    )

    assert type(result) is PublishedValidationReport
    snapshot_ref = _payload(foundation, result.validation_plan_ref)[
        "sample_consumption_snapshot_ref"
    ]
    snapshot = next(
        envelope["payload"]
        for envelope in _artifact_payloads(foundation)
        if envelope["artifact_type"] == "sample_consumption_ledger_snapshot"
        and _wire(
            ArtifactRef(
                envelope["artifact_type"],
                envelope["schema_version"],
                envelope["content_hash"],
            )
        )
        == snapshot_ref
    )
    assessment = next(
        envelope["payload"]
        for envelope in _artifact_payloads(foundation)
        if envelope["artifact_type"] == "sample_integrity_assessment"
    )
    assert snapshot["checkpoint"]["upper_log_sequence"] == 2
    assert assessment["untouched"] is True
    assert len(foundation.entries(SAMPLE_LOG)) == 4
    assert _payload(foundation, result.validation_report_ref)["result"] == "rejected"


def test_forged_candidate_stops_before_validation_publication(tmp_path: Path) -> None:
    foundation, ledger, port, candidate_ref, policy = _setup(tmp_path)
    forged = deepcopy(_payload(foundation, candidate_ref))
    forged["validated"] = True
    forged_ref = _publish(
        foundation,
        RESEARCH_ARTIFACT_LOG,
        "strategy_candidate",
        forged,
    )

    result = validate_candidate(
        forged_ref,
        policy,
        {"fixture_case": "adverse_completed"},
        RESERVED_AT,
        foundation,
        ledger,
        port,
    )

    assert result == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
    assert foundation.entries(ARTIFACT_LOG) == ()
    assert port.run_sample_counts == []


def test_replay_reuses_the_original_plan_and_requires_an_explicit_fresh_run(
    tmp_path: Path,
) -> None:
    foundation, ledger, port, candidate_ref, policy = _setup(tmp_path)
    first = validate_candidate(
        candidate_ref,
        policy,
        {"fixture_case": "adverse_completed"},
        RESERVED_AT,
        foundation,
        ledger,
        port,
    )
    assert type(first) is PublishedValidationReport
    before = (
        len(port.run_sample_counts),
        port.derive_calls,
        len(foundation.entries(ARTIFACT_LOG)),
        len(foundation.entries(SAMPLE_LOG)),
    )

    replay = validate_candidate(
        candidate_ref,
        policy,
        {"fixture_case": "adverse_completed"},
        RESERVED_AT,
        foundation,
        ledger,
        port,
    )

    assert replay == first
    assert (
        len(port.run_sample_counts),
        port.derive_calls,
        len(foundation.entries(ARTIFACT_LOG)),
        len(foundation.entries(SAMPLE_LOG)),
    ) == before

    fresh = validate_candidate(
        candidate_ref,
        policy,
        {"fixture_case": "adverse_completed"},
        RESERVED_AT,
        foundation,
        ledger,
        port,
        fresh=True,
    )
    assert type(fresh) is PublishedValidationReport
    assert fresh.validation_plan_ref != first.validation_plan_ref


def test_foundation_failure_during_candidate_resolution_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, port, candidate_ref, policy = _setup(tmp_path)
    entries = foundation.entries

    def unavailable(log_name: str, through: object = None):
        if log_name == RESEARCH_ARTIFACT_LOG:
            raise FoundationFailure("WRITE_LOCK_UNAVAILABLE")
        return entries(log_name, through)

    monkeypatch.setattr(foundation, "entries", unavailable)
    with pytest.raises(FoundationFailure) as raised:
        validate_candidate(
            candidate_ref,
            policy,
            {"fixture_case": "adverse_completed"},
            RESERVED_AT,
            foundation,
            ledger,
            port,
        )
    assert raised.value.code == "WRITE_LOCK_UNAVAILABLE"
    assert foundation.entries(ARTIFACT_LOG) == ()
    assert port.run_sample_counts == []


def test_runtime_uses_no_research_or_backtest_adapter_import() -> None:
    path = (
        Path(__file__).resolve().parents[1] / "src/crypto_quant_validation/runtime.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports = {
        name
        for node in ast.walk(tree)
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }

    assert not any(name.startswith("crypto_quant_research") for name in imports)
    assert not any(name.startswith("crypto_quant_backtest") for name in imports)
    assert not any(name.startswith("tests") for name in imports)
    assert "Protocol" not in source
    assert "InMemoryBacktestConsumerPort" not in source
