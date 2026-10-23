from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from crypto_quant_research import verify_candidate_provenance
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
        _add_research_cases(self)
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
        _add_research_cases(self)
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

    def load_analysis(self, analysis_ref: dict) -> dict:
        record = super().load_analysis(analysis_ref)
        if analysis_ref == self.case("adverse_completed")["analysis"]["analysis_ref"]:
            self._mutate(record)
        return record


def _incomplete_candidate(
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


def _candidate(
    foundation: LocalFoundation,
    ledger: SampleConsumptionLedger,
    port: RecordingPort,
    *,
    case_id: str = "adverse_completed",
    grade: str = "development",
) -> ArtifactRef:
    """Produce actual Research closure; the incomplete writer above is negative/history only."""
    from crypto_quant_research import FrozenExperimentInputs, TrialExecution, execute_experiment
    from crypto_quant_research.integration import DataSlice, build_trial_declarations
    path = ROOT / "research-platform/tests/test_research_shell.py"
    spec_module = importlib.util.spec_from_file_location("validation_research_fixture", path)
    module = importlib.util.module_from_spec(spec_module)
    spec_module.loader.exec_module(module)
    source = module._spec() if grade == "development" else module._decision_grade_inputs().experiment_spec
    spec = replace(
        source, data_slices=(DataSlice(source.data_slices[0].market_bundle_ref, "eth-usdt-v1", DATA_START, DATA_END),),
        parameter_combinations=source.parameter_combinations[:1], seeds=(1,), budget={"max_trials": 1},
    )
    trial = build_trial_declarations(spec)[0]
    inputs = FrozenExperimentInputs(
        spec, module._policy(spec, grades=(grade,)), {"type": "actor_ref", "actor_id": "research"},
        (TrialExecution(trial.ref, {"fixture_case": "research:" + case_id}, {"type": "backtest_request_ref", "id": "fixture"}),),
        RESERVED_AT,
    )
    result = execute_experiment(inputs, foundation, ledger, port)
    port.research_run_count = len(port.run_requests)
    port.research_derive_count = port.derive_calls
    return result.strategy_candidate_ref


def _add_research_cases(port: InMemoryBacktestConsumerPort) -> None:
    # Separate immutable records, never call-order-dependent evidence for one ref.
    for original in tuple(port._contract["cases"]):
        field = "completed_v3" if "completed_v3" in original else "completed"
        if field not in original:
            continue
        case = deepcopy(original)
        case["case_id"] = "research:" + original["case_id"]
        case["request_spec"] = {"fixture_case": case["case_id"]}
        for ref in (case["run"]["ref"], case[field]["publication_ref"], case["analysis_v2" if field == "completed_v3" else "analysis"]["source_publication_ref"]):
            ref["artifact_ref"]["content_hash"] = canonical_sha256(("research-publication", original["case_id"]))
        analysis = case["analysis_v2" if field == "completed_v3" else "analysis"]
        for ref in (case["derive"]["analysis_ref"], analysis["analysis_ref"]):
            ref["artifact_ref"]["content_hash"] = canonical_sha256(("research-analysis", original["case_id"]))
        execution_hash = canonical_sha256(("research-execution", original["case_id"]))
        case[field]["semantic_run_id"] = "run_" + canonical_sha256(("research-semantic-run", original["case_id"]))[7:]
        case[field]["execution_result_hash"] = execution_hash
        analysis["source_execution_result_hash"] = execution_hash
        port._contract["cases"].append(case)


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
        candidate_provenance_verifier=verify_candidate_provenance,
    )

    assert type(result) is PublishedValidationReport
    report = _payload(foundation, result.validation_report_ref)
    plan = _payload(foundation, result.validation_plan_ref)
    assert report["result"] == "rejected"
    assert report["validation_plan_ref"] == _wire(result.validation_plan_ref)
    assert plan["candidate_ref"] == _wire(candidate_ref)
    assert port.run_sample_counts[port.research_run_count:] == [3]
    context_ref = json.loads(port.run_requests[port.research_run_count]["experiment_id"])
    assert context_ref["artifact_type"] == "validation_case"
    assert foundation.read(
        ref=ArtifactRef(
            context_ref["artifact_type"],
            context_ref["schema_version"],
            context_ref["content_hash"],
        )
    ).envelope.artifact_type == "validation_case"
    assert port.derive_calls - port.research_derive_count == 1
    assert port.run_artifact_types[port.research_run_count:] == [
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


def _historical_wire_report(foundation, ledger, port, candidate_ref, policy):
    """Pure-kernel serialization history only: this incomplete producer is NOT admitted."""
    from crypto_quant_validation import (CandidateGraph, ResolvedArtifact, VerifiedCandidateProvenance,
        assess_admission, assess_oos, aggregate_validation_report, build_validation_plan,
        OosObservation, AnalysisObservation)
    from crypto_quant_validation import runtime as wire
    def node(ref):
        return ResolvedArtifact(_wire(ref), _payload(foundation, ref))
    def ref(value):
        return ArtifactRef(value["artifact_type"], value["schema_version"], value["content_hash"])
    c = _payload(foundation, candidate_ref)
    family = node(ref(c["candidate_family_ref"]))
    manifest = node(ref(family.payload["execution_manifest_ref"]))
    selection = node(ref(c["selection_declaration_ref"]))
    outcomes = [node(ref(value)) for value in manifest.payload["task_outcome_refs"]]
    cutoff = foundation.entries(RESEARCH_EXECUTION_LOG)[-1].entry_ref
    graph = CandidateGraph(
        node(candidate_ref), family, manifest, selection,
        node(ref(selection.payload["selection_policy_ref"])),
        node(ref(c["selected_trial_declaration_ref"])), node(ref(c["selected_trial_spec_ref"])),
        outcomes[0], node(ref(outcomes[1].payload["task_ref"]["task_artifact_ref"])), outcomes[1],
        port.load_completed(c["selected_publication_ref"]), port.load_analysis(c["selected_analysis_ref"]),
        tuple(record for record, _, _ in wire._sample_entries(foundation)),
        provenance=VerifiedCandidateProvenance(_wire(candidate_ref), family.ref, family.payload["experiment_ref"], manifest.ref, selection.ref, wire._entry_ref_wire(cutoff)),
    )
    snapshot_ref = ledger.freeze_snapshot()
    plan = build_validation_plan(_wire(candidate_ref), _wire(snapshot_ref), policy)
    plan_ref = wire._publish(foundation, "validation_plan", wire._plan_payload(plan))
    assessment_ref = ledger.assess_holdout(snapshot_ref, plan.holdout)
    admission_case = wire._publish(foundation, "validation_case", {"validation_plan_ref": _wire(plan_ref), "case_type": "evidence_integrity"})
    admission = assess_admission(plan, graph, wire._snapshot_evidence(foundation, snapshot_ref, assessment_ref, policy))
    admission_ref = wire._publish(foundation, "validation_case_result", wire._case_result_payload(admission_case, admission))
    oos_case = wire._publish(foundation, "validation_case", {"validation_plan_ref": _wire(plan_ref), "case_type": "out_of_sample"})
    ledger.reserve(_record(oos_case, "validation"), oos_case)
    case = port.case("adverse_completed")
    oos = assess_oos(plan, OosObservation(plan, _wire(oos_case), _wire(oos_case), case["completed"]), AnalysisObservation(plan, _wire(oos_case), case["analysis"]))
    oos_ref = wire._publish(foundation, "validation_case_result", wire._case_result_payload(oos_case, oos))
    report = aggregate_validation_report(plan, (admission, oos))
    report_ref = wire._publish(foundation, "validation_report", wire._report_payload(report, plan_ref, (admission_ref, oos_ref), assessment_ref))
    return PublishedValidationReport(plan_ref, report_ref)


def test_legacy_validation_v1_canonical_bytes_are_frozen(tmp_path: Path) -> None:
    foundation = LocalFoundation(tmp_path, clock=lambda: ACCEPTED_AT)
    ledger = SampleConsumptionLedger(foundation)
    port = RecordingPort(foundation)
    candidate_ref = _incomplete_candidate(foundation, ledger, port)
    policy = _policy(port.case("adverse_completed")["derive"]["metric_profile_ref"])
    result = _historical_wire_report(foundation, ledger, port, candidate_ref, policy)

    assert type(result) is PublishedValidationReport
    assert result.validation_plan_ref.content_hash == (
        "sha256:3e96ed28dd5f9ceb93daa5fe7fab19d47db483c6c75524b210810369593b7ca7"
    )
    assert result.validation_report_ref.content_hash == (
        "sha256:d3ef01c6682d7a9250200cc6bee6ee91c5c1977ad98727f7051db3ba8976d35b"
    )
    assert [
        (
            json.loads(entry.payload)["artifact_type"],
            json.loads(entry.payload)["schema_version"],
            hashlib.sha256(entry.payload).hexdigest(),
        )
        for entry in foundation.entries(ARTIFACT_LOG)
    ] == [
        ("sample_consumption_ledger_snapshot", 1, "64f3dfd53999f3592b69476b5f296346fa9f1d60207585d3f9957d31526c8f2f"),
        ("validation_plan", 1, "ece0ee44375f3b1988fccbcdc3bddd955d7c74bb743f67ce4907e0c058f84857"),
        ("sample_integrity_assessment", 1, "9a9c821df1d5243872c62029bce640f56ab522e06cba65b27d88696b1c8dc7fd"),
        ("validation_case", 1, "d9e61b3748cb5d1559672eea3a84b3853893e14e18a9fb65ec24ae7a80301810"),
        ("validation_case_result", 1, "ed473e729f22fc9c10ec26e037d1c094411e0f6c823fd146bb7d00bdbba9d781"),
        ("validation_case", 1, "03e1bfff89dc6f1fc39e7c0753d8bedf0cb8820058364b3254b94aaa80d5ba99"),
        ("validation_case_result", 1, "83bbe3ff8979864d5203ed6c0b5b7c809e45ec2f52aea18d35367baa10cec5b7"),
        ("validation_report", 1, "c3d0e63b75a9d2ebe94e67937b26a1b41f3543fb08674fccc492357d5bc49c20"),
    ]
    assert [hashlib.sha256(entry.payload).hexdigest() for entry in foundation.entries(SAMPLE_LOG)] == [
        "45080578b47d4fad2996ad567bb6d688171b1349bfe379e4efeb6135e5673b53",
        "eab001401ce0a7987e09fb7a16a0531b674c20dea553edb458be6a8b82fa274b",
        "b0752cf826202734c06c594eb8b14efb9576420fe00e2014a7b95b4576a1a3c7",
    ]


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
        candidate_provenance_verifier=verify_candidate_provenance,
    )

    assert type(result) is PublishedValidationReport
    report = _payload(foundation, result.validation_report_ref)
    plan = _payload(foundation, result.validation_plan_ref)
    assert report["result"] == "supported"
    assert plan["accepted_backtest_grades"] == ["decision_grade"]
    assert port.completed_v3_calls
    assert port.analysis_v2_calls
    assert port.derive_calls - port.research_derive_count == 1
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
        candidate_provenance_verifier=verify_candidate_provenance,
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
        candidate_provenance_verifier=verify_candidate_provenance,
    )

    assert type(result) is NoReport
    assert result.reason_code == "PORT_STATIC_PROOF_MISMATCH"
    assert not any(item["artifact_type"] == "validation_report" for item in _artifact_payloads(foundation))
    assert len(port.run_requests[port.research_run_count:]) == 1


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
        candidate_provenance_verifier=verify_candidate_provenance,
    )

    assert port.run_sample_counts[port.research_run_count:] == [3]
    assert port.derive_calls - port.research_derive_count == 0
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
        candidate_provenance_verifier=verify_candidate_provenance,
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
        candidate_provenance_verifier=verify_candidate_provenance,
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
        candidate_provenance_verifier=verify_candidate_provenance,
    )

    assert result == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
    assert foundation.entries(ARTIFACT_LOG) == ()
    assert port.run_sample_counts[port.research_run_count:] == []


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
        candidate_provenance_verifier=verify_candidate_provenance,
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
        candidate_provenance_verifier=verify_candidate_provenance,
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
        candidate_provenance_verifier=verify_candidate_provenance,
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
            candidate_provenance_verifier=verify_candidate_provenance,
        )
    assert raised.value.code == "WRITE_LOCK_UNAVAILABLE"
    assert foundation.entries(ARTIFACT_LOG) == ()
    assert port.run_sample_counts[port.research_run_count:] == []


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


def test_missing_verifier_fails_closed_before_io(tmp_path, monkeypatch):
    foundation, ledger, port, candidate_ref, policy = _setup(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("missing verifier performed I/O")
    for name in ("read", "entries", "put", "append", "checkpoint"):
        monkeypatch.setattr(foundation, name, forbidden)
    assert validate_candidate(candidate_ref, policy, {}, RESERVED_AT, foundation, ledger, port) == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")


def test_replay_rejects_unbacked_supported_report(tmp_path):
    foundation, ledger, port, candidate_ref, policy = _setup(tmp_path)
    first = validate_candidate(candidate_ref, policy, {"fixture_case": "provider_failure"}, RESERVED_AT, foundation, ledger, port, candidate_provenance_verifier=verify_candidate_provenance)
    assert type(first) is NoReport and first.validation_plan_ref is not None
    _publish(foundation, ARTIFACT_LOG, "validation_report", {
        "validation_plan_ref": _wire(first.validation_plan_ref), "result": "supported",
        "case_result_refs": [], "threshold_evaluations": [],
        "sample_integrity_ref": _wire(candidate_ref), "limitations": [],
    })
    before = (len(foundation.entries(ARTIFACT_LOG)), len(foundation.entries(SAMPLE_LOG)), len(port.run_requests), port.derive_calls)
    assert validate_candidate(candidate_ref, policy, {}, RESERVED_AT, foundation, ledger, port, candidate_provenance_verifier=verify_candidate_provenance) == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
    assert before == (len(foundation.entries(ARTIFACT_LOG)), len(foundation.entries(SAMPLE_LOG)), len(port.run_requests), port.derive_calls)
