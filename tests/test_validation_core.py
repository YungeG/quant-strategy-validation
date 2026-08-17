from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crypto_quant_validation import (  # noqa: E402
    AnalysisObservation,
    CandidateGraph,
    CompletedCaseEvidence,
    Holdout,
    OosObservation,
    OosRule,
    ProviderFailureEvidence,
    ResolvedArtifact,
    SampleAdmissionEvidence,
    SampleConsumptionRecord,
    TerminalCaseEvidence,
    ValidationCoreFailure,
    ValidationPlan,
    ValidationPolicy,
    aggregate_validation_report,
    assess_admission,
    assess_oos,
    assess_untouched_holdout,
    build_snapshot,
    build_validation_plan,
)
from tests.support.backtest_consumer_port import (  # noqa: E402
    InMemoryBacktestConsumerPort,
    PortFailure,
)

INTEGRATION_PATH = (
    ROOT / "strategy-validation/src/crypto_quant_validation/integration.py"
)
DATA_START = "2026-01-01T00:00:00.000000Z"
DATA_END = "2026-02-01T00:00:00.000000Z"
HOLDOUT_START = "2026-03-01T00:00:00.000000Z"
HOLDOUT_END = "2026-04-01T00:00:00.000000Z"
AS_OF = "2026-02-15T00:00:00.000000Z"
HEAD = "sha256:" + "a" * 64


def _port_evidence() -> tuple[dict, dict, dict]:
    port = InMemoryBacktestConsumerPort()
    fixture = port.case("adverse_completed")
    completed_ref = port.run(fixture["request_spec"])
    completed = port.load_completed(completed_ref)
    analysis_ref = port.derive(completed_ref, fixture["derive"]["metric_profile_ref"])
    analysis = port.load_analysis(analysis_ref)
    return completed, analysis, fixture["derive"]["metric_profile_ref"]


def _record(
    purpose: str, consumer_id: str, *, start: str = DATA_START, end: str = DATA_END
) -> SampleConsumptionRecord:
    return SampleConsumptionRecord(
        "eth-usdt-v1",
        start,
        end,
        purpose,
        consumer_id,
        "2026-02-10T00:00:00.000000Z",
    )


def _policy(metric_profile_ref: object) -> ValidationPolicy:
    return ValidationPolicy(
        accepted_backtest_grades=("development",),
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


def _plan(metric_profile_ref: object) -> ValidationPlan:
    return build_validation_plan(
        "candidate:selected",
        "snapshot:pre-oos",
        _policy(metric_profile_ref),
    )


def _node(ref: str, payload: dict | None) -> ResolvedArtifact:
    return ResolvedArtifact(ref, payload)


def _candidate_graph(
    plan: ValidationPlan,
    completed: dict,
    analysis: dict,
) -> CandidateGraph:
    experiment_ref = "experiment:one"
    family_ref = "family:one"
    manifest_ref = "manifest:one"
    selection_ref = "selection:one"
    policy_ref = "selection-policy:one"
    trial_ref = "trial:t10-1"
    trial_spec_ref = "trial-spec:t10-1"
    trial_outcome_ref = "outcome:trial:t10-1"
    analysis_task_ref = "analysis-task:t10-1"
    analysis_outcome_ref = "outcome:analysis:t10-1"
    required = (
        _record("discovery", "trial-consumer"),
        _record("selection", "selection-consumer"),
    )

    return CandidateGraph(
        candidate=_node(
            "candidate:selected",
            {
                "candidate_family_ref": family_ref,
                "selection_declaration_ref": selection_ref,
                "selected_trial_declaration_ref": trial_ref,
                "selected_trial_spec_ref": trial_spec_ref,
                "selected_publication_ref": completed["publication_ref"],
                "selected_analysis_ref": analysis["analysis_ref"],
                "selection_rank": 1,
                "validated": False,
            },
        ),
        candidate_family=_node(
            family_ref,
            {
                "experiment_ref": experiment_ref,
                "execution_manifest_ref": manifest_ref,
            },
        ),
        execution_manifest=_node(
            manifest_ref,
            {
                "experiment_ref": experiment_ref,
                "task_outcome_refs": (trial_outcome_ref, analysis_outcome_ref),
            },
        ),
        selection_declaration=_node(
            selection_ref,
            {
                "experiment_ref": experiment_ref,
                "selection_policy_ref": policy_ref,
                "universe_kind": "candidate_trial_declarations_v1",
                "declared_by_ref": "actor:researcher",
            },
        ),
        selection_policy=_node(
            policy_ref,
            {
                "metric_profile_ref": analysis["metric_profile_ref"],
                "eligible_trial_statuses": ("COMPLETED",),
                "accepted_backtest_grades": ("development",),
                "hard_filters": (),
                "ordering": ("simple_period_return:descending",),
                "max_selections": 1,
                "tie_break": "trial_declaration_ref_ascending",
            },
        ),
        selected_trial_declaration=_node(
            trial_ref,
            {
                "experiment_ref": experiment_ref,
                "parameter_values": (("threshold", "0.1"), ("window", "10")),
                "data_slice": {
                    "market_bundle_ref": "market-bundle:development",
                    "dataset_revision": "eth-usdt-v1",
                    "interval_start": DATA_START,
                    "interval_end": DATA_END,
                },
                "scenario_ref": "scenario:base",
                "seed": 1,
                "backtest_template_ref": "template:one",
                "model_input_bindings": (),
            },
        ),
        selected_trial_spec=_node(
            trial_spec_ref,
            {
                "trial_declaration_ref": trial_ref,
                "resolved_model_refs": (),
                "backtest_request_ref": "backtest-request:t10-1",
            },
        ),
        selected_trial_outcome=_node(
            trial_outcome_ref,
            {
                "task_ref": {"kind": "TRIAL", "task_artifact_ref": trial_ref},
                "state": "COMPLETED",
                "witness": {
                    "trial_completed_publication": {
                        "publication_ref": completed["publication_ref"],
                    }
                },
            },
        ),
        selected_analysis_task=_node(
            analysis_task_ref,
            {
                "experiment_ref": experiment_ref,
                "trial_declaration_ref": trial_ref,
                "metric_profile_ref": analysis["metric_profile_ref"],
            },
        ),
        selected_analysis_outcome=_node(
            analysis_outcome_ref,
            {
                "task_ref": {
                    "kind": "ANALYSIS",
                    "task_artifact_ref": analysis_task_ref,
                },
                "state": "COMPLETED",
                "witness": {
                    "analysis_derivation": {
                        "analysis_ref": analysis["analysis_ref"],
                        "source_publication_ref": completed["publication_ref"],
                    }
                },
            },
        ),
        selected_completed=completed,
        selected_analysis=analysis,
        required_sample_records=required,
    )


def _sample_evidence(
    plan: ValidationPlan,
    records: tuple[SampleConsumptionRecord, ...],
    *,
    ledger_conflict: bool = False,
    snapshot_ref: object | None = None,
) -> SampleAdmissionEvidence:
    snapshot = build_snapshot(records, as_of=AS_OF)
    integrity = assess_untouched_holdout(
        snapshot,
        dataset_revision=plan.holdout.dataset_revision,
        interval_start=plan.holdout.interval_start,
        interval_end=plan.holdout.interval_end,
    )
    return SampleAdmissionEvidence(
        plan.sample_consumption_snapshot_ref if snapshot_ref is None else snapshot_ref,
        {
            "log_name": "validation.sample-consumption.v1",
            "as_of": snapshot.as_of,
            "upper_log_sequence": len(snapshot.records),
            "head_receipt_hash": None if not snapshot.records else HEAD,
        },
        snapshot,
        integrity,
        ledger_conflict,
    )


def _admitted() -> tuple[ValidationPlan, CandidateGraph, object, dict, dict]:
    completed, analysis, metric_profile_ref = _port_evidence()
    plan = _plan(metric_profile_ref)
    graph = _candidate_graph(plan, completed, analysis)
    evidence = _sample_evidence(plan, graph.required_sample_records)
    admission = assess_admission(plan, graph, evidence)
    assert admission.outcome == "PASS"
    return plan, graph, admission, completed, analysis


def _completed_observation(
    plan: ValidationPlan, completed: dict, *, context: str = "case:oos"
) -> OosObservation:
    return OosObservation(plan, "case:oos", context, completed)


def _analysis_observation(
    plan: ValidationPlan, analysis: dict, *, case_ref: str = "case:oos"
) -> AnalysisObservation:
    return AnalysisObservation(plan, case_ref, analysis)


def test_plan_is_pre_result_candidate_specific_and_preserves_six_field_record() -> None:
    _, _, metric_profile_ref = _port_evidence()
    plan = _plan(metric_profile_ref)

    assert set(ValidationPlan.__dataclass_fields__) == {
        "candidate_ref",
        "sample_consumption_snapshot_ref",
        "accepted_backtest_grades",
        "accepted_metric_profile_refs",
        "holdout",
        "oos_rule",
        "decision_rule",
    }
    assert plan.accepted_backtest_grades == ("development",)
    assert plan.holdout.selection_observed is False
    assert set(SampleConsumptionRecord.__dataclass_fields__) == {
        "dataset_revision",
        "interval_start",
        "interval_end",
        "purpose",
        "consumer_id",
        "consumed_at",
    }
    for forbidden in ("result", "analysis_ref", "simple_period_return", "trade_count"):
        assert not hasattr(plan, forbidden)

    forged = object.__new__(ValidationPolicy)
    object.__setattr__(forged, "accepted_backtest_grades", ("decision_grade",))
    object.__setattr__(forged, "accepted_metric_profile_refs", (metric_profile_ref,))
    object.__setattr__(forged, "holdout", plan.holdout)
    object.__setattr__(forged, "oos_rule", plan.oos_rule)
    object.__setattr__(forged, "decision_rule", plan.decision_rule)
    with pytest.raises(ValidationCoreFailure) as caught:
        build_validation_plan("candidate:selected", "snapshot:pre-oos", forged)
    assert caught.value.code == "VALIDATION_PLAN_INVALID"


def test_admission_enforces_accepted_candidate_checkpoint_and_reservations() -> None:
    plan, graph, admission, _, _ = _admitted()
    assert admission.outcome == "PASS"
    assert admission.evidence.snapshot_ref == plan.sample_consumption_snapshot_ref
    assert admission.evidence.untouched is True

    missing_family = replace(
        graph,
        candidate_family=ResolvedArtifact(graph.candidate_family.ref, None),
    )
    result = assess_admission(
        plan,
        missing_family,
        _sample_evidence(plan, graph.required_sample_records),
    )
    assert (result.outcome, result.reason_codes) == (
        "BLOCKED",
        ("CANDIDATE_PROVENANCE_INVALID",),
    )

    forged_candidate = deepcopy(graph.candidate.payload)
    forged_candidate["validated"] = True
    forged_graph = replace(
        graph,
        candidate=ResolvedArtifact(graph.candidate.ref, forged_candidate),
    )
    precedence = assess_admission(
        plan,
        forged_graph,
        _sample_evidence(
            plan,
            graph.required_sample_records,
            ledger_conflict=True,
            snapshot_ref="snapshot:wrong",
        ),
    )
    assert (precedence.outcome, precedence.reason_codes) == (
        "FAILED",
        ("CANDIDATE_PROVENANCE_INVALID",),
    )

    wrong_checkpoint = _sample_evidence(
        plan,
        graph.required_sample_records,
        snapshot_ref="snapshot:wrong",
    )
    result = assess_admission(plan, graph, wrong_checkpoint)
    assert (result.outcome, result.reason_codes) == (
        "FAILED",
        ("SAMPLE_LEDGER_CONFLICT",),
    )

    missing_reservation = _sample_evidence(
        plan,
        (graph.required_sample_records[0],),
    )
    result = assess_admission(plan, graph, missing_reservation)
    assert (result.outcome, result.reason_codes) == (
        "BLOCKED",
        ("SAMPLE_RESERVATION_COVERAGE_MISSING",),
    )

    contaminated = _sample_evidence(
        plan,
        graph.required_sample_records
        + (
            _record(
                "validation", "old-validation", start=HOLDOUT_START, end=HOLDOUT_END
            ),
        ),
    )
    result = assess_admission(plan, graph, contaminated)
    assert (result.outcome, result.reason_codes) == (
        "BLOCKED",
        ("HOLDOUT_CONTAMINATED",),
    )


def test_verified_development_comparison_rejects_adverse_oos() -> None:
    plan, _, admission, completed, analysis = _admitted()
    oos = assess_oos(
        plan,
        _completed_observation(plan, completed),
        _analysis_observation(plan, analysis),
    )
    report = aggregate_validation_report(plan, (oos, admission))

    assert oos.outcome == "FAIL"
    assert oos.reason_codes == ("OOS_THRESHOLD_NOT_MET",)
    assert isinstance(oos.evidence, CompletedCaseEvidence)
    assert oos.threshold_evaluation.observed == "-0.1"
    assert oos.threshold_evaluation.passed is False
    assert report is not None
    assert report.result == "rejected"
    assert tuple(result.case_type for result in report.case_results) == (
        "evidence_integrity",
        "out_of_sample",
    )
    assert not hasattr(report, "candidate_ref")
    assert not hasattr(report, "deployment_authorized")
    assert not hasattr(report, "shadow_ready")


def test_completed_only_analysis_checks_case_source_profile_grade_then_metric() -> None:
    plan, _, _, completed, analysis = _admitted()

    wrong_context = assess_oos(
        plan,
        _completed_observation(plan, completed, context="case:other"),
        _analysis_observation(plan, analysis),
    )
    assert (wrong_context.outcome, wrong_context.reason_codes) == (
        "FAILED",
        ("ANALYSIS_LINK_INVALID",),
    )

    forged = deepcopy(analysis)
    forged["source_execution_result_hash"] = "sha256:" + "b" * 64
    forged["result_grade"] = "decision_grade"
    forged.pop("simple_period_return")
    precedence = assess_oos(
        plan,
        _completed_observation(plan, completed),
        _analysis_observation(plan, forged),
    )
    assert precedence.reason_codes == ("ANALYSIS_LINK_INVALID",)

    unaccepted_completed = deepcopy(completed)
    unaccepted_analysis = deepcopy(analysis)
    unaccepted_completed["result_grade"] = "decision_grade"
    unaccepted_analysis["result_grade"] = "decision_grade"
    unaccepted = assess_oos(
        plan,
        _completed_observation(plan, unaccepted_completed),
        _analysis_observation(plan, unaccepted_analysis),
    )
    assert (unaccepted.outcome, unaccepted.reason_codes) == (
        "FAILED",
        ("RESULT_GRADE_UNACCEPTED",),
    )

    missing_metric = deepcopy(analysis)
    missing_metric.pop("simple_period_return")
    missing = assess_oos(
        plan,
        _completed_observation(plan, completed),
        _analysis_observation(plan, missing_metric),
    )
    assert (missing.outcome, missing.reason_codes) == (
        "INCONCLUSIVE",
        ("METRIC_MISSING_OR_INSUFFICIENT",),
    )
    assert missing.evidence.metric_value is None
    assert missing.threshold_evaluation is None

    low_trade_count = deepcopy(analysis)
    low_trade_count["trade_count"] = 0
    low = assess_oos(
        plan,
        _completed_observation(plan, completed),
        _analysis_observation(plan, low_trade_count),
    )
    assert (low.outcome, low.reason_codes) == (
        "INCONCLUSIVE",
        ("METRIC_MISSING_OR_INSUFFICIENT",),
    )
    assert low.threshold_evaluation is None


@pytest.mark.parametrize(
    ("case_id", "outcome", "reason"),
    [
        ("terminal_blocked", "BLOCKED", "BACKTEST_TERMINAL_BLOCKED"),
        ("terminal_failed", "FAILED", "BACKTEST_TERMINAL_FAILED"),
        ("terminal_cancelled", "INCONCLUSIVE", "BACKTEST_TERMINAL_CANCELLED"),
    ],
)
def test_terminal_matrix_has_no_fabricated_metrics(
    case_id: str, outcome: str, reason: str
) -> None:
    plan, _, _, _, analysis = _admitted()
    port = InMemoryBacktestConsumerPort()
    fixture = port.case(case_id)
    terminal_ref = port.run(fixture["request_spec"])
    terminal = port.load_terminal(terminal_ref)

    result = assess_oos(
        plan,
        OosObservation(plan, "case:oos", "case:oos", terminal),
        _analysis_observation(plan, analysis),
    )

    assert (result.outcome, result.reason_codes) == (outcome, (reason,))
    assert isinstance(result.evidence, TerminalCaseEvidence)
    assert set(TerminalCaseEvidence.__dataclass_fields__) == {
        "status",
        "durable_evidence_ref",
    }
    for metric in ("result_grade", "simple_period_return", "trade_count"):
        assert not hasattr(result.evidence, metric)
    assert result.threshold_evaluation is None


def test_provider_failure_remains_failure_and_produces_no_report() -> None:
    plan, _, admission, _, _ = _admitted()
    port = InMemoryBacktestConsumerPort()
    fixture = port.case("provider_failure")
    with pytest.raises(PortFailure) as caught:
        port.run(fixture["request_spec"])

    oos = assess_oos(plan, caught.value, None)
    assert oos.outcome == "FAILED"
    assert oos.reason_codes == ("PORT_RETENTION_UNAVAILABLE",)
    assert isinstance(oos.evidence, ProviderFailureEvidence)
    assert not hasattr(oos.evidence, "status")
    assert aggregate_validation_report(plan, (admission, oos)) is None


def test_report_exact_cover_is_deterministic_and_failed_execution_precedes_cover() -> (
    None
):
    plan, _, admission, completed, analysis = _admitted()
    oos = assess_oos(
        plan,
        _completed_observation(plan, completed),
        _analysis_observation(plan, analysis),
    )
    assert aggregate_validation_report(
        plan, (oos, admission)
    ) == aggregate_validation_report(plan, (admission, oos))

    with pytest.raises(ValidationCoreFailure) as caught:
        aggregate_validation_report(plan, (admission, admission))
    assert caught.value.code == "CASE_COVER_INVALID"

    failed = replace(oos, outcome="FAILED", reason_codes=("ANALYSIS_LINK_INVALID",))
    assert aggregate_validation_report(plan, (failed,)) is None


def test_core_imports_no_foundation_research_or_backtest_runtime() -> None:
    source = INTEGRATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(INTEGRATION_PATH))
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
    assert not any(name.startswith("crypto_quant_foundation") for name in imports)
    assert not any(name.startswith("crypto_quant_backtest") for name in imports)
    assert not any(name.startswith("backtest") for name in imports)
    assert "Protocol" not in source
    assert "simulation" not in source.lower()
