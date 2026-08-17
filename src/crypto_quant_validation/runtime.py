from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from crypto_quant_domain import (
    ArtifactEnvelope,
    ArtifactRef,
    canonical_bytes,
    canonical_sha256,
)
from crypto_quant_foundation import (
    FoundationFailure,
    LocalFoundation,
    LogCheckpoint,
    LogEntryRef,
)

from .integration import (
    AnalysisObservation,
    CandidateGraph,
    CaseResult,
    CompletedCaseEvidence,
    OosObservation,
    ProviderFailure,
    ProviderFailureEvidence,
    ResolvedArtifact,
    SampleAdmissionEvidence,
    SampleCaseEvidence,
    TerminalCaseEvidence,
    ThresholdEvaluation,
    ValidationCoreFailure,
    ValidationPlan,
    ValidationPolicy,
    ValidationReport,
    aggregate_validation_report,
    assess_admission,
    assess_oos,
    build_validation_plan,
)
from .ledger import SampleConsumptionLedger
from .sample_consumption import (
    SampleConsumptionRecord,
    assess_untouched_holdout,
    build_snapshot,
)

_ARTIFACT_LOG = "validation.artifacts.v1"
_RESEARCH_ARTIFACT_LOG = "research.artifacts.v1"
_RESEARCH_EXECUTION_LOG = "research.execution.v1"
_SAMPLE_LOG = "validation.sample-consumption.v1"


@dataclass(frozen=True, slots=True)
class PublishedValidationReport:
    validation_plan_ref: ArtifactRef
    validation_report_ref: ArtifactRef


@dataclass(frozen=True, slots=True)
class NoReport:
    validation_plan_ref: ArtifactRef | None
    reason_code: str


class _GraphFailure(ValueError):
    def __init__(self, code: str = "CANDIDATE_PROVENANCE_INVALID") -> None:
        self.code = code
        super().__init__(code)


def _plain(value: object) -> Any:
    try:
        return json.loads(canonical_bytes(value).decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("value must be canonical JSON") from error


def _wire(ref: ArtifactRef) -> dict[str, object]:
    return ref.to_canonical_dict()


def _artifact_ref(value: object, name: str) -> ArtifactRef:
    if type(value) is not ArtifactRef:
        raise TypeError(f"{name} must be an ArtifactRef")
    try:
        normalized = ArtifactRef(
            value.artifact_type,
            value.schema_version,
            value.content_hash,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be canonical") from error
    if normalized != value:
        raise ValueError(f"{name} must be canonical")
    return normalized


def _ref_from_wire(value: object) -> ArtifactRef:
    if type(value) is not dict:
        raise ValueError("artifact ref wire must be an object")
    try:
        ref = ArtifactRef(
            value["artifact_type"],
            value["schema_version"],
            value["content_hash"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("artifact ref wire is invalid") from error
    if value != ref.to_canonical_dict():
        raise ValueError("artifact ref wire is not canonical")
    return ref


def _event_id(log_name: str, ref: ArtifactRef) -> str:
    return canonical_sha256(("artifact-publication-v1", log_name, ref))


def _publish(
    foundation: LocalFoundation, artifact_type: str, payload: dict[str, object]
) -> ArtifactRef:
    envelope = ArtifactEnvelope.create(artifact_type, 1, payload)
    ref = foundation.put(envelope=envelope)
    foundation.append(
        _ARTIFACT_LOG, _event_id(_ARTIFACT_LOG, ref), canonical_bytes(envelope)
    )
    return ref


def _published(
    foundation: LocalFoundation,
    ref: ArtifactRef,
    artifact_type: str,
    log_name: str,
) -> dict[str, Any]:
    stored = foundation.read(ref=ref)
    if (
        stored.envelope.artifact_type != artifact_type
        or stored.envelope.schema_version != 1
        or not any(
            entry.event_id == _event_id(log_name, ref)
            and entry.payload == stored.source_bytes
            for entry in foundation.entries(log_name)
        )
    ):
        raise ValueError("artifact is not published in its owner log")
    payload = _plain(stored.envelope.payload)
    if type(payload) is not dict:
        raise ValueError("artifact payload must be an object")
    return payload


def _sample_entry(payload: bytes) -> tuple[SampleConsumptionRecord, ArtifactRef]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
        envelope = ArtifactEnvelope(
            decoded["artifact_type"],
            decoded["schema_version"],
            decoded["payload"],
            decoded["content_hash"],
        )
        if canonical_bytes(envelope) != payload:
            raise ValueError("sample append is not canonical")
        value = _plain(envelope.payload)
        if (
            envelope.artifact_type != "sample_consumption_append"
            or envelope.schema_version != 1
            or set(value) != {"record", "producer_ref"}
        ):
            raise ValueError("sample append has the wrong shape")
        record_value = value["record"]
        if type(record_value) is not dict or set(record_value) != {
            "dataset_revision",
            "interval_start",
            "interval_end",
            "purpose",
            "consumer_id",
            "consumed_at",
        }:
            raise ValueError("sample record has the wrong shape")
        record = SampleConsumptionRecord(
            record_value["dataset_revision"],
            record_value["interval_start"],
            record_value["interval_end"],
            record_value["purpose"],
            record_value["consumer_id"],
            record_value["consumed_at"],
        )
        return record, _ref_from_wire(value["producer_ref"])
    except (
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError("sample append is invalid") from error


def _sample_entries(
    foundation: LocalFoundation, checkpoint: LogCheckpoint | None = None
) -> tuple[tuple[SampleConsumptionRecord, ArtifactRef, LogEntryRef], ...]:
    return tuple(
        (*_sample_entry(entry.payload), entry.entry_ref)
        for entry in foundation.entries(_SAMPLE_LOG, checkpoint)
    )


def _consumer_id(ref: ArtifactRef) -> str:
    return canonical_sha256(("sample-consumer-v1", ref))


def _required_records(
    entries: tuple[tuple[SampleConsumptionRecord, ArtifactRef, LogEntryRef], ...],
    trial_ref: ArtifactRef,
    selection_ref: ArtifactRef,
    trial: dict[str, Any],
    reservation_at: str,
) -> tuple[SampleConsumptionRecord, ...]:
    data_slice = trial["data_slice"]
    if type(data_slice) is not dict:
        raise ValueError("trial data slice must be an object")

    def expected(ref: ArtifactRef, purpose: str) -> SampleConsumptionRecord:
        return SampleConsumptionRecord(
            data_slice["dataset_revision"],
            data_slice["interval_start"],
            data_slice["interval_end"],
            purpose,
            _consumer_id(ref),
            reservation_at,
        )

    def found(
        ref: ArtifactRef, record: SampleConsumptionRecord
    ) -> SampleConsumptionRecord:
        matches = [
            item
            for item, producer, _ in entries
            if producer == ref
            and item.dataset_revision == record.dataset_revision
            and item.interval_start == record.interval_start
            and item.interval_end == record.interval_end
            and item.purpose == record.purpose
            and item.consumer_id == record.consumer_id
        ]
        return matches[0] if len(matches) == 1 else record

    return (
        found(trial_ref, expected(trial_ref, "discovery")),
        found(selection_ref, expected(selection_ref, "selection")),
    )


def _node(
    foundation: LocalFoundation,
    ref: ArtifactRef,
    artifact_type: str,
    log_name: str,
) -> tuple[ResolvedArtifact, dict[str, Any]]:
    payload = _published(foundation, ref, artifact_type, log_name)
    return ResolvedArtifact(_wire(ref), payload), payload


def _candidate_graph(
    candidate_ref: ArtifactRef,
    foundation: LocalFoundation,
    backtest: object,
    reservation_at: str,
) -> tuple[
    CandidateGraph,
    tuple[tuple[SampleConsumptionRecord, ArtifactRef, LogEntryRef], ...],
]:
    try:
        candidate, candidate_payload = _node(
            foundation,
            candidate_ref,
            "strategy_candidate",
            _RESEARCH_ARTIFACT_LOG,
        )
        family_ref = _ref_from_wire(candidate_payload["candidate_family_ref"])
        selection_ref = _ref_from_wire(candidate_payload["selection_declaration_ref"])
        trial_ref = _ref_from_wire(candidate_payload["selected_trial_declaration_ref"])
        trial_spec_ref = _ref_from_wire(candidate_payload["selected_trial_spec_ref"])
        family, family_payload = _node(
            foundation, family_ref, "candidate_family", _RESEARCH_ARTIFACT_LOG
        )
        manifest_ref = _ref_from_wire(family_payload["execution_manifest_ref"])
        manifest, manifest_payload = _node(
            foundation,
            manifest_ref,
            "experiment_execution_manifest",
            _RESEARCH_EXECUTION_LOG,
        )
        selection, selection_payload = _node(
            foundation,
            selection_ref,
            "selection_declaration",
            _RESEARCH_ARTIFACT_LOG,
        )
        selection_policy_ref = _ref_from_wire(selection_payload["selection_policy_ref"])
        selection_policy, _ = _node(
            foundation,
            selection_policy_ref,
            "selection_policy",
            _RESEARCH_ARTIFACT_LOG,
        )
        trial, trial_payload = _node(
            foundation,
            trial_ref,
            "trial_declaration",
            _RESEARCH_ARTIFACT_LOG,
        )
        trial_spec, _ = _node(
            foundation,
            trial_spec_ref,
            "backtest_trial_spec",
            _RESEARCH_ARTIFACT_LOG,
        )

        outcome_refs = manifest_payload["task_outcome_refs"]
        if type(outcome_refs) is not list:
            raise ValueError("manifest outcomes must be a list")
        outcomes = tuple(
            (
                _ref_from_wire(value),
                *_node(
                    foundation,
                    _ref_from_wire(value),
                    "task_outcome",
                    _RESEARCH_EXECUTION_LOG,
                ),
            )
            for value in outcome_refs
        )
        trial_outcomes = [
            node
            for _, node, payload in outcomes
            if payload.get("task_ref")
            == {"kind": "TRIAL", "task_artifact_ref": _wire(trial_ref)}
        ]
        analysis_outcomes = [
            (node, payload)
            for _, node, payload in outcomes
            if type(payload.get("task_ref")) is dict
            and payload["task_ref"].get("kind") == "ANALYSIS"
            and type(payload.get("witness")) is dict
            and type(payload["witness"].get("analysis_derivation")) is dict
            and payload["witness"]["analysis_derivation"].get("analysis_ref")
            == candidate_payload["selected_analysis_ref"]
        ]
        if len(trial_outcomes) != 1 or len(analysis_outcomes) != 1:
            raise ValueError("candidate outcomes are ambiguous")
        analysis_outcome, analysis_outcome_payload = analysis_outcomes[0]
        analysis_task_ref = _ref_from_wire(
            analysis_outcome_payload["task_ref"]["task_artifact_ref"]
        )
        analysis_task, _ = _node(
            foundation,
            analysis_task_ref,
            "analysis_task",
            _RESEARCH_ARTIFACT_LOG,
        )
        entries = _sample_entries(foundation)
        required_records = _required_records(
            entries,
            trial_ref,
            selection_ref,
            trial_payload,
            reservation_at,
        )
    except FoundationFailure:
        raise
    except Exception as error:
        raise _GraphFailure() from error

    try:
        completed = backtest.load_completed(
            _plain(candidate_payload["selected_publication_ref"])
        )
        analysis = backtest.load_analysis(
            _plain(candidate_payload["selected_analysis_ref"])
        )
    except FoundationFailure:
        raise
    except Exception as error:
        raise _GraphFailure(_failure_code(error)) from error

    try:
        return (
            CandidateGraph(
                candidate,
                family,
                manifest,
                selection,
                selection_policy,
                trial,
                trial_spec,
                trial_outcomes[0],
                analysis_task,
                analysis_outcome,
                completed,
                analysis,
                required_records,
            ),
            entries,
        )
    except (TypeError, ValueError) as error:
        raise _GraphFailure() from error


def _checkpoint_wire(checkpoint: LogCheckpoint) -> dict[str, object]:
    return {
        "log_name": checkpoint.log_name,
        "as_of": checkpoint.as_of,
        "upper_log_sequence": checkpoint.upper_log_sequence,
        "head_receipt_hash": checkpoint.head_receipt_hash,
    }


def _entry_ref_wire(ref: LogEntryRef) -> dict[str, object]:
    return {
        "log_name": ref.log_name,
        "log_sequence": ref.log_sequence,
        "receipt_hash": ref.receipt_hash,
    }


def _preflight_admission(
    candidate_ref: ArtifactRef,
    policy: ValidationPolicy,
    graph: CandidateGraph,
    entries: tuple[tuple[SampleConsumptionRecord, ArtifactRef, LogEntryRef], ...],
    reservation_at: str,
) -> CaseResult:
    records = tuple(record for record, _, _ in entries)
    snapshot = build_snapshot(
        records,
        as_of=max((record.consumed_at for record in records), default=reservation_at),
    )
    checkpoint = {
        "log_name": _SAMPLE_LOG,
        "as_of": snapshot.as_of,
        "upper_log_sequence": len(entries),
        "head_receipt_hash": entries[-1][2].receipt_hash if entries else None,
    }
    plan = build_validation_plan(_wire(candidate_ref), _wire(candidate_ref), policy)
    integrity = assess_untouched_holdout(
        snapshot,
        dataset_revision=plan.holdout.dataset_revision,
        interval_start=plan.holdout.interval_start,
        interval_end=plan.holdout.interval_end,
    )
    return assess_admission(
        plan,
        graph,
        SampleAdmissionEvidence(_wire(candidate_ref), checkpoint, snapshot, integrity),
    )


def _snapshot_evidence(
    foundation: LocalFoundation,
    snapshot_ref: ArtifactRef,
    assessment_ref: ArtifactRef,
    policy: ValidationPolicy,
) -> SampleAdmissionEvidence:
    snapshot_payload = _published(
        foundation,
        snapshot_ref,
        "sample_consumption_ledger_snapshot",
        _ARTIFACT_LOG,
    )
    try:
        raw_checkpoint = snapshot_payload["checkpoint"]
        if type(raw_checkpoint) is not dict or set(raw_checkpoint) != {
            "log_name",
            "as_of",
            "upper_log_sequence",
            "head_receipt_hash",
        }:
            raise ValueError("snapshot checkpoint is invalid")
        checkpoint = LogCheckpoint(
            raw_checkpoint["log_name"],
            raw_checkpoint["as_of"],
            raw_checkpoint["upper_log_sequence"],
            raw_checkpoint["head_receipt_hash"],
        )
        entries = _sample_entries(foundation, checkpoint)
        snapshot = build_snapshot(
            tuple(record for record, _, _ in entries), as_of=checkpoint.as_of
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("snapshot prefix is invalid") from error

    integrity = assess_untouched_holdout(
        snapshot,
        dataset_revision=policy.holdout.dataset_revision,
        interval_start=policy.holdout.interval_start,
        interval_end=policy.holdout.interval_end,
    )
    refs_by_record: dict[SampleConsumptionRecord, list[LogEntryRef]] = {}
    for record, _, entry_ref in entries:
        refs_by_record.setdefault(record, []).append(entry_ref)
    conflicting_refs = [
        _entry_ref_wire(refs_by_record[record].pop(0))
        for record in integrity.conflicting_records
    ]
    assessment = _published(
        foundation,
        assessment_ref,
        "sample_integrity_assessment",
        _ARTIFACT_LOG,
    )
    expected_assessment = {
        "snapshot_ref": _wire(snapshot_ref),
        "dataset_revision": policy.holdout.dataset_revision,
        "interval_start": policy.holdout.interval_start,
        "interval_end": policy.holdout.interval_end,
        "untouched": integrity.untouched,
        "conflicting_append_entry_refs": conflicting_refs,
    }
    return SampleAdmissionEvidence(
        _wire(snapshot_ref),
        _checkpoint_wire(checkpoint),
        snapshot,
        integrity,
        assessment != expected_assessment,
    )


def _plan_payload(plan: ValidationPlan) -> dict[str, object]:
    return {
        "candidate_ref": _plain(plan.candidate_ref),
        "sample_consumption_snapshot_ref": _plain(plan.sample_consumption_snapshot_ref),
        "accepted_backtest_grades": list(plan.accepted_backtest_grades),
        "accepted_metric_profile_refs": [
            _plain(ref) for ref in plan.accepted_metric_profile_refs
        ],
        "holdout": {
            "market_bundle_ref": _plain(plan.holdout.market_bundle_ref),
            "dataset_revision": plan.holdout.dataset_revision,
            "interval_start": plan.holdout.interval_start,
            "interval_end": plan.holdout.interval_end,
            "role": plan.holdout.role,
            "selection_observed": plan.holdout.selection_observed,
        },
        "oos_rule": {
            "metric_profile_ref": _plain(plan.oos_rule.metric_profile_ref),
            "metric_key": plan.oos_rule.metric_key,
            "unit": plan.oos_rule.unit,
            "operator": plan.oos_rule.operator,
            "threshold": plan.oos_rule.threshold,
            "minimum_trade_count": plan.oos_rule.minimum_trade_count,
        },
        "decision_rule": {
            "required_case_types": list(plan.decision_rule.required_case_types),
            "required_fail": plan.decision_rule.required_fail,
            "blocked_or_inconclusive": plan.decision_rule.blocked_or_inconclusive,
            "failed_execution": plan.decision_rule.failed_execution,
        },
    }


def _artifact_log_payloads(
    foundation: LocalFoundation,
) -> tuple[tuple[ArtifactRef, dict[str, object]], ...]:
    artifacts: list[tuple[ArtifactRef, dict[str, object]]] = []
    for entry in foundation.entries(_ARTIFACT_LOG):
        try:
            decoded = json.loads(entry.payload.decode("utf-8"))
            envelope = ArtifactEnvelope(
                decoded["artifact_type"],
                decoded["schema_version"],
                decoded["payload"],
                decoded["content_hash"],
            )
            if canonical_bytes(envelope) != entry.payload:
                continue
            ref = ArtifactRef.from_envelope(envelope)
            if entry.event_id != _event_id(_ARTIFACT_LOG, ref):
                continue
            payload = _plain(envelope.payload)
            if type(payload) is dict:
                artifacts.append((ref, payload))
        except (KeyError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            continue
    return tuple(artifacts)


def _existing_plan(
    foundation: LocalFoundation, candidate_ref: ArtifactRef, policy: ValidationPolicy
) -> tuple[ArtifactRef, dict[str, object]] | None:
    expected = _plan_payload(
        build_validation_plan(_wire(candidate_ref), _wire(candidate_ref), policy)
    )
    matches: list[tuple[ArtifactRef, dict[str, object]]] = []
    for ref, payload in _artifact_log_payloads(foundation):
        if ref.artifact_type != "validation_plan":
            continue
        candidate = payload.get("candidate_ref")
        snapshot = payload.get("sample_consumption_snapshot_ref")
        if candidate != expected["candidate_ref"] or type(snapshot) is not dict:
            continue
        candidate_expected = dict(expected)
        candidate_expected["sample_consumption_snapshot_ref"] = snapshot
        if payload == candidate_expected:
            matches.append((ref, payload))
    if not matches:
        return None
    ref, payload = matches[-1]
    if _published(foundation, ref, "validation_plan", _ARTIFACT_LOG) != payload:
        raise _GraphFailure()
    return ref, payload


def _existing_report(
    foundation: LocalFoundation, plan_ref: ArtifactRef
) -> PublishedValidationReport | None:
    matches = [
        (ref, payload)
        for ref, payload in _artifact_log_payloads(foundation)
        if ref.artifact_type == "validation_report"
        and payload.get("validation_plan_ref") == _wire(plan_ref)
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise _GraphFailure()
    report_ref, payload = matches[0]
    if (
        set(payload)
        != {
            "validation_plan_ref",
            "result",
            "case_result_refs",
            "threshold_evaluations",
            "sample_integrity_ref",
            "limitations",
        }
        or payload["result"] not in {"supported", "rejected", "inconclusive"}
        or _published(foundation, report_ref, "validation_report", _ARTIFACT_LOG)
        != payload
    ):
        raise _GraphFailure()
    return PublishedValidationReport(plan_ref, report_ref)


def _record_payload(record: SampleConsumptionRecord) -> dict[str, str]:
    return {
        "dataset_revision": record.dataset_revision,
        "interval_start": record.interval_start,
        "interval_end": record.interval_end,
        "purpose": record.purpose,
        "consumer_id": record.consumer_id,
        "consumed_at": record.consumed_at,
    }


def _evidence_payload(value: object) -> object:
    if type(value) is SampleCaseEvidence:
        return {
            "snapshot_ref": _plain(value.snapshot_ref),
            "untouched": value.untouched,
            "conflicting_records": [
                _record_payload(record) for record in value.conflicting_records
            ],
        }
    if type(value) is TerminalCaseEvidence:
        return {
            "status": value.status,
            "durable_evidence_ref": _plain(value.durable_evidence_ref),
        }
    if type(value) is ProviderFailureEvidence:
        return {"code": value.code}
    if type(value) is CompletedCaseEvidence:
        return {
            "publication_ref": _plain(value.publication_ref),
            "analysis_ref": _plain(value.analysis_ref),
            "metric_profile_ref": _plain(value.metric_profile_ref),
            "source_execution_result_hash": value.source_execution_result_hash,
            "result_grade": value.result_grade,
            "metric_key": value.metric_key,
            "metric_value": value.metric_value,
            "trade_count": value.trade_count,
        }
    return None


def _case_result_payload(
    case_ref: ArtifactRef, result: CaseResult
) -> dict[str, object]:
    return {
        "case_ref": _wire(case_ref),
        "outcome": result.outcome,
        "reason_codes": list(result.reason_codes),
        "limitations": list(result.limitations),
        "evidence": _evidence_payload(result.evidence),
    }


def _threshold_payload(value: ThresholdEvaluation) -> dict[str, object]:
    return {
        "metric_key": value.metric_key,
        "observed": value.observed,
        "operator": value.operator,
        "threshold": value.threshold,
        "passed": value.passed,
        "trade_count": value.trade_count,
        "minimum_trade_count": value.minimum_trade_count,
    }


def _report_payload(
    report: ValidationReport,
    plan_ref: ArtifactRef,
    case_result_refs: tuple[ArtifactRef, ArtifactRef],
    assessment_ref: ArtifactRef,
) -> dict[str, object]:
    return {
        "validation_plan_ref": _wire(plan_ref),
        "result": report.result,
        "case_result_refs": [_wire(ref) for ref in case_result_refs],
        "threshold_evaluations": [
            _threshold_payload(value) for value in report.threshold_evaluations
        ],
        "sample_integrity_ref": _wire(assessment_ref),
        "limitations": list(report.limitations),
    }


def _failure_code(error: Exception) -> str:
    code = getattr(error, "code", None)
    if type(code) is str and code:
        return code
    value = getattr(code, "value", None)
    return value if type(value) is str and value else "BACKTEST_OPERATION_FAILED"


def _require_backtest(backtest: object) -> None:
    if not all(
        callable(getattr(backtest, name, None))
        for name in (
            "run",
            "derive",
            "load_completed",
            "load_terminal",
            "load_analysis",
        )
    ):
        raise TypeError("backtest must expose the frozen BT-PORT operations")


def _oos_result(
    plan: ValidationPlan,
    case_ref: ArtifactRef,
    request_spec: dict[str, object],
    backtest: object,
) -> CaseResult:
    case_wire = _wire(case_ref)
    request = dict(request_spec)
    request["experiment_id"] = canonical_bytes(case_ref).decode("utf-8")
    try:
        run_ref = backtest.run(request)
    except FoundationFailure:
        raise
    except Exception as error:  # noqa: BLE001 - frozen provider boundary
        return assess_oos(plan, ProviderFailure(_failure_code(error)), None)
    try:
        completed = backtest.load_completed(run_ref)
    except FoundationFailure:
        raise
    except Exception as error:  # noqa: BLE001 - frozen provider boundary
        if _failure_code(error) != "PORT_REF_TYPE_MISMATCH":
            return assess_oos(plan, ProviderFailure(_failure_code(error)), None)
        try:
            terminal = backtest.load_terminal(run_ref)
        except FoundationFailure:
            raise
        except Exception as terminal_error:  # noqa: BLE001 - frozen provider boundary
            return assess_oos(
                plan, ProviderFailure(_failure_code(terminal_error)), None
            )
        return assess_oos(
            plan, OosObservation(plan, case_wire, case_wire, terminal), None
        )

    observation = OosObservation(plan, case_wire, case_wire, completed)
    try:
        analysis_ref = backtest.derive(
            run_ref, _plain(plan.oos_rule.metric_profile_ref)
        )
        analysis = backtest.load_analysis(analysis_ref)
    except FoundationFailure:
        raise
    except Exception as error:  # noqa: BLE001 - frozen provider boundary
        return assess_oos(plan, observation, ProviderFailure(_failure_code(error)))
    return assess_oos(plan, observation, AnalysisObservation(plan, case_wire, analysis))


def _blocked_oos(plan: ValidationPlan, admission: CaseResult) -> CaseResult:
    return CaseResult(
        plan,
        "out_of_sample",
        "FAILED" if admission.outcome == "FAILED" else "BLOCKED",
        admission.reason_codes,
        admission.limitations,
        None,
    )


def _no_report_reason(results: tuple[CaseResult, ...]) -> str:
    for result in results:
        if result.outcome == "FAILED" and result.reason_codes:
            return result.reason_codes[0]
    return "NO_REPORT"


def validate_candidate(
    candidate_ref: ArtifactRef,
    policy: ValidationPolicy,
    request_spec: dict[str, object],
    reservation_at: str,
    foundation: LocalFoundation,
    sample_ledger: SampleConsumptionLedger,
    backtest: object,
    *,
    fresh: bool = False,
) -> PublishedValidationReport | NoReport:
    """Publish one fixture-backed Validation result without a provider seam."""

    candidate_ref = _artifact_ref(candidate_ref, "candidate_ref")
    if type(policy) is not ValidationPolicy:
        raise TypeError("policy must be a ValidationPolicy")
    policy = ValidationPolicy(
        policy.accepted_backtest_grades,
        policy.accepted_metric_profile_refs,
        policy.holdout,
        policy.oos_rule,
        policy.decision_rule,
    )
    request_spec = _plain(request_spec)
    if type(request_spec) is not dict:
        raise ValueError("request_spec must be an object")
    reservation_at = build_snapshot((), as_of=reservation_at).as_of
    if type(foundation) is not LocalFoundation:
        raise TypeError("foundation must be a LocalFoundation")
    if type(sample_ledger) is not SampleConsumptionLedger:
        raise TypeError("sample_ledger must be a SampleConsumptionLedger")
    if type(fresh) is not bool:
        raise TypeError("fresh must be a bool")
    _require_backtest(backtest)

    existing: tuple[ArtifactRef, dict[str, object]] | None = None
    plan_ref: ArtifactRef | None = None
    plan: ValidationPlan | None = None
    snapshot_ref: ArtifactRef | None = None
    preflight: CaseResult | None = None
    try:
        existing = None if fresh else _existing_plan(foundation, candidate_ref, policy)
        if existing is not None:
            plan_ref, plan_payload = existing
            published = _existing_report(foundation, plan_ref)
            if published is not None:
                return published
            snapshot_ref = _ref_from_wire(
                plan_payload["sample_consumption_snapshot_ref"]
            )
            plan = build_validation_plan(
                _wire(candidate_ref), _wire(snapshot_ref), policy
            )
            if _plan_payload(plan) != plan_payload:
                raise _GraphFailure()
            graph, _ = _candidate_graph(
                candidate_ref, foundation, backtest, reservation_at
            )
        else:
            graph, entries = _candidate_graph(
                candidate_ref, foundation, backtest, reservation_at
            )
            preflight = _preflight_admission(
                candidate_ref, policy, graph, entries, reservation_at
            )
    except FoundationFailure:
        raise
    except _GraphFailure as error:
        return NoReport(None, error.code)
    except (KeyError, TypeError, ValueError, ValidationCoreFailure):
        return NoReport(None, "CANDIDATE_PROVENANCE_INVALID")

    if existing is None:
        if preflight is None:
            raise AssertionError("preflight is missing")
        if preflight.reason_codes == ("CANDIDATE_PROVENANCE_INVALID",):
            return NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
        snapshot_ref = sample_ledger.freeze_snapshot()
        plan = build_validation_plan(_wire(candidate_ref), _wire(snapshot_ref), policy)
        plan_ref = _publish(foundation, "validation_plan", _plan_payload(plan))

    if plan_ref is None or plan is None or snapshot_ref is None:
        raise AssertionError("validation plan is missing")
    assessment_ref = sample_ledger.assess_holdout(snapshot_ref, plan.holdout)
    evidence = _snapshot_evidence(foundation, snapshot_ref, assessment_ref, policy)

    evidence_case_ref = _publish(
        foundation,
        "validation_case",
        {"validation_plan_ref": _wire(plan_ref), "case_type": "evidence_integrity"},
    )
    admission = assess_admission(plan, graph, evidence)
    evidence_result_ref = _publish(
        foundation,
        "validation_case_result",
        _case_result_payload(evidence_case_ref, admission),
    )

    oos_case_ref = _publish(
        foundation,
        "validation_case",
        {"validation_plan_ref": _wire(plan_ref), "case_type": "out_of_sample"},
    )
    if admission.outcome == "PASS":
        sample_ledger.reserve(
            SampleConsumptionRecord(
                plan.holdout.dataset_revision,
                plan.holdout.interval_start,
                plan.holdout.interval_end,
                "validation",
                _consumer_id(oos_case_ref),
                reservation_at,
            ),
            oos_case_ref,
        )
        oos = _oos_result(plan, oos_case_ref, request_spec, backtest)
    else:
        oos = _blocked_oos(plan, admission)
    oos_result_ref = _publish(
        foundation,
        "validation_case_result",
        _case_result_payload(oos_case_ref, oos),
    )

    report = aggregate_validation_report(plan, (admission, oos))
    if report is None:
        return NoReport(plan_ref, _no_report_reason((admission, oos)))
    report_ref = _publish(
        foundation,
        "validation_report",
        _report_payload(
            report,
            plan_ref,
            (evidence_result_ref, oos_result_ref),
            assessment_ref,
        ),
    )
    return PublishedValidationReport(plan_ref, report_ref)


__all__ = ["NoReport", "PublishedValidationReport", "validate_candidate"]
