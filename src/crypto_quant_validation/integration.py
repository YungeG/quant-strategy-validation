from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from .sample_consumption import (
    SampleConsumptionRecord,
    SampleConsumptionSnapshot,
    SampleIntegrityResult,
    assess_untouched_holdout,
)

FAILURE_PRECEDENCE = (
    "VALIDATION_PLAN_INVALID",
    "CANDIDATE_PROVENANCE_INVALID",
    "SAMPLE_LEDGER_CONFLICT",
    "SAMPLE_RESERVATION_COVERAGE_MISSING",
    "HOLDOUT_CONTAMINATED",
    "BACKTEST_TERMINAL_BLOCKED",
    "BACKTEST_TERMINAL_FAILED",
    "BACKTEST_TERMINAL_CANCELLED",
    "ANALYSIS_LINK_INVALID",
    "RESULT_GRADE_UNACCEPTED",
    "METRIC_MISSING_OR_INSUFFICIENT",
    "CASE_COVER_INVALID",
)

_CASE_TYPES = ("evidence_integrity", "out_of_sample")
_CASE_OUTCOMES = frozenset({"PASS", "FAIL", "INCONCLUSIVE", "BLOCKED", "FAILED"})
_REPORT_RESULTS = frozenset({"supported", "rejected", "inconclusive"})
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z")
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z\Z")

_CANDIDATE_FIELDS = {
    "candidate_family_ref",
    "selection_declaration_ref",
    "selected_trial_declaration_ref",
    "selected_trial_spec_ref",
    "selected_publication_ref",
    "selected_analysis_ref",
    "selection_rank",
    "validated",
}
_FAMILY_FIELDS = {"experiment_ref", "execution_manifest_ref"}
_MANIFEST_FIELDS = {"experiment_ref", "task_outcome_refs"}
_SELECTION_DECLARATION_FIELDS = {
    "experiment_ref",
    "selection_policy_ref",
    "universe_kind",
    "declared_by_ref",
}
_SELECTION_POLICY_FIELDS = {
    "metric_profile_ref",
    "eligible_trial_statuses",
    "accepted_backtest_grades",
    "hard_filters",
    "ordering",
    "max_selections",
    "tie_break",
}
_TRIAL_DECLARATION_FIELDS = {
    "experiment_ref",
    "parameter_values",
    "data_slice",
    "scenario_ref",
    "seed",
    "backtest_template_ref",
    "model_input_bindings",
}
_TRIAL_SPEC_FIELDS = {
    "trial_declaration_ref",
    "resolved_model_refs",
    "backtest_request_ref",
}
_ANALYSIS_TASK_FIELDS = {
    "experiment_ref",
    "trial_declaration_ref",
    "metric_profile_ref",
}
_TASK_OUTCOME_FIELDS = {"task_ref", "state", "witness"}
_COMPLETED_FIELDS = {
    "publication_ref",
    "semantic_run_id",
    "execution_result_hash",
    "result_grade",
}
_ANALYSIS_BASE_FIELDS = {
    "analysis_ref",
    "metric_profile_ref",
    "source_publication_ref",
    "source_execution_result_hash",
    "trade_count",
    "result_grade",
}
_TERMINAL_FIELDS = {"status", "durable_evidence_ref"}
_FEATURE_RECIPE_FIELDS = {
    "feature_key",
    "feature_code_hash",
    "feature_schema_hash",
    "input_names",
}
_TRAINER_RECIPE_FIELDS = {
    "trainer_key",
    "training_code_hash",
    "model_key",
    "hyperparameters",
}
_MODEL_BUILD_PLAN_FIELDS = {
    "feature_recipe_ref",
    "trainer_recipe_ref",
    "training_slice",
    "seed",
}
_FEATURE_BUILD_TASK_FIELDS = {"experiment_ref", "model_build_plan_ref"}
_MODEL_TRAINING_TASK_FIELDS = {
    "experiment_ref",
    "model_build_plan_ref",
    "feature_build_task_ref",
}
_FEATURE_MANIFEST_FIELDS = {
    "model_build_plan_ref",
    "dataset_revision",
    "interval_start",
    "interval_end",
    "feature_schema_hash",
    "training_data_hash",
    "row_count",
}
_MODEL_EVIDENCE_FIELDS = {
    "model_build_plan_ref",
    "feature_dataset_manifest_ref",
    "model_artifact",
}
_MODEL_ARTIFACT_FIELDS = {
    "type",
    "schema_version",
    "model_key",
    "model_hash",
    "training_data_hash",
    "training_start",
    "training_end",
    "training_code_hash",
    "feature_schema_hash",
    "available_at",
    "revision_id",
    "supersedes_revision_id",
    "artifact_ref_hash",
}
_MODEL_BINDING_FIELDS = {
    "type",
    "schema_version",
    "strategy_id",
    "input_name",
    "model_key",
    "timeline_hash",
    "artifact_ref_hash",
}


class ValidationCoreFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        if code not in FAILURE_PRECEDENCE:
            raise ValueError(f"unknown validation failure: {code}")
        self.code = code
        super().__init__(code)


def _require_nonempty_str(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _wire_key(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("value must be canonical JSON data") from error


def _require_ref(value: object, name: str) -> object:
    if type(value) is str:
        return _require_nonempty_str(value, name)
    if type(value) is not dict or not value:
        raise ValueError(f"{name} must be an opaque reference wire value")
    _wire_key(value)
    return value


def _require_ref_tuple(value: object, name: str) -> tuple[object, ...]:
    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a nonempty tuple")
    refs = tuple(_require_ref(item, name) for item in value)
    keys = tuple(_wire_key(item) for item in refs)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} must be unique")
    return tuple(item for _, item in sorted(zip(keys, refs), key=lambda pair: pair[0]))


def _require_string_tuple(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a nonempty tuple")
    values = tuple(_require_nonempty_str(item, name) for item in value)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    return values


def _canonical_decimal(value: object, name: str) -> str:
    value = _require_nonempty_str(value, name)
    if value == "-0" or _DECIMAL.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical decimal string")
    try:
        Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{name} must be a canonical decimal string") from error
    return value


def _canonical_sample_record(value: object) -> SampleConsumptionRecord:
    if type(value) is not SampleConsumptionRecord:
        raise ValueError("sample records must be SampleConsumptionRecord values")
    try:
        normalized = SampleConsumptionRecord(
            value.dataset_revision,
            value.interval_start,
            value.interval_end,
            value.purpose,
            value.consumer_id,
            value.consumed_at,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "sample records must contain canonical six-field values"
        ) from error
    if normalized != value:
        raise ValueError("sample records must contain canonical six-field values")
    return normalized


@dataclass(frozen=True, slots=True)
class Holdout:
    market_bundle_ref: object
    dataset_revision: str
    interval_start: str
    interval_end: str
    role: str
    selection_observed: bool

    def __post_init__(self) -> None:
        _require_ref(self.market_bundle_ref, "market_bundle_ref")
        _require_nonempty_str(self.dataset_revision, "dataset_revision")
        for name in ("interval_start", "interval_end"):
            value = getattr(self, name)
            if type(value) is not str or _UTC.fullmatch(value) is None:
                raise ValueError(f"{name} must be a canonical UTC instant")
            try:
                datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError as error:
                raise ValueError(f"{name} must be a canonical UTC instant") from error
        if self.interval_start >= self.interval_end:
            raise ValueError("interval_start must be before interval_end")
        if self.role != "HOLDOUT":
            raise ValueError("holdout role must be HOLDOUT")
        if type(self.selection_observed) is not bool or self.selection_observed:
            raise ValueError(
                "holdout must be precommitted before selection observation"
            )


@dataclass(frozen=True, slots=True)
class OosRule:
    metric_profile_ref: object
    metric_key: str
    unit: str
    operator: str
    threshold: str
    minimum_trade_count: int

    def __post_init__(self) -> None:
        _require_ref(self.metric_profile_ref, "metric_profile_ref")
        if self.metric_key != "simple_period_return":
            raise ValueError("metric_key is not supported in v1")
        if self.unit != "fraction":
            raise ValueError("unit is not supported in v1")
        if self.operator != "gte":
            raise ValueError("operator is not supported in v1")
        _canonical_decimal(self.threshold, "threshold")
        if type(self.minimum_trade_count) is not int or self.minimum_trade_count < 0:
            raise ValueError("minimum_trade_count must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class DecisionRule:
    required_case_types: tuple[str, ...] = _CASE_TYPES
    required_fail: str = "rejected"
    blocked_or_inconclusive: str = "inconclusive"
    failed_execution: str = "no_report"

    def __post_init__(self) -> None:
        if self.required_case_types != _CASE_TYPES:
            raise ValueError("required_case_types must be the v1 exact cover")
        if self.required_fail != "rejected":
            raise ValueError("required_fail must be rejected")
        if self.blocked_or_inconclusive != "inconclusive":
            raise ValueError("blocked_or_inconclusive must be inconclusive")
        if self.failed_execution != "no_report":
            raise ValueError("failed_execution must be no_report")


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    accepted_backtest_grades: tuple[str, ...]
    accepted_metric_profile_refs: tuple[object, ...]
    holdout: Holdout
    oos_rule: OosRule
    decision_rule: DecisionRule = DecisionRule()

    def __post_init__(self) -> None:
        grades = _require_string_tuple(
            self.accepted_backtest_grades, "accepted_backtest_grades"
        )
        if grades != ("development",):
            raise ValueError("v1 accepts only the development Backtest grade")
        refs = _require_ref_tuple(
            self.accepted_metric_profile_refs, "accepted_metric_profile_refs"
        )
        object.__setattr__(self, "accepted_metric_profile_refs", refs)
        try:
            if type(self.holdout) is not Holdout:
                raise ValueError("holdout must be a Holdout")
            holdout = Holdout(
                self.holdout.market_bundle_ref,
                self.holdout.dataset_revision,
                self.holdout.interval_start,
                self.holdout.interval_end,
                self.holdout.role,
                self.holdout.selection_observed,
            )
            if type(self.oos_rule) is not OosRule:
                raise ValueError("oos_rule must be an OosRule")
            oos_rule = OosRule(
                self.oos_rule.metric_profile_ref,
                self.oos_rule.metric_key,
                self.oos_rule.unit,
                self.oos_rule.operator,
                self.oos_rule.threshold,
                self.oos_rule.minimum_trade_count,
            )
            if type(self.decision_rule) is not DecisionRule:
                raise ValueError("decision_rule must be a DecisionRule")
            decision_rule = DecisionRule(
                self.decision_rule.required_case_types,
                self.decision_rule.required_fail,
                self.decision_rule.blocked_or_inconclusive,
                self.decision_rule.failed_execution,
            )
        except AttributeError as error:
            raise ValueError("policy must contain canonical fields") from error
        object.__setattr__(self, "holdout", holdout)
        object.__setattr__(self, "oos_rule", oos_rule)
        object.__setattr__(self, "decision_rule", decision_rule)
        if _wire_key(oos_rule.metric_profile_ref) not in {
            _wire_key(ref) for ref in refs
        }:
            raise ValueError("OOS metric profile must be accepted by the plan")


@dataclass(frozen=True, slots=True)
class ValidationPlan:
    candidate_ref: object
    sample_consumption_snapshot_ref: object
    accepted_backtest_grades: tuple[str, ...]
    accepted_metric_profile_refs: tuple[object, ...]
    holdout: Holdout
    oos_rule: OosRule
    decision_rule: DecisionRule

    def __post_init__(self) -> None:
        _require_ref(self.candidate_ref, "candidate_ref")
        _require_ref(
            self.sample_consumption_snapshot_ref, "sample_consumption_snapshot_ref"
        )
        policy = ValidationPolicy(
            self.accepted_backtest_grades,
            self.accepted_metric_profile_refs,
            self.holdout,
            self.oos_rule,
            self.decision_rule,
        )
        object.__setattr__(
            self, "accepted_backtest_grades", policy.accepted_backtest_grades
        )
        object.__setattr__(
            self, "accepted_metric_profile_refs", policy.accepted_metric_profile_refs
        )
        object.__setattr__(self, "holdout", policy.holdout)
        object.__setattr__(self, "oos_rule", policy.oos_rule)
        object.__setattr__(self, "decision_rule", policy.decision_rule)


@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    ref: object
    payload: dict[str, Any] | None

    def __post_init__(self) -> None:
        _require_ref(self.ref, "artifact ref")
        if self.payload is not None and type(self.payload) is not dict:
            raise ValueError("resolved payload must be a dict or None")
        object.__setattr__(self, "payload", deepcopy(self.payload))


@dataclass(frozen=True, slots=True)
class ModelBuildGraph:
    feature_recipe: ResolvedArtifact
    trainer_recipe: ResolvedArtifact
    model_build_plan: ResolvedArtifact
    feature_build_task: ResolvedArtifact
    model_training_task: ResolvedArtifact
    feature_dataset_manifest: ResolvedArtifact
    model_build_evidence: ResolvedArtifact
    feature_build_outcome: ResolvedArtifact
    model_training_outcome: ResolvedArtifact

    def __post_init__(self) -> None:
        for name in (
            "feature_recipe",
            "trainer_recipe",
            "model_build_plan",
            "feature_build_task",
            "model_training_task",
            "feature_dataset_manifest",
            "model_build_evidence",
            "feature_build_outcome",
            "model_training_outcome",
        ):
            if type(getattr(self, name)) is not ResolvedArtifact:
                raise ValueError(f"{name} must be a ResolvedArtifact")


@dataclass(frozen=True, slots=True)
class CandidateGraph:
    candidate: ResolvedArtifact
    candidate_family: ResolvedArtifact
    execution_manifest: ResolvedArtifact
    selection_declaration: ResolvedArtifact
    selection_policy: ResolvedArtifact
    selected_trial_declaration: ResolvedArtifact
    selected_trial_spec: ResolvedArtifact
    selected_trial_outcome: ResolvedArtifact
    selected_analysis_task: ResolvedArtifact
    selected_analysis_outcome: ResolvedArtifact
    selected_completed: dict[str, Any] | None
    selected_analysis: dict[str, Any] | None
    required_sample_records: tuple[SampleConsumptionRecord, ...]
    model_build: ModelBuildGraph | None = None

    def __post_init__(self) -> None:
        for name in (
            "candidate",
            "candidate_family",
            "execution_manifest",
            "selection_declaration",
            "selection_policy",
            "selected_trial_declaration",
            "selected_trial_spec",
            "selected_trial_outcome",
            "selected_analysis_task",
            "selected_analysis_outcome",
        ):
            if type(getattr(self, name)) is not ResolvedArtifact:
                raise ValueError(f"{name} must be a ResolvedArtifact")
        for name in ("selected_completed", "selected_analysis"):
            value = getattr(self, name)
            if value is not None and type(value) is not dict:
                raise ValueError(f"{name} must be a dict or None")
            object.__setattr__(self, name, deepcopy(value))
        if type(self.required_sample_records) is not tuple:
            raise ValueError("required_sample_records must be a tuple")
        if self.model_build is not None and type(self.model_build) is not ModelBuildGraph:
            raise ValueError("model_build must be a ModelBuildGraph or None")
        object.__setattr__(
            self,
            "required_sample_records",
            tuple(
                _canonical_sample_record(record)
                for record in self.required_sample_records
            ),
        )


@dataclass(frozen=True, slots=True)
class SampleAdmissionEvidence:
    snapshot_ref: object
    checkpoint: dict[str, Any]
    snapshot: SampleConsumptionSnapshot
    integrity: SampleIntegrityResult
    ledger_conflict: bool = False

    def __post_init__(self) -> None:
        _require_ref(self.snapshot_ref, "snapshot_ref")
        if type(self.checkpoint) is not dict:
            raise ValueError("checkpoint must be a dict")
        if type(self.snapshot) is not SampleConsumptionSnapshot:
            raise ValueError("snapshot must be a SampleConsumptionSnapshot")
        if type(self.integrity) is not SampleIntegrityResult:
            raise ValueError("integrity must be a SampleIntegrityResult")
        if type(self.ledger_conflict) is not bool:
            raise ValueError("ledger_conflict must be a bool")
        object.__setattr__(self, "checkpoint", deepcopy(self.checkpoint))


@dataclass(frozen=True, slots=True)
class OosObservation:
    validation_plan: ValidationPlan
    case_ref: object
    request_context_ref: object
    observation: dict[str, Any]

    def __post_init__(self) -> None:
        _validated_plan(self.validation_plan)
        _require_ref(self.case_ref, "case_ref")
        _require_ref(self.request_context_ref, "request_context_ref")
        if type(self.observation) is not dict:
            raise ValueError("observation must be a dict")
        object.__setattr__(self, "observation", deepcopy(self.observation))


@dataclass(frozen=True, slots=True)
class AnalysisObservation:
    validation_plan: ValidationPlan
    case_ref: object
    analysis: dict[str, Any]

    def __post_init__(self) -> None:
        _validated_plan(self.validation_plan)
        _require_ref(self.case_ref, "case_ref")
        if type(self.analysis) is not dict:
            raise ValueError("analysis must be a dict")
        object.__setattr__(self, "analysis", deepcopy(self.analysis))


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    code: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.code, "provider failure code")


@dataclass(frozen=True, slots=True)
class SampleCaseEvidence:
    snapshot_ref: object
    untouched: bool
    conflicting_records: tuple[SampleConsumptionRecord, ...]


@dataclass(frozen=True, slots=True)
class TerminalCaseEvidence:
    status: str
    durable_evidence_ref: object


@dataclass(frozen=True, slots=True)
class ProviderFailureEvidence:
    code: str


@dataclass(frozen=True, slots=True)
class CompletedCaseEvidence:
    publication_ref: object
    analysis_ref: object
    metric_profile_ref: object
    source_execution_result_hash: str
    result_grade: str
    metric_key: str
    metric_value: str | None
    trade_count: int | None


@dataclass(frozen=True, slots=True)
class ThresholdEvaluation:
    metric_key: str
    observed: str
    operator: str
    threshold: str
    passed: bool
    trade_count: int
    minimum_trade_count: int


@dataclass(frozen=True, slots=True)
class CaseResult:
    validation_plan: ValidationPlan
    case_type: str
    outcome: str
    reason_codes: tuple[str, ...]
    limitations: tuple[str, ...]
    evidence: object | None
    threshold_evaluation: ThresholdEvaluation | None = None

    def __post_init__(self) -> None:
        _validated_plan(self.validation_plan)
        if self.case_type not in _CASE_TYPES:
            raise ValueError("case_type is not supported")
        if self.outcome not in _CASE_OUTCOMES:
            raise ValueError("case outcome is not supported")
        if type(self.reason_codes) is not tuple or any(
            type(code) is not str or not code for code in self.reason_codes
        ):
            raise ValueError("reason_codes must be a tuple of nonempty strings")
        if type(self.limitations) is not tuple or any(
            type(item) is not str or not item for item in self.limitations
        ):
            raise ValueError("limitations must be a tuple of nonempty strings")
        object.__setattr__(self, "reason_codes", _ordered_reasons(self.reason_codes))
        object.__setattr__(self, "limitations", tuple(sorted(set(self.limitations))))
        if (
            self.case_type == "evidence_integrity"
            and self.threshold_evaluation is not None
        ):
            raise ValueError("evidence integrity cannot carry a threshold evaluation")
        if (
            isinstance(self.evidence, TerminalCaseEvidence)
            and self.threshold_evaluation is not None
        ):
            raise ValueError("terminal evidence cannot carry metrics")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    validation_plan: ValidationPlan
    result: str
    case_results: tuple[CaseResult, ...]
    threshold_evaluations: tuple[ThresholdEvaluation, ...]
    sample_integrity: SampleCaseEvidence
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        _validated_plan(self.validation_plan)
        if self.result not in _REPORT_RESULTS:
            raise ValueError("validation report result is not supported")
        if type(self.case_results) is not tuple:
            raise ValueError("case_results must be a tuple")
        if type(self.threshold_evaluations) is not tuple:
            raise ValueError("threshold_evaluations must be a tuple")
        if type(self.sample_integrity) is not SampleCaseEvidence:
            raise ValueError("sample_integrity must be SampleCaseEvidence")
        if type(self.limitations) is not tuple:
            raise ValueError("limitations must be a tuple")


def build_validation_plan(
    candidate_ref: object,
    ledger_snapshot_ref: object,
    policy: ValidationPolicy,
) -> ValidationPlan:
    try:
        _require_ref(candidate_ref, "candidate_ref")
        _require_ref(ledger_snapshot_ref, "ledger_snapshot_ref")
        if type(policy) is not ValidationPolicy:
            raise ValueError("policy must be a ValidationPolicy")
        normalized = ValidationPolicy(
            policy.accepted_backtest_grades,
            policy.accepted_metric_profile_refs,
            policy.holdout,
            policy.oos_rule,
            policy.decision_rule,
        )
        return ValidationPlan(
            deepcopy(candidate_ref),
            deepcopy(ledger_snapshot_ref),
            normalized.accepted_backtest_grades,
            deepcopy(normalized.accepted_metric_profile_refs),
            normalized.holdout,
            normalized.oos_rule,
            normalized.decision_rule,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValidationCoreFailure("VALIDATION_PLAN_INVALID") from error


def assess_admission(
    plan: ValidationPlan,
    candidate_graph: CandidateGraph,
    sample_integrity: SampleAdmissionEvidence,
) -> CaseResult:
    plan = _plan_or_fail(plan)
    try:
        graph_status = _candidate_graph_status(plan, candidate_graph)
    except (AttributeError, KeyError, TypeError, ValueError):
        graph_status = "invalid"
    if graph_status != "ok":
        return _result(
            plan,
            "evidence_integrity",
            "BLOCKED" if graph_status == "missing" else "FAILED",
            "CANDIDATE_PROVENANCE_INVALID",
        )

    try:
        sample_status, recomputed = _sample_status(
            plan, candidate_graph, sample_integrity
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        sample_status, recomputed = "ledger_conflict", None
    if sample_status == "ledger_conflict":
        return _result(
            plan,
            "evidence_integrity",
            "FAILED",
            "SAMPLE_LEDGER_CONFLICT",
        )

    if recomputed is None:
        return _result(
            plan,
            "evidence_integrity",
            "FAILED",
            "CANDIDATE_PROVENANCE_INVALID",
        )

    evidence = SampleCaseEvidence(
        deepcopy(plan.sample_consumption_snapshot_ref),
        recomputed.untouched,
        recomputed.conflicting_records,
    )
    if sample_status == "coverage_missing":
        return _result(
            plan,
            "evidence_integrity",
            "BLOCKED",
            "SAMPLE_RESERVATION_COVERAGE_MISSING",
            evidence=evidence,
        )
    if sample_status == "contaminated":
        return _result(
            plan,
            "evidence_integrity",
            "BLOCKED",
            "HOLDOUT_CONTAMINATED",
            evidence=evidence,
        )
    return _result(plan, "evidence_integrity", "PASS", evidence=evidence)


def assess_oos(
    plan: ValidationPlan,
    completed_or_terminal: OosObservation | ProviderFailure | BaseException,
    analysis_or_failure: AnalysisObservation | ProviderFailure | BaseException | None,
) -> CaseResult:
    plan = _plan_or_fail(plan)

    failure = _provider_failure(completed_or_terminal)
    if failure is not None:
        return _result(
            plan,
            "out_of_sample",
            "FAILED",
            failure.code,
            evidence=ProviderFailureEvidence(failure.code),
        )
    if type(completed_or_terminal) is not OosObservation:
        return _result(plan, "out_of_sample", "FAILED", "ANALYSIS_LINK_INVALID")

    try:
        observation = completed_or_terminal.observation
    except AttributeError:
        return _result(plan, "out_of_sample", "FAILED", "ANALYSIS_LINK_INVALID")
    if set(observation) == _TERMINAL_FIELDS and observation.get("status") in {
        "BLOCKED",
        "FAILED",
        "CANCELLED",
    }:
        if not _terminal_record_valid(observation):
            return _result(plan, "out_of_sample", "FAILED", "ANALYSIS_LINK_INVALID")
        status = observation["status"]
        outcome, reason = {
            "BLOCKED": ("BLOCKED", "BACKTEST_TERMINAL_BLOCKED"),
            "FAILED": ("FAILED", "BACKTEST_TERMINAL_FAILED"),
            "CANCELLED": ("INCONCLUSIVE", "BACKTEST_TERMINAL_CANCELLED"),
        }[status]
        return _result(
            plan,
            "out_of_sample",
            outcome,
            reason,
            evidence=TerminalCaseEvidence(
                status,
                deepcopy(observation["durable_evidence_ref"]),
            ),
        )

    if not _oos_context_valid(plan, completed_or_terminal):
        return _result(plan, "out_of_sample", "FAILED", "ANALYSIS_LINK_INVALID")
    if not _completed_record_valid(observation):
        return _result(plan, "out_of_sample", "FAILED", "ANALYSIS_LINK_INVALID")

    failure = _provider_failure(analysis_or_failure)
    if failure is not None:
        return _result(
            plan,
            "out_of_sample",
            "FAILED",
            failure.code,
            evidence=ProviderFailureEvidence(failure.code),
        )
    if type(analysis_or_failure) is not AnalysisObservation:
        return _result(plan, "out_of_sample", "FAILED", "ANALYSIS_LINK_INVALID")
    if not _analysis_context_valid(plan, completed_or_terminal, analysis_or_failure):
        return _result(plan, "out_of_sample", "FAILED", "ANALYSIS_LINK_INVALID")

    analysis = analysis_or_failure.analysis
    allowed_fields = _ANALYSIS_BASE_FIELDS | {plan.oos_rule.metric_key}
    if set(analysis) not in (_ANALYSIS_BASE_FIELDS, allowed_fields):
        return _result(plan, "out_of_sample", "FAILED", "ANALYSIS_LINK_INVALID")
    if not _analysis_links_valid(plan, observation, analysis):
        return _result(plan, "out_of_sample", "FAILED", "ANALYSIS_LINK_INVALID")
    if observation["result_grade"] not in plan.accepted_backtest_grades:
        return _result(plan, "out_of_sample", "FAILED", "RESULT_GRADE_UNACCEPTED")

    trade_count = analysis.get("trade_count")
    metric_value = analysis.get(plan.oos_rule.metric_key)
    evidence = CompletedCaseEvidence(
        deepcopy(observation["publication_ref"]),
        deepcopy(analysis["analysis_ref"]),
        deepcopy(analysis["metric_profile_ref"]),
        analysis["source_execution_result_hash"],
        analysis["result_grade"],
        plan.oos_rule.metric_key,
        metric_value if type(metric_value) is str else None,
        trade_count
        if type(trade_count) is int and type(trade_count) is not bool
        else None,
    )
    if (
        metric_value is None
        or type(trade_count) is not int
        or type(trade_count) is bool
    ):
        return _result(
            plan,
            "out_of_sample",
            "INCONCLUSIVE",
            "METRIC_MISSING_OR_INSUFFICIENT",
            evidence=evidence,
        )
    try:
        metric_value = _canonical_decimal(metric_value, plan.oos_rule.metric_key)
    except ValueError:
        return _result(plan, "out_of_sample", "FAILED", "ANALYSIS_LINK_INVALID")
    if trade_count < plan.oos_rule.minimum_trade_count:
        return _result(
            plan,
            "out_of_sample",
            "INCONCLUSIVE",
            "METRIC_MISSING_OR_INSUFFICIENT",
            evidence=evidence,
        )

    passed = Decimal(metric_value) >= Decimal(plan.oos_rule.threshold)
    threshold = ThresholdEvaluation(
        plan.oos_rule.metric_key,
        metric_value,
        plan.oos_rule.operator,
        plan.oos_rule.threshold,
        passed,
        trade_count,
        plan.oos_rule.minimum_trade_count,
    )
    return _result(
        plan,
        "out_of_sample",
        "PASS" if passed else "FAIL",
        None if passed else "OOS_THRESHOLD_NOT_MET",
        evidence=evidence,
        threshold_evaluation=threshold,
    )


def aggregate_validation_report(
    plan: ValidationPlan,
    case_results: tuple[CaseResult, ...],
) -> ValidationReport | None:
    plan = _plan_or_fail(plan)
    if type(case_results) is not tuple:
        raise ValidationCoreFailure("CASE_COVER_INVALID")

    valid_results: list[CaseResult] = []
    for result in case_results:
        if type(result) is not CaseResult:
            raise ValidationCoreFailure("CASE_COVER_INVALID")
        try:
            normalized = CaseResult(
                result.validation_plan,
                result.case_type,
                result.outcome,
                result.reason_codes,
                result.limitations,
                result.evidence,
                result.threshold_evaluation,
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValidationCoreFailure("CASE_COVER_INVALID") from error
        if normalized.validation_plan != plan:
            raise ValidationCoreFailure("CASE_COVER_INVALID")
        valid_results.append(normalized)

    if any(result.outcome == "FAILED" for result in valid_results):
        return None
    by_type = {result.case_type: result for result in valid_results}
    if len(by_type) != len(valid_results) or tuple(by_type) != _CASE_TYPES:
        if set(by_type) != set(_CASE_TYPES) or len(valid_results) != len(_CASE_TYPES):
            raise ValidationCoreFailure("CASE_COVER_INVALID")
    try:
        ordered = tuple(by_type[case_type] for case_type in _CASE_TYPES)
    except KeyError as error:
        raise ValidationCoreFailure("CASE_COVER_INVALID") from error

    integrity_evidence = ordered[0].evidence
    if type(integrity_evidence) is not SampleCaseEvidence:
        raise ValidationCoreFailure("CASE_COVER_INVALID")
    if any(result.outcome == "FAIL" for result in ordered):
        report_result = plan.decision_rule.required_fail
    elif any(result.outcome in {"BLOCKED", "INCONCLUSIVE"} for result in ordered):
        report_result = plan.decision_rule.blocked_or_inconclusive
    else:
        report_result = "supported"

    thresholds = tuple(
        result.threshold_evaluation
        for result in ordered
        if result.threshold_evaluation is not None
    )
    limitations = tuple(
        sorted({item for result in ordered for item in result.limitations})
    )
    return ValidationReport(
        plan,
        report_result,
        ordered,
        thresholds,
        integrity_evidence,
        limitations,
    )


def _validated_plan(value: object) -> ValidationPlan:
    if type(value) is not ValidationPlan:
        raise ValueError("plan must be a ValidationPlan")
    try:
        normalized = ValidationPlan(
            value.candidate_ref,
            value.sample_consumption_snapshot_ref,
            value.accepted_backtest_grades,
            value.accepted_metric_profile_refs,
            value.holdout,
            value.oos_rule,
            value.decision_rule,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("plan must contain canonical fields") from error
    if normalized != value:
        raise ValueError("plan must contain canonical fields")
    return normalized


def _plan_or_fail(value: object) -> ValidationPlan:
    try:
        return _validated_plan(value)
    except (TypeError, ValueError) as error:
        raise ValidationCoreFailure("VALIDATION_PLAN_INVALID") from error


def _payload(node: object, fields: set[str]) -> tuple[str, dict[str, Any] | None]:
    if type(node) is not ResolvedArtifact:
        return "invalid", None
    try:
        _require_ref(node.ref, "artifact ref")
    except ValueError:
        return "invalid", None
    if node.payload is None:
        return "missing", None
    if type(node.payload) is not dict or set(node.payload) != fields:
        return "invalid", None
    return "ok", node.payload


def _utc_epoch_nanoseconds(value: object) -> int | None:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        return None
    try:
        instant = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return None
    epoch = datetime(1970, 1, 1)
    delta = instant - epoch
    return ((delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1_000


def _model_build_graph_valid(
    graph: ModelBuildGraph,
    *,
    candidate: dict[str, Any],
    experiment_ref: object,
    manifest: dict[str, Any],
    trial: dict[str, Any],
    trial_spec: dict[str, Any],
    completed: dict[str, Any],
    required: tuple[SampleConsumptionRecord, ...],
) -> bool:
    nodes = (
        (graph.feature_recipe, _FEATURE_RECIPE_FIELDS),
        (graph.trainer_recipe, _TRAINER_RECIPE_FIELDS),
        (graph.model_build_plan, _MODEL_BUILD_PLAN_FIELDS),
        (graph.feature_build_task, _FEATURE_BUILD_TASK_FIELDS),
        (graph.model_training_task, _MODEL_TRAINING_TASK_FIELDS),
        (graph.feature_dataset_manifest, _FEATURE_MANIFEST_FIELDS),
        (graph.model_build_evidence, _MODEL_EVIDENCE_FIELDS),
        (graph.feature_build_outcome, _TASK_OUTCOME_FIELDS),
        (graph.model_training_outcome, _TASK_OUTCOME_FIELDS),
    )
    payloads: list[dict[str, Any]] = []
    for node, fields in nodes:
        status, payload = _payload(node, fields)
        if status != "ok" or payload is None:
            return False
        payloads.append(payload)
    (
        feature_recipe,
        trainer_recipe,
        plan,
        feature_task,
        training_task,
        feature_manifest,
        model_evidence,
        feature_outcome,
        training_outcome,
    ) = payloads
    training_slice = plan["training_slice"]
    model_artifact = model_evidence["model_artifact"]
    model_binding = completed.get("model_binding")
    if (
        type(training_slice) is not dict
        or set(training_slice) != {
            "market_bundle_ref",
            "dataset_revision",
            "interval_start",
            "interval_end",
        }
        or type(model_artifact) is not dict
        or set(model_artifact) != _MODEL_ARTIFACT_FIELDS
        or type(model_binding) is not dict
        or set(model_binding) != _MODEL_BINDING_FIELDS
    ):
        return False
    body = {
        key: value
        for key, value in model_artifact.items()
        if key != "artifact_ref_hash"
    }
    artifact_hash = "sha256:" + hashlib.sha256(
        json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    training_start = model_artifact.get("training_start")
    training_end = model_artifact.get("training_end")
    if (
        type(training_start) is not dict
        or set(training_start) != {"type", "epoch_nanoseconds"}
        or training_start.get("type") != "utc_instant"
        or type(training_start.get("epoch_nanoseconds")) is not int
        or type(training_end) is not dict
        or set(training_end) != {"type", "epoch_nanoseconds"}
        or training_end.get("type") != "utc_instant"
        or type(training_end.get("epoch_nanoseconds")) is not int
    ):
        return False
    feature_witness = feature_outcome.get("witness")
    training_witness = training_outcome.get("witness")
    feature_task_ref = graph.feature_build_task.ref
    training_task_ref = graph.model_training_task.ref
    links_valid = (
        candidate.get("model_build_evidence_ref") == graph.model_build_evidence.ref
        and plan["feature_recipe_ref"] == graph.feature_recipe.ref
        and plan["trainer_recipe_ref"] == graph.trainer_recipe.ref
        and feature_task["experiment_ref"] == experiment_ref
        and feature_task["model_build_plan_ref"] == graph.model_build_plan.ref
        and training_task["experiment_ref"] == experiment_ref
        and training_task["model_build_plan_ref"] == graph.model_build_plan.ref
        and training_task["feature_build_task_ref"] == feature_task_ref
        and feature_manifest["model_build_plan_ref"] == graph.model_build_plan.ref
        and feature_manifest["dataset_revision"] == training_slice["dataset_revision"]
        and feature_manifest["interval_start"] == training_slice["interval_start"]
        and feature_manifest["interval_end"] == training_slice["interval_end"]
        and feature_manifest["feature_schema_hash"]
        == feature_recipe["feature_schema_hash"]
        and model_evidence["model_build_plan_ref"] == graph.model_build_plan.ref
        and model_evidence["feature_dataset_manifest_ref"]
        == graph.feature_dataset_manifest.ref
        and model_artifact["artifact_ref_hash"] == artifact_hash
        and model_artifact["model_key"] == trainer_recipe["model_key"]
        and model_artifact["training_data_hash"]
        == feature_manifest["training_data_hash"]
        and model_artifact["training_code_hash"]
        == trainer_recipe["training_code_hash"]
        and model_artifact["feature_schema_hash"]
        == feature_recipe["feature_schema_hash"]
        and model_artifact["supersedes_revision_id"] is None
        and training_start["epoch_nanoseconds"]
        == _utc_epoch_nanoseconds(training_slice["interval_start"])
        and training_end["epoch_nanoseconds"]
        == _utc_epoch_nanoseconds(training_slice["interval_end"])
        and trial["model_input_bindings"]
        == {"primary_model": graph.model_build_plan.ref}
        and trial_spec["resolved_model_refs"] == [model_artifact]
        and model_binding["type"] == "model_request_binding"
        and model_binding["schema_version"] == 1
        and model_binding["input_name"] == "primary_model"
        and model_binding["model_key"] == model_artifact["model_key"]
        and model_binding["artifact_ref_hash"] == model_artifact["artifact_ref_hash"]
        and type(model_binding["timeline_hash"]) is str
        and _HASH.fullmatch(model_binding["timeline_hash"]) is not None
        and feature_outcome.get("state") == "COMPLETED"
        and feature_outcome.get("task_ref")
        == {"kind": "FEATURE_BUILD", "task_artifact_ref": feature_task_ref}
        and feature_witness
        == {
            "feature_dataset_manifest": {
                "feature_dataset_manifest_ref": graph.feature_dataset_manifest.ref
            }
        }
        and training_outcome.get("state") == "COMPLETED"
        and training_outcome.get("task_ref")
        == {"kind": "MODEL_TRAINING", "task_artifact_ref": training_task_ref}
        and training_witness
        == {
            "model_build_evidence": {
                "model_build_evidence_ref": graph.model_build_evidence.ref
            }
        }
        and tuple(manifest["task_outcome_refs"]).count(
            graph.feature_build_outcome.ref
        )
        == 1
        and tuple(manifest["task_outcome_refs"]).count(
            graph.model_training_outcome.ref
        )
        == 1
    )
    if not links_valid:
        return False
    feature_record = any(
        record.dataset_revision == training_slice["dataset_revision"]
        and record.interval_start == training_slice["interval_start"]
        and record.interval_end == training_slice["interval_end"]
        and record.purpose == "feature_build"
        for record in required
    )
    training_record = any(
        record.dataset_revision == training_slice["dataset_revision"]
        and record.interval_start == training_slice["interval_start"]
        and record.interval_end == training_slice["interval_end"]
        and record.purpose == "model_training"
        for record in required
    )
    return feature_record and training_record


def _candidate_graph_status(plan: ValidationPlan, graph: object) -> str:
    if type(graph) is not CandidateGraph:
        return "invalid"
    candidate_fields = (
        _CANDIDATE_FIELDS
        if graph.model_build is None
        else _CANDIDATE_FIELDS | {"model_build_evidence_ref"}
    )
    nodes = (
        (graph.candidate, candidate_fields),
        (graph.candidate_family, _FAMILY_FIELDS),
        (graph.execution_manifest, _MANIFEST_FIELDS),
        (graph.selection_declaration, _SELECTION_DECLARATION_FIELDS),
        (graph.selection_policy, _SELECTION_POLICY_FIELDS),
        (graph.selected_trial_declaration, _TRIAL_DECLARATION_FIELDS),
        (graph.selected_trial_spec, _TRIAL_SPEC_FIELDS),
        (graph.selected_trial_outcome, _TASK_OUTCOME_FIELDS),
        (graph.selected_analysis_task, _ANALYSIS_TASK_FIELDS),
        (graph.selected_analysis_outcome, _TASK_OUTCOME_FIELDS),
    )
    payloads: list[dict[str, Any]] = []
    for node, fields in nodes:
        status, payload = _payload(node, fields)
        if status != "ok":
            return status
        if payload is None:
            return "invalid"
        payloads.append(payload)
    (
        candidate,
        family,
        manifest,
        selection,
        policy,
        trial,
        trial_spec,
        trial_outcome,
        analysis_task,
        analysis_outcome,
    ) = payloads
    if graph.selected_completed is None or graph.selected_analysis is None:
        return "missing"
    completed = graph.selected_completed
    analysis = graph.selected_analysis
    completed_fields = (
        _COMPLETED_FIELDS
        if graph.model_build is None
        else _COMPLETED_FIELDS | {"model_binding"}
    )
    if set(completed) != completed_fields or not _completed_record_valid(
        completed
    ) or set(analysis) != (_ANALYSIS_BASE_FIELDS | {"simple_period_return"}):
        return "invalid"

    experiment_ref = family["experiment_ref"]
    data_slice = trial["data_slice"]
    if type(data_slice) is not dict or set(data_slice) != {
        "market_bundle_ref",
        "dataset_revision",
        "interval_start",
        "interval_end",
    }:
        return "invalid"

    def is_sequence(value: object) -> bool:
        return type(value) in (tuple, list)

    links_valid = (
        graph.candidate.ref == plan.candidate_ref
        and candidate["candidate_family_ref"] == graph.candidate_family.ref
        and candidate["selection_declaration_ref"] == graph.selection_declaration.ref
        and candidate["selected_trial_declaration_ref"]
        == graph.selected_trial_declaration.ref
        and candidate["selected_trial_spec_ref"] == graph.selected_trial_spec.ref
        and candidate["selected_publication_ref"] == completed["publication_ref"]
        and candidate["selected_analysis_ref"] == analysis["analysis_ref"]
        and type(candidate["selection_rank"]) is int
        and candidate["selection_rank"] > 0
        and type(candidate["validated"]) is bool
        and not candidate["validated"]
        and family["execution_manifest_ref"] == graph.execution_manifest.ref
        and manifest["experiment_ref"] == experiment_ref
        and selection["experiment_ref"] == experiment_ref
        and selection["selection_policy_ref"] == graph.selection_policy.ref
        and selection["universe_kind"] == "candidate_trial_declarations_v1"
        and trial["experiment_ref"] == experiment_ref
        and trial_spec["trial_declaration_ref"] == graph.selected_trial_declaration.ref
        and analysis_task["experiment_ref"] == experiment_ref
        and analysis_task["trial_declaration_ref"]
        == graph.selected_trial_declaration.ref
        and analysis_task["metric_profile_ref"] == policy["metric_profile_ref"]
        and policy["metric_profile_ref"] == analysis["metric_profile_ref"]
        and _wire_key(policy["metric_profile_ref"])
        in {_wire_key(ref) for ref in plan.accepted_metric_profile_refs}
        and is_sequence(policy["eligible_trial_statuses"])
        and tuple(policy["eligible_trial_statuses"]) == ("COMPLETED",)
        and is_sequence(policy["accepted_backtest_grades"])
        and completed["result_grade"] in tuple(policy["accepted_backtest_grades"])
        and completed["result_grade"] in plan.accepted_backtest_grades
        and analysis["result_grade"] == completed["result_grade"]
        and analysis["source_publication_ref"] == completed["publication_ref"]
        and analysis["source_execution_result_hash"]
        == completed["execution_result_hash"]
        and analysis["analysis_ref"] == candidate["selected_analysis_ref"]
        and is_sequence(manifest["task_outcome_refs"])
        and tuple(manifest["task_outcome_refs"]).count(graph.selected_trial_outcome.ref)
        == 1
        and tuple(manifest["task_outcome_refs"]).count(
            graph.selected_analysis_outcome.ref
        )
        == 1
        and _completed_trial_outcome_valid(
            trial_outcome,
            graph.selected_trial_declaration.ref,
            completed["publication_ref"],
        )
        and _completed_analysis_outcome_valid(
            analysis_outcome,
            graph.selected_analysis_task.ref,
            analysis["analysis_ref"],
            completed["publication_ref"],
        )
    )
    if not links_valid:
        return "invalid"

    required = graph.required_sample_records
    if graph.model_build is not None and not _model_build_graph_valid(
        graph.model_build,
        candidate=candidate,
        experiment_ref=experiment_ref,
        manifest=manifest,
        trial=trial,
        trial_spec=trial_spec,
        completed=completed,
        required=required,
    ):
        return "invalid"
    discovery = any(
        record.dataset_revision == data_slice["dataset_revision"]
        and record.interval_start == data_slice["interval_start"]
        and record.interval_end == data_slice["interval_end"]
        and record.purpose == "discovery"
        for record in required
    )
    selection_record = any(
        record.dataset_revision == data_slice["dataset_revision"]
        and record.interval_start == data_slice["interval_start"]
        and record.interval_end == data_slice["interval_end"]
        and record.purpose == "selection"
        for record in required
    )
    model_presence_valid = (
        "model_build_evidence_ref" not in candidate
        and "model_binding" not in completed
        if graph.model_build is None
        else True
    )
    return "ok" if discovery and selection_record and model_presence_valid else "invalid"


def _completed_trial_outcome_valid(
    outcome: dict[str, Any], trial_ref: object, publication_ref: object
) -> bool:
    return (
        outcome["state"] == "COMPLETED"
        and outcome["task_ref"] == {"kind": "TRIAL", "task_artifact_ref": trial_ref}
        and outcome["witness"]
        == {"trial_completed_publication": {"publication_ref": publication_ref}}
    )


def _completed_analysis_outcome_valid(
    outcome: dict[str, Any],
    analysis_task_ref: object,
    analysis_ref: object,
    publication_ref: object,
) -> bool:
    return (
        outcome["state"] == "COMPLETED"
        and outcome["task_ref"]
        == {"kind": "ANALYSIS", "task_artifact_ref": analysis_task_ref}
        and outcome["witness"]
        == {
            "analysis_derivation": {
                "analysis_ref": analysis_ref,
                "source_publication_ref": publication_ref,
            }
        }
    )


def _sample_status(
    plan: ValidationPlan,
    graph: CandidateGraph,
    evidence: object,
) -> tuple[str, SampleIntegrityResult | None]:
    if type(evidence) is not SampleAdmissionEvidence:
        return "ledger_conflict", None
    try:
        snapshot = SampleConsumptionSnapshot(
            evidence.snapshot.as_of, evidence.snapshot.records
        )
        supplied = SampleIntegrityResult(
            evidence.integrity.untouched, evidence.integrity.conflicting_records
        )
    except (AttributeError, TypeError, ValueError):
        return "ledger_conflict", None
    if (
        evidence.snapshot_ref != plan.sample_consumption_snapshot_ref
        or not _checkpoint_valid(evidence.checkpoint, snapshot)
        or evidence.ledger_conflict
    ):
        return "ledger_conflict", None
    recomputed = assess_untouched_holdout(
        snapshot,
        dataset_revision=plan.holdout.dataset_revision,
        interval_start=plan.holdout.interval_start,
        interval_end=plan.holdout.interval_end,
    )
    if supplied != recomputed:
        return "ledger_conflict", None
    if any(record not in snapshot.records for record in graph.required_sample_records):
        return "coverage_missing", recomputed
    if not recomputed.untouched:
        return "contaminated", recomputed
    return "ok", recomputed


def _checkpoint_valid(checkpoint: object, snapshot: SampleConsumptionSnapshot) -> bool:
    if type(checkpoint) is not dict or set(checkpoint) != {
        "log_name",
        "as_of",
        "upper_log_sequence",
        "head_receipt_hash",
    }:
        return False
    upper = checkpoint["upper_log_sequence"]
    head = checkpoint["head_receipt_hash"]
    return (
        checkpoint["log_name"] == "validation.sample-consumption.v1"
        and checkpoint["as_of"] == snapshot.as_of
        and type(upper) is int
        and upper >= 0
        and (
            (upper == 0 and head is None)
            or (upper > 0 and type(head) is str and _HASH.fullmatch(head) is not None)
        )
    )


def _completed_record_valid(record: object) -> bool:
    return (
        type(record) is dict
        and frozenset(record) in {
            frozenset(_COMPLETED_FIELDS),
            frozenset(_COMPLETED_FIELDS | {"model_binding"}),
        }
        and type(record["semantic_run_id"]) is str
        and bool(record["semantic_run_id"])
        and type(record["execution_result_hash"]) is str
        and _HASH.fullmatch(record["execution_result_hash"]) is not None
        and type(record["result_grade"]) is str
        and bool(record["result_grade"])
    )


def _terminal_record_valid(record: dict[str, Any]) -> bool:
    return (
        set(record) == _TERMINAL_FIELDS
        and record["status"] in {"BLOCKED", "FAILED", "CANCELLED"}
        and record["durable_evidence_ref"] is not None
    )


def _oos_context_valid(plan: ValidationPlan, observation: OosObservation) -> bool:
    return (
        observation.validation_plan == plan
        and observation.case_ref == observation.request_context_ref
    )


def _analysis_context_valid(
    plan: ValidationPlan,
    completed: OosObservation,
    analysis: AnalysisObservation,
) -> bool:
    return analysis.validation_plan == plan and analysis.case_ref == completed.case_ref


def _analysis_links_valid(
    plan: ValidationPlan,
    completed: dict[str, Any],
    analysis: dict[str, Any],
) -> bool:
    try:
        trade_count = analysis.get("trade_count")
        return (
            analysis.get("metric_profile_ref") == plan.oos_rule.metric_profile_ref
            and _wire_key(analysis.get("metric_profile_ref"))
            in {_wire_key(ref) for ref in plan.accepted_metric_profile_refs}
            and analysis.get("source_publication_ref") == completed["publication_ref"]
            and analysis.get("source_execution_result_hash")
            == completed["execution_result_hash"]
            and analysis.get("result_grade") == completed["result_grade"]
            and type(trade_count) is int
            and type(trade_count) is not bool
            and trade_count >= 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def _provider_failure(value: object) -> ProviderFailure | None:
    if type(value) is ProviderFailure:
        try:
            return ProviderFailure(value.code)
        except (AttributeError, TypeError, ValueError):
            return None
    if isinstance(value, BaseException):
        code = getattr(value, "code", None)
        if type(code) is str and code:
            return ProviderFailure(code)
    return None


def _ordered_reasons(codes: tuple[str, ...]) -> tuple[str, ...]:
    rank = {code: index for index, code in enumerate(FAILURE_PRECEDENCE)}
    return tuple(sorted(set(codes), key=lambda code: (rank.get(code, len(rank)), code)))


def _result(
    plan: ValidationPlan,
    case_type: str,
    outcome: str,
    reason: str | None = None,
    *,
    evidence: object | None = None,
    threshold_evaluation: ThresholdEvaluation | None = None,
) -> CaseResult:
    return CaseResult(
        plan,
        case_type,
        outcome,
        () if reason is None else (reason,),
        (),
        evidence,
        threshold_evaluation,
    )
