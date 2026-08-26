from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from crypto_quant_domain import (
    ArtifactEnvelope,
    ArtifactRef,
    canonical_bytes,
    canonical_sha256,
)
from crypto_quant_foundation import FoundationFailure, LocalFoundation

from .integration import (
    AnalysisObservation,
    CandidateGraph,
    CaseResult,
    OosObservation,
    ProviderFailure,
    ResolvedArtifact,
    ValidationPlan,
    ValidationPolicy,
    assess_admission,
    assess_oos,
    build_validation_plan,
)
from .ledger import SampleConsumptionLedger
from .runtime import (
    NoReport,
    PublishedValidationReport,
    _artifact_log_payloads,
    _candidate_graph,
    _consumer_id,
    _failure_code,
    _is_terminal_ref,
    _load_analysis,
    _load_completed,
    _plan_payload,
    _preflight_admission,
    _published,
    _ref_from_wire,
    _snapshot_evidence,
    _threshold_payload,
    _wire,
)
from .sample_consumption import SampleConsumptionRecord, build_snapshot

_ARTIFACT_LOG = "validation.artifacts.v1"
_RESEARCH_ARTIFACT_LOG = "research.artifacts.v1"
_RESEARCH_EXECUTION_LOG = "research.execution.v1"
_SAMPLE_LOG = "validation.sample-consumption.v1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")

_CANDIDATE_V3_FIELDS = {
    "candidate_family_ref",
    "selection_declaration_ref",
    "selected_trial_declaration_ref",
    "selected_trial_spec_ref",
    "selected_publication_ref",
    "selected_analysis_ref",
    "selection_rank",
    "validated",
    "selected_target_materialization_evidence_ref",
}
_EXPERIMENT_V2_FIELDS = {
    "hypothesis_ref",
    "strategy_definition_ref",
    "data_slices",
    "parameter_combinations",
    "seeds",
    "scenario_refs",
    "backtest_template_ref",
    "model_build_plan",
    "metric_profile_refs",
    "budget",
    "target_recipe_ref",
}
_TARGET_RECIPE_FIELDS = {
    "target_key",
    "strategy_artifact",
    "target_schema_hash",
    "input_names",
}
_TARGET_TASK_FIELDS = {
    "experiment_ref",
    "trial_declaration_ref",
    "target_recipe_ref",
}
_DISCOVERY_EVIDENCE_FIELDS = {
    "target_build_task_ref",
    "trial_declaration_ref",
    "target_recipe_ref",
    "materialization_request_hash",
    "input_data_hash",
    "target_stream_ref",
    "target_stream_digest",
    "event_count",
}
_VALIDATION_EVIDENCE_FIELDS = {
    "validation_case_ref",
    "candidate_ref",
    "target_recipe_ref",
    "materialization_request_hash",
    "input_data_hash",
    "target_stream_ref",
    "target_stream_digest",
    "event_count",
}
_TARGET_RESULT_FIELDS = {
    "type",
    "schema_version",
    "request_hash",
    "strategy_artifact",
    "input_data_hash",
    "target_stream",
}
_TARGET_REF_FIELDS = {"type", "artifact_ref"}
_TARGET_STREAM_FIELDS = {"type", "schema_version", "stream_key", "events"}
_TARGET_EVENT_FIELDS = {
    "type",
    "event_id",
    "stream_key",
    "event_type",
    "capability",
    "instrument_id",
    "event_time",
    "available_time",
    "phase",
    "source_sequence",
    "revision_id",
    "supersedes_revision_id",
    "source_key",
    "source_hash",
    "payload",
}
_CASE_RESULT_V2_FIELDS = {
    "case_ref",
    "outcome",
    "reason_codes",
    "limitations",
    "evidence",
    "threshold_evaluation",
    "validation_target_materialization_evidence_ref",
}
_REPORT_V2_FIELDS = {
    "validation_plan_ref",
    "result",
    "case_result_refs",
    "threshold_evaluations",
    "sample_integrity_ref",
    "limitations",
    "validation_target_materialization_evidence_ref",
}
_COMPLETED_CASE_EVIDENCE_FIELDS = {
    "publication_ref",
    "analysis_ref",
    "metric_profile_ref",
    "source_execution_result_hash",
    "result_grade",
    "metric_key",
    "metric_value",
    "trade_count",
}
_TERMINAL_CASE_EVIDENCE_FIELDS = {"status", "durable_evidence_ref"}
_PROVIDER_FAILURE_EVIDENCE_FIELDS = {"code"}
_SAMPLE_CASE_EVIDENCE_FIELDS = {"snapshot_ref", "untouched", "conflicting_records"}
_THRESHOLD_EVALUATION_FIELDS = {
    "metric_key",
    "observed",
    "operator",
    "threshold",
    "passed",
    "trade_count",
    "minimum_trade_count",
}


class _OosReservationConflict(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidationTargetMaterializationEvidence:
    validation_case_ref: object
    candidate_ref: object
    target_recipe_ref: object
    materialization_request_hash: str
    input_data_hash: str
    target_stream_ref: object
    target_stream_digest: str
    event_count: int

    def __post_init__(self) -> None:
        for name, artifact_type, schema_version in (
            ("validation_case_ref", "validation_case", 1),
            ("candidate_ref", "strategy_candidate", 3),
            ("target_recipe_ref", "target_recipe", 1),
        ):
            ref = _canonical_artifact_ref(getattr(self, name), name)
            if (
                ref.artifact_type != artifact_type
                or ref.schema_version != schema_version
            ):
                raise ValueError(
                    f"{name} must reference {artifact_type}@{schema_version}"
                )
        _canonical_target_ref(self.target_stream_ref)
        _content_hash(self.materialization_request_hash, "materialization_request_hash")
        _content_hash(self.input_data_hash, "input_data_hash")
        _content_hash(self.target_stream_digest, "target_stream_digest")
        if type(self.event_count) is not int or self.event_count < 0:
            raise ValueError("event_count must be a nonnegative integer")

    @property
    def payload(self) -> dict[str, object]:
        return {
            "validation_case_ref": deepcopy(self.validation_case_ref),
            "candidate_ref": deepcopy(self.candidate_ref),
            "target_recipe_ref": deepcopy(self.target_recipe_ref),
            "materialization_request_hash": self.materialization_request_hash,
            "input_data_hash": self.input_data_hash,
            "target_stream_ref": deepcopy(self.target_stream_ref),
            "target_stream_digest": self.target_stream_digest,
            "event_count": self.event_count,
        }


@dataclass(frozen=True, slots=True)
class _TargetContext:
    target_recipe_ref: ArtifactRef
    strategy_artifact: dict[str, object]
    discovery_target_ref: dict[str, object]
    parameter_values: object
    seed: object


def _plain(value: object) -> Any:
    try:
        return json.loads(canonical_bytes(value).decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("value must be canonical JSON") from error


def _content_hash(value: object, name: str) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical sha256 hash")
    return value


def _canonical_artifact_ref(value: object, name: str) -> ArtifactRef:
    plain = _plain(value)
    if type(plain) is not dict:
        raise ValueError(f"{name} must be an ArtifactRef")
    ref = _ref_from_wire(plain)
    if ref.to_canonical_dict() != plain:
        raise ValueError(f"{name} must be canonical")
    return ref


def _canonical_target_ref(value: object) -> dict[str, object]:
    plain = _plain(value)
    if type(plain) is not dict or set(plain) != _TARGET_REF_FIELDS:
        raise ValueError("target_stream_ref must be one exact nominal reference")
    if plain["type"] != "backtest_target_stream_ref":
        raise ValueError("target_stream_ref nominal type is invalid")
    ref = _canonical_artifact_ref(
        plain["artifact_ref"], "target_stream_ref.artifact_ref"
    )
    if ref.artifact_type != "backtest_target_stream" or ref.schema_version != 1:
        raise ValueError("target_stream_ref must reference backtest_target_stream@1")
    return plain


def _canonical_strategy_artifact(value: object) -> dict[str, object]:
    plain = _plain(value)
    fields = {
        "type",
        "role",
        "artifact_key",
        "artifact_version",
        "install_mode",
        "source_tree_state",
        "content_hash",
        "source_snapshot_hash",
    }
    if type(plain) is not dict or set(plain) != fields:
        raise ValueError("strategy_artifact must be the exact BuildArtifactRef wire")
    if plain["type"] != "build_artifact_ref" or plain["role"] != "decision_source":
        raise ValueError("strategy_artifact must be a DECISION_SOURCE BuildArtifactRef")
    for name in ("artifact_key", "artifact_version"):
        if type(plain[name]) is not str or not plain[name]:
            raise ValueError(f"strategy_artifact.{name} must be nonempty")
    if plain["install_mode"] not in {"wheel", "container", "editable"}:
        raise ValueError("strategy_artifact.install_mode is invalid")
    if plain["source_tree_state"] not in {"clean", "dirty", "unknown"}:
        raise ValueError("strategy_artifact.source_tree_state is invalid")
    for name in ("content_hash", "source_snapshot_hash"):
        if plain[name] is not None:
            _content_hash(plain[name], f"strategy_artifact.{name}")
    if plain["content_hash"] is None and plain["source_snapshot_hash"] is None:
        raise ValueError("strategy_artifact must have immutable identity")
    return plain


def _canonical_target_stream(value: object) -> dict[str, Any]:
    plain = _plain(value)
    if type(plain) is not dict or set(plain) != _TARGET_STREAM_FIELDS:
        raise ValueError("target_stream must be the exact PrecomputedTargetStream wire")
    if plain["type"] != "precomputed_target_stream" or plain["schema_version"] != 1:
        raise ValueError("target_stream must be precomputed_target_stream@1")
    stream_key = plain["stream_key"]
    events = plain["events"]
    if type(stream_key) is not str or not stream_key or type(events) is not list:
        raise ValueError("target_stream is not canonical")
    ordering: list[tuple[int, int, str, int, str]] = []
    ids: list[str] = []
    for event in events:
        if type(event) is not dict or set(event) != _TARGET_EVENT_FIELDS:
            raise ValueError("target_stream event fields are not canonical")
        if event["type"] != "market_event" or event["stream_key"] != stream_key:
            raise ValueError("target_stream event identity is invalid")
        event_id = event["event_id"]
        if type(event_id) is not str or not event_id:
            raise ValueError("target_stream event_id is invalid")
        capability = event["capability"]
        phase = event["phase"]
        sequence = event["source_sequence"]
        if (
            type(capability) is not dict
            or set(capability) != {"type", "key", "version"}
            or capability["type"] != "market_bundle_capability"
            or type(capability["key"]) is not str
            or not capability["key"]
            or type(capability["version"]) is not int
            or capability["version"] <= 0
            or type(phase) is not dict
            or set(phase) != {"type", "rank", "code"}
            or phase["type"] != "timeline_phase"
            or type(phase["rank"]) is not int
            or type(phase["code"]) is not str
            or not phase["code"]
            or type(sequence) is not dict
            or set(sequence) != {"type", "value"}
            or sequence["type"] != "source_sequence"
            or type(sequence["value"]) is not int
        ):
            raise ValueError("target_stream event ordering metadata is invalid")
        instants: list[int] = []
        for name in ("event_time", "available_time"):
            instant = event[name]
            if (
                type(instant) is not dict
                or set(instant) != {"type", "epoch_nanoseconds"}
                or instant["type"] != "utc_instant"
                or type(instant["epoch_nanoseconds"]) is not int
            ):
                raise ValueError(f"target_stream event {name} is invalid")
            instants.append(instant["epoch_nanoseconds"])
        if instants[1] < instants[0]:
            raise ValueError("target event is available before event_time")
        instrument = event["instrument_id"]
        if instrument is not None and (
            type(instrument) is not dict
            or set(instrument) != {"type", "venue", "stable_key"}
            or instrument["type"] != "instrument_id"
        ):
            raise ValueError("target event instrument_id is invalid")
        for name in ("event_type", "revision_id", "source_key"):
            if type(event[name]) is not str or not event[name]:
                raise ValueError(f"target event {name} is invalid")
        if event["supersedes_revision_id"] is not None and (
            type(event["supersedes_revision_id"]) is not str
            or not event["supersedes_revision_id"]
        ):
            raise ValueError("target event supersedes_revision_id is invalid")
        _content_hash(event["source_hash"], "target event.source_hash")
        if type(event["payload"]) is not dict:
            raise ValueError("target event payload must be an object")
        ids.append(event_id)
        ordering.append(
            (instants[1], phase["rank"], phase["code"], sequence["value"], event_id)
        )
    if len(ids) != len(set(ids)) or ordering != sorted(ordering):
        raise ValueError("target events must have unique canonical ordering")
    if len({item[:4] for item in ordering}) != len(ordering):
        raise ValueError("target event ordering keys must be unique")
    return plain


def _artifact_event_id(ref: ArtifactRef) -> str:
    return canonical_sha256(("artifact-publication-v1", _ARTIFACT_LOG, ref))


def _publish(
    foundation: LocalFoundation,
    artifact_type: str,
    schema_version: int,
    payload: dict[str, object],
) -> ArtifactRef:
    envelope = ArtifactEnvelope.create(artifact_type, schema_version, payload)
    ref = foundation.put(envelope=envelope)
    foundation.append(_ARTIFACT_LOG, _artifact_event_id(ref), canonical_bytes(envelope))
    return ref


def _entries(
    foundation: LocalFoundation, log_name: str
) -> tuple[tuple[Any, ArtifactRef, dict[str, Any]], ...]:
    result: list[tuple[Any, ArtifactRef, dict[str, Any]]] = []
    for entry in foundation.entries(log_name):
        try:
            decoded = json.loads(entry.payload.decode("utf-8"))
            envelope = ArtifactEnvelope(
                decoded["artifact_type"],
                decoded["schema_version"],
                decoded["payload"],
                decoded["content_hash"],
            )
            ref = ArtifactRef.from_envelope(envelope)
            payload = _plain(envelope.payload)
            if (
                canonical_bytes(envelope) != entry.payload
                or entry.event_id
                != canonical_sha256(("artifact-publication-v1", log_name, ref))
                or type(payload) is not dict
            ):
                raise ValueError("owner publication is invalid")
        except Exception as error:
            raise ValueError("owner publication is invalid") from error
        result.append((entry, ref, payload))
    return tuple(result)


def _sample_entry(payload: bytes) -> tuple[SampleConsumptionRecord, ArtifactRef]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
        envelope = ArtifactEnvelope(
            decoded["artifact_type"],
            decoded["schema_version"],
            decoded["payload"],
            decoded["content_hash"],
        )
        value = _plain(envelope.payload)
        record_value = value["record"]
        if (
            canonical_bytes(envelope) != payload
            or envelope.artifact_type != "sample_consumption_append"
            or envelope.schema_version != 1
            or type(value) is not dict
            or set(value) != {"record", "producer_ref"}
            or type(record_value) is not dict
            or set(record_value)
            != {
                "dataset_revision",
                "interval_start",
                "interval_end",
                "purpose",
                "consumer_id",
                "consumed_at",
            }
        ):
            raise ValueError("sample entry shape is invalid")
        record = SampleConsumptionRecord(
            record_value["dataset_revision"],
            record_value["interval_start"],
            record_value["interval_end"],
            record_value["purpose"],
            record_value["consumer_id"],
            record_value["consumed_at"],
        )
        return record, _ref_from_wire(value["producer_ref"])
    except Exception as error:
        raise ValueError("sample entry is invalid") from error


def _require_discovery_reservation(
    foundation: LocalFoundation,
    trial_ref: ArtifactRef,
    trial: dict[str, Any],
    *,
    before_ledger_sequence: int,
) -> None:
    data_slice = trial.get("data_slice")
    if type(data_slice) is not dict:
        raise ValueError("trial data slice is invalid")
    matches = []
    for entry in foundation.entries(_SAMPLE_LOG):
        record, producer = _sample_entry(entry.payload)
        if (
            producer == trial_ref
            and record.dataset_revision == data_slice.get("dataset_revision")
            and record.interval_start == data_slice.get("interval_start")
            and record.interval_end == data_slice.get("interval_end")
            and record.purpose == "discovery"
            and record.consumer_id == _consumer_id(trial_ref)
        ):
            matches.append((entry, record))
    expected_event = canonical_sha256(
        (
            "sample-consumption-append-v1",
            trial_ref,
            data_slice.get("dataset_revision"),
            data_slice.get("interval_start"),
            data_slice.get("interval_end"),
            "discovery",
        )
    )
    if (
        len(matches) != 1
        or matches[0][0].event_id != expected_event
        or matches[0][1].consumed_at > matches[0][0].accepted_at
        or matches[0][0].ledger_sequence >= before_ledger_sequence
    ):
        raise ValueError("discovery reservation is not exact and preceding")


def _verified_target(
    backtest: Any,
    target_ref: object,
    producer_context_ref: ArtifactRef,
    *,
    expected_digest: str,
    expected_event_count: int,
    expected_stream: object | None = None,
) -> dict[str, Any]:
    try:
        target_ref = _canonical_target_ref(target_ref)
        loaded = _plain(backtest.load_target(deepcopy(target_ref)))
        if type(loaded) is not dict or set(loaded) != {
            "ref",
            "producer_context_ref",
            "target_stream",
            "digest",
        }:
            raise ValueError("load_target returned the wrong record")
        loaded_ref = _canonical_target_ref(loaded["ref"])
        context_ref = _canonical_artifact_ref(
            loaded["producer_context_ref"], "producer_context_ref"
        )
        stream = _canonical_target_stream(loaded["target_stream"])
        digest = canonical_sha256(stream)
        expected_artifact = ArtifactRef.from_envelope(
            ArtifactEnvelope.create(
                "backtest_target_stream",
                1,
                {
                    "producer_context_ref": context_ref,
                    "target_stream": stream,
                },
            )
        )
        if (
            loaded_ref != target_ref
            or _canonical_artifact_ref(target_ref["artifact_ref"], "target ref")
            != expected_artifact
            or context_ref != producer_context_ref
            or loaded["digest"] != digest
            or digest != expected_digest
            or len(stream["events"]) != expected_event_count
            or (
                expected_stream is not None
                and canonical_bytes(stream) != canonical_bytes(expected_stream)
            )
        ):
            raise ValueError("loaded target does not exactly bind its evidence")
        return loaded
    except Exception as error:
        raise ValueError("TARGET_STORE_INVALID") from error


def _target_context(
    candidate_ref: ArtifactRef,
    foundation: LocalFoundation,
    backtest: Any,
    reservation_at: str,
):
    candidate = _published(
        foundation,
        candidate_ref,
        "strategy_candidate",
        _RESEARCH_ARTIFACT_LOG,
        (3,),
    )
    if set(candidate) != _CANDIDATE_V3_FIELDS:
        raise ValueError("candidate@3 fields are invalid")
    graph, sample_entries = _candidate_graph(
        candidate_ref, foundation, backtest, reservation_at, (3,)
    )
    base_candidate = dict(candidate)
    evidence_wire = base_candidate.pop("selected_target_materialization_evidence_ref")
    graph = replace(
        graph, candidate=ResolvedArtifact(_wire(candidate_ref), base_candidate)
    )

    evidence_ref = _ref_from_wire(evidence_wire)
    evidence = _published(
        foundation,
        evidence_ref,
        "target_materialization_evidence",
        _RESEARCH_ARTIFACT_LOG,
    )
    if set(evidence) != _DISCOVERY_EVIDENCE_FIELDS:
        raise ValueError("discovery target evidence fields are invalid")
    task_ref = _ref_from_wire(evidence["target_build_task_ref"])
    recipe_ref = _ref_from_wire(evidence["target_recipe_ref"])
    trial_ref = _ref_from_wire(candidate["selected_trial_declaration_ref"])
    task = _published(foundation, task_ref, "target_build_task", _RESEARCH_ARTIFACT_LOG)
    recipe = _published(foundation, recipe_ref, "target_recipe", _RESEARCH_ARTIFACT_LOG)
    family = graph.candidate_family.payload
    manifest = graph.execution_manifest.payload
    trial = graph.selected_trial_declaration.payload
    if (
        type(family) is not dict
        or type(manifest) is not dict
        or type(trial) is not dict
    ):
        raise ValueError("candidate graph is incomplete")
    experiment_ref = _ref_from_wire(family["experiment_ref"])
    experiment = _published(
        foundation,
        experiment_ref,
        "experiment_spec",
        _RESEARCH_ARTIFACT_LOG,
        (2,),
    )
    target_outcomes = []
    for outcome_wire in manifest["task_outcome_refs"]:
        outcome_ref = _ref_from_wire(outcome_wire)
        outcome = _published(
            foundation,
            outcome_ref,
            "task_outcome",
            _RESEARCH_EXECUTION_LOG,
        )
        if outcome.get("task_ref") == {
            "kind": "TARGET_BUILD",
            "task_artifact_ref": _wire(task_ref),
        }:
            target_outcomes.append(outcome)
    if (
        len(target_outcomes) != 1
        or target_outcomes[0]
        != {
            "task_ref": {
                "kind": "TARGET_BUILD",
                "task_artifact_ref": _wire(task_ref),
            },
            "state": "COMPLETED",
            "witness": {
                "target_materialization_evidence": {
                    "target_materialization_evidence_ref": _wire(evidence_ref)
                }
            },
        }
        or set(task) != _TARGET_TASK_FIELDS
        or set(recipe) != _TARGET_RECIPE_FIELDS
        or set(experiment) != _EXPERIMENT_V2_FIELDS
        or task["experiment_ref"] != _wire(experiment_ref)
        or task["trial_declaration_ref"] != _wire(trial_ref)
        or task["target_recipe_ref"] != _wire(recipe_ref)
        or evidence["trial_declaration_ref"] != _wire(trial_ref)
        or evidence["target_recipe_ref"] != _wire(recipe_ref)
        or experiment["target_recipe_ref"] != _wire(recipe_ref)
        or experiment["model_build_plan"] is not None
    ):
        raise ValueError("candidate@3 target provenance is invalid")
    strategy_artifact = _canonical_strategy_artifact(recipe["strategy_artifact"])
    discovery_request = {
        "type": "target_materialization_request",
        "schema_version": 1,
        "consumer_ref": _wire(trial_ref),
        "target_recipe_ref": _wire(recipe_ref),
        "market_bundle_ref": deepcopy(trial["data_slice"]["market_bundle_ref"]),
        "dataset_revision": deepcopy(trial["data_slice"]["dataset_revision"]),
        "interval_start": deepcopy(trial["data_slice"]["interval_start"]),
        "interval_end": deepcopy(trial["data_slice"]["interval_end"]),
        "parameter_values": deepcopy(trial["parameter_values"]),
        "seed": deepcopy(trial["seed"]),
    }
    if evidence["materialization_request_hash"] != canonical_sha256(discovery_request):
        raise ValueError("discovery materialization request hash is invalid")
    evidence_entries = [
        entry
        for entry, ref, _ in _entries(foundation, _RESEARCH_ARTIFACT_LOG)
        if ref == evidence_ref
    ]
    if len(evidence_entries) != 1:
        raise ValueError("discovery target evidence publication is ambiguous")
    _require_discovery_reservation(
        foundation,
        trial_ref,
        trial,
        before_ledger_sequence=evidence_entries[0].ledger_sequence,
    )
    _content_hash(
        evidence["materialization_request_hash"], "materialization_request_hash"
    )
    _content_hash(evidence["input_data_hash"], "input_data_hash")
    digest = _content_hash(evidence["target_stream_digest"], "target_stream_digest")
    event_count = evidence["event_count"]
    if type(event_count) is not int or event_count < 0:
        raise ValueError("discovery event_count is invalid")
    discovery_ref = _canonical_target_ref(evidence["target_stream_ref"])
    _verified_target(
        backtest,
        discovery_ref,
        trial_ref,
        expected_digest=digest,
        expected_event_count=event_count,
    )
    return (
        graph,
        sample_entries,
        _TargetContext(
            recipe_ref,
            strategy_artifact,
            discovery_ref,
            deepcopy(trial["parameter_values"]),
            deepcopy(trial["seed"]),
        ),
    )


def _target_plan_payload(
    candidate_ref: ArtifactRef,
    snapshot_ref: ArtifactRef,
    policy: ValidationPolicy,
    context: _TargetContext,
) -> tuple[ValidationPlan, dict[str, object]]:
    plan = build_validation_plan(_wire(candidate_ref), _wire(snapshot_ref), policy)
    payload = _plan_payload(plan)
    payload["target_recipe_ref"] = _wire(context.target_recipe_ref)
    payload["strategy_artifact"] = deepcopy(context.strategy_artifact)
    return plan, payload


def _existing_plan(
    foundation: LocalFoundation,
    candidate_ref: ArtifactRef,
    policy: ValidationPolicy,
    context: _TargetContext,
) -> tuple[ArtifactRef, ValidationPlan, dict[str, object]] | None:
    matches: list[tuple[ArtifactRef, ValidationPlan, dict[str, object]]] = []
    for ref, payload in _artifact_log_payloads(foundation):
        if ref.artifact_type != "validation_plan" or ref.schema_version != 2:
            continue
        if payload.get("candidate_ref") != _wire(candidate_ref):
            continue
        snapshot_wire = payload.get("sample_consumption_snapshot_ref")
        try:
            snapshot_ref = _ref_from_wire(snapshot_wire)
            plan, expected = _target_plan_payload(
                candidate_ref, snapshot_ref, policy, context
            )
        except Exception:
            continue
        if payload == expected:
            matches.append((ref, plan, payload))
    if len(matches) > 1:
        raise ValueError("multiple ValidationPlan@2 values exist")
    return matches[0] if matches else None


def _case_ref(plan_ref: ArtifactRef, case_type: str) -> ArtifactRef:
    return ArtifactRef.from_envelope(
        ArtifactEnvelope.create(
            "validation_case",
            1,
            {"validation_plan_ref": _wire(plan_ref), "case_type": case_type},
        )
    )


def _existing_report(
    foundation: LocalFoundation,
    plan_ref: ArtifactRef,
    plan: ValidationPlan,
    evidence_ref: ArtifactRef,
    admission_result_ref: ArtifactRef,
    admission_payload: dict[str, object],
    oos_case_ref: ArtifactRef,
    assessment_ref: ArtifactRef,
    backtest: object,
) -> PublishedValidationReport | None:
    matches = [
        (ref, payload)
        for ref, payload in _artifact_log_payloads(foundation)
        if ref.artifact_type == "validation_report"
        and ref.schema_version == 2
        and (
            payload.get("validation_plan_ref") == _wire(plan_ref)
            or payload.get("validation_target_materialization_evidence_ref")
            == _wire(evidence_ref)
        )
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("multiple ValidationReport@2 values exist")
    report_ref, payload = matches[0]
    try:
        result_refs = payload["case_result_refs"]
        if type(result_refs) is not list or len(result_refs) != 2:
            raise ValueError("report case result cover is invalid")
        stored_admission_ref = _ref_from_wire(result_refs[0])
        stored_oos_ref = _ref_from_wire(result_refs[1])
        if stored_admission_ref != admission_result_ref:
            raise ValueError("report admission result link is invalid")
        admission_existing = _existing_case_result(
            foundation,
            _ref_from_wire(admission_payload["case_ref"]),
            None,
            plan_ref,
            "evidence_integrity",
        )
        oos_existing = _existing_case_result(
            foundation,
            oos_case_ref,
            evidence_ref,
            plan_ref,
            "out_of_sample",
        )
        if (
            admission_existing != (admission_result_ref, admission_payload)
            or oos_existing is None
            or oos_existing[0] != stored_oos_ref
        ):
            raise ValueError("report case result payload is invalid")
        _verify_completed_oos_payload(
            plan,
            oos_case_ref,
            evidence_ref,
            oos_existing[1],
            backtest,
        )
        _published(
            foundation,
            assessment_ref,
            "sample_integrity_assessment",
            _ARTIFACT_LOG,
            (1,),
        )
        expected = _report_payload_from_results(
            plan_ref,
            admission_result_ref,
            admission_payload,
            stored_oos_ref,
            oos_existing[1],
            assessment_ref,
            evidence_ref,
        )
        if (
            expected is None
            or set(payload) != _REPORT_V2_FIELDS
            or payload != expected
            or _published(
                foundation,
                report_ref,
                "validation_report",
                _ARTIFACT_LOG,
                (2,),
            )
            != payload
        ):
            raise ValueError("ValidationReport@2 is invalid")
    except Exception as error:
        raise ValueError("ValidationReport@2 is invalid") from error
    return PublishedValidationReport(plan_ref, report_ref)


def _closed_report_candidate(
    foundation: LocalFoundation,
    plan_ref: ArtifactRef,
    evidence_case_ref: ArtifactRef,
    oos_case_ref: ArtifactRef,
    evidence_ref: ArtifactRef | None,
) -> tuple[ArtifactRef, dict[str, Any]] | None:
    case_wires = (_wire(evidence_case_ref), _wire(oos_case_ref))
    result_wires = tuple(
        _wire(ref)
        for ref, payload in _artifact_log_payloads(foundation)
        if ref.artifact_type == "validation_case_result"
        and ref.schema_version == 2
        and payload.get("case_ref") in case_wires
    )
    matches = []
    for ref, payload in _artifact_log_payloads(foundation):
        if ref.artifact_type != "validation_report" or ref.schema_version != 2:
            continue
        result_refs = payload.get("case_result_refs")
        if (
            payload.get("validation_plan_ref") == _wire(plan_ref)
            or (
                type(result_refs) is list
                and any(value in result_wires for value in result_refs)
            )
            or (
                evidence_ref is not None
                and payload.get(
                    "validation_target_materialization_evidence_ref"
                )
                == _wire(evidence_ref)
            )
        ):
            matches.append((ref, payload))
    if len(matches) > 1:
        raise ValueError("multiple ValidationReport@2 values exist")
    return matches[0] if matches else None


def _closed_report_replay(
    foundation: LocalFoundation,
    candidate_ref: ArtifactRef,
    plan_ref: ArtifactRef,
    plan: ValidationPlan,
    snapshot_ref: ArtifactRef,
    policy: ValidationPolicy,
    graph: CandidateGraph,
    context: _TargetContext,
    reservation_at: str,
    backtest: Any,
) -> PublishedValidationReport | None:
    evidence_case_ref = _case_ref(plan_ref, "evidence_integrity")
    oos_case_ref = _case_ref(plan_ref, "out_of_sample")
    recovered_evidence = _existing_evidence(
        foundation,
        backtest,
        candidate_ref,
        oos_case_ref,
        policy,
        context,
        reservation_at,
    )
    evidence_ref = None if recovered_evidence is None else recovered_evidence[0]
    stored_report = _closed_report_candidate(
        foundation,
        plan_ref,
        evidence_case_ref,
        oos_case_ref,
        evidence_ref,
    )
    if stored_report is None:
        return None
    report_ref, report_payload = stored_report
    evidence_wire = report_payload.get("validation_target_materialization_evidence_ref")
    if evidence_wire is not None:
        if recovered_evidence is None:
            raise ValueError("ValidationReport@2 target evidence is missing")
        if evidence_wire != _wire(evidence_ref):
            raise ValueError("ValidationReport@2 target evidence link is invalid")

    assessment_ref = _ref_from_wire(report_payload["sample_integrity_ref"])
    try:
        admission_evidence = _snapshot_evidence(
            foundation, snapshot_ref, assessment_ref, policy
        )
    except (FoundationFailure, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "ValidationReport@2 sample integrity link is invalid"
        ) from error
    admission = assess_admission(plan, graph, admission_evidence)
    admission_payload = _case_result_payload(evidence_case_ref, admission, None)
    admission_existing = _existing_case_result(
        foundation,
        evidence_case_ref,
        None,
        plan_ref,
        "evidence_integrity",
    )
    if admission_existing is None or admission_existing[1] != admission_payload:
        raise ValueError("ValidationReport@2 admission result is invalid")
    admission_result_ref, _ = admission_existing

    if recovered_evidence is None:
        if admission.outcome == "PASS":
            raise ValueError("ValidationReport@2 target evidence link is invalid")
        blocked = CaseResult(
            plan,
            "out_of_sample",
            "FAILED" if admission.outcome == "FAILED" else "BLOCKED",
            admission.reason_codes,
            admission.limitations,
            None,
        )
        blocked_payload = _case_result_payload(oos_case_ref, blocked, None)
        blocked_existing = _existing_case_result(
            foundation, oos_case_ref, None, plan_ref, "out_of_sample"
        )
        if blocked_existing is None or blocked_existing[1] != blocked_payload:
            raise ValueError("ValidationReport@2 OOS result is invalid")
        expected = {
            "validation_plan_ref": _wire(plan_ref),
            "result": "inconclusive",
            "case_result_refs": [
                _wire(admission_result_ref),
                _wire(blocked_existing[0]),
            ],
            "threshold_evaluations": [],
            "sample_integrity_ref": _wire(assessment_ref),
            "limitations": sorted(
                set(admission.limitations) | set(blocked.limitations)
            ),
            "validation_target_materialization_evidence_ref": None,
        }
        if (
            set(report_payload) != _REPORT_V2_FIELDS
            or report_payload != expected
            or _published(
                foundation,
                report_ref,
                "validation_report",
                _ARTIFACT_LOG,
                (2,),
            )
            != report_payload
        ):
            raise ValueError("ValidationReport@2 is invalid")
        return PublishedValidationReport(plan_ref, report_ref)

    evidence_ref, _ = recovered_evidence
    report = _existing_report(
        foundation,
        plan_ref,
        plan,
        evidence_ref,
        admission_result_ref,
        admission_payload,
        oos_case_ref,
        assessment_ref,
        backtest,
    )
    if report is None:
        raise ValueError("ValidationReport@2 is invalid")
    return report


def _reservation_bytes(
    case_ref: ArtifactRef,
    policy: ValidationPolicy,
    reservation_at: str,
) -> bytes:
    record = SampleConsumptionRecord(
        policy.holdout.dataset_revision,
        policy.holdout.interval_start,
        policy.holdout.interval_end,
        "validation",
        _consumer_id(case_ref),
        reservation_at,
    )
    return canonical_bytes(
        ArtifactEnvelope.create(
            "sample_consumption_append",
            1,
            {
                "record": {
                    "dataset_revision": record.dataset_revision,
                    "interval_start": record.interval_start,
                    "interval_end": record.interval_end,
                    "purpose": record.purpose,
                    "consumer_id": record.consumer_id,
                    "consumed_at": record.consumed_at,
                },
                "producer_ref": case_ref,
            },
        )
    )


def _require_reservation(
    foundation: LocalFoundation,
    case_ref: ArtifactRef,
    policy: ValidationPolicy,
    reservation_at: str,
    *,
    before_ledger_sequence: int | None = None,
) -> None:
    expected_payload = _reservation_bytes(case_ref, policy, reservation_at)
    expected_event = canonical_sha256(
        (
            "sample-consumption-append-v1",
            case_ref,
            policy.holdout.dataset_revision,
            policy.holdout.interval_start,
            policy.holdout.interval_end,
            "validation",
        )
    )
    matches = tuple(
        entry
        for entry in foundation.entries(_SAMPLE_LOG)
        if entry.payload == expected_payload
    )
    if (
        len(matches) != 1
        or matches[0].event_id != expected_event
        or reservation_at > matches[0].accepted_at
        or (
            before_ledger_sequence is not None
            and matches[0].ledger_sequence >= before_ledger_sequence
        )
    ):
        raise _OosReservationConflict("OOS reservation is not exact and preceding")


def _target_request(
    case_ref: ArtifactRef,
    policy: ValidationPolicy,
    context: _TargetContext,
) -> dict[str, object]:
    return {
        "type": "target_materialization_request",
        "schema_version": 1,
        "consumer_ref": _wire(case_ref),
        "target_recipe_ref": _wire(context.target_recipe_ref),
        "market_bundle_ref": _plain(policy.holdout.market_bundle_ref),
        "dataset_revision": _plain(policy.holdout.dataset_revision),
        "interval_start": _plain(policy.holdout.interval_start),
        "interval_end": _plain(policy.holdout.interval_end),
        "parameter_values": deepcopy(context.parameter_values),
        "seed": deepcopy(context.seed),
    }


def _existing_evidence(
    foundation: LocalFoundation,
    backtest: Any,
    candidate_ref: ArtifactRef,
    case_ref: ArtifactRef,
    policy: ValidationPolicy,
    context: _TargetContext,
    reservation_at: str,
) -> tuple[ArtifactRef, ValidationTargetMaterializationEvidence] | None:
    matches: list[tuple[Any, ArtifactRef, dict[str, Any]]] = []
    for entry, ref, payload in _entries(foundation, _ARTIFACT_LOG):
        if (
            ref.artifact_type == "validation_target_materialization_evidence"
            and ref.schema_version == 1
            and payload.get("validation_case_ref") == _wire(case_ref)
        ):
            matches.append((entry, ref, payload))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("multiple Validation target evidence values exist")
    entry, ref, payload = matches[0]
    if set(payload) != _VALIDATION_EVIDENCE_FIELDS:
        raise ValueError("Validation target evidence fields are invalid")
    evidence = ValidationTargetMaterializationEvidence(
        payload["validation_case_ref"],
        payload["candidate_ref"],
        payload["target_recipe_ref"],
        payload["materialization_request_hash"],
        payload["input_data_hash"],
        payload["target_stream_ref"],
        payload["target_stream_digest"],
        payload["event_count"],
    )
    request_hash = canonical_sha256(_target_request(case_ref, policy, context))
    if (
        evidence.payload != payload
        or evidence.candidate_ref != _wire(candidate_ref)
        or evidence.target_recipe_ref != _wire(context.target_recipe_ref)
        or evidence.materialization_request_hash != request_hash
        or canonical_bytes(evidence.target_stream_ref)
        == canonical_bytes(context.discovery_target_ref)
    ):
        raise ValueError("Validation target evidence links are invalid")
    _require_reservation(
        foundation,
        case_ref,
        policy,
        reservation_at,
        before_ledger_sequence=entry.ledger_sequence,
    )
    _verified_target(
        backtest,
        evidence.target_stream_ref,
        case_ref,
        expected_digest=evidence.target_stream_digest,
        expected_event_count=evidence.event_count,
    )
    return ref, evidence


def _materialize_evidence(
    foundation: LocalFoundation,
    candidate_ref: ArtifactRef,
    case_ref: ArtifactRef,
    policy: ValidationPolicy,
    context: _TargetContext,
    materializer: Any,
    backtest: Any,
    reservation_at: str,
) -> tuple[ArtifactRef, ValidationTargetMaterializationEvidence]:
    try:
        materializer_strategy = _canonical_strategy_artifact(
            materializer.strategy_artifact
        )
    except Exception as error:
        raise ValueError("TARGET_MATERIALIZATION_INVALID") from error
    if canonical_bytes(materializer_strategy) != canonical_bytes(
        context.strategy_artifact
    ):
        raise ValueError("TARGET_MATERIALIZATION_INVALID")
    request = _target_request(case_ref, policy, context)
    request_hash = canonical_sha256(request)
    materializer_request = _plain(request)
    try:
        result_value = materializer.materialize_target(materializer_request)
    except Exception as error:
        raise ValueError("TARGET_MATERIALIZATION_INVALID") from error
    if canonical_bytes(materializer_request) != canonical_bytes(request):
        raise ValueError("TARGET_MATERIALIZATION_INVALID")
    result = _plain(result_value)
    if type(result) is not dict or set(result) != _TARGET_RESULT_FIELDS:
        raise ValueError("TARGET_MATERIALIZATION_INVALID")
    try:
        strategy = _canonical_strategy_artifact(result["strategy_artifact"])
        stream = _canonical_target_stream(result["target_stream"])
        input_data_hash = _content_hash(result["input_data_hash"], "input_data_hash")
    except ValueError as error:
        raise ValueError("TARGET_MATERIALIZATION_INVALID") from error
    if (
        result["type"] != "target_materialization_result"
        or result["schema_version"] != 1
        or result["request_hash"] != request_hash
        or canonical_bytes(strategy) != canonical_bytes(context.strategy_artifact)
        or canonical_bytes(strategy) != canonical_bytes(materializer_strategy)
    ):
        raise ValueError("TARGET_MATERIALIZATION_INVALID")
    try:
        target_ref = _canonical_target_ref(
            backtest.publish_target(_wire(case_ref), deepcopy(stream))
        )
        digest = canonical_sha256(stream)
        _verified_target(
            backtest,
            target_ref,
            case_ref,
            expected_digest=digest,
            expected_event_count=len(stream["events"]),
            expected_stream=stream,
        )
    except Exception as error:
        raise ValueError("TARGET_STORE_INVALID") from error
    if canonical_bytes(target_ref) == canonical_bytes(context.discovery_target_ref):
        raise ValueError("TARGET_SUBSTITUTION_INVALID")
    evidence = ValidationTargetMaterializationEvidence(
        _wire(case_ref),
        _wire(candidate_ref),
        _wire(context.target_recipe_ref),
        request_hash,
        input_data_hash,
        target_ref,
        digest,
        len(stream["events"]),
    )
    try:
        ref = _publish(
            foundation,
            "validation_target_materialization_evidence",
            1,
            evidence.payload,
        )
    except Exception as error:
        recovered = _existing_evidence(
            foundation,
            backtest,
            candidate_ref,
            case_ref,
            policy,
            context,
            reservation_at,
        )
        if recovered is None:
            raise ValueError("TARGET_EVIDENCE_PUBLICATION_FAILED") from error
        return recovered
    return ref, evidence


def _evidence_payload(value: object) -> object:
    from .runtime import _evidence_payload as old_evidence_payload

    return old_evidence_payload(value)


def _case_result_payload(
    case_ref: ArtifactRef,
    result: CaseResult,
    evidence_ref: ArtifactRef | None,
) -> dict[str, object]:
    return {
        "case_ref": _wire(case_ref),
        "outcome": result.outcome,
        "reason_codes": list(result.reason_codes),
        "limitations": list(result.limitations),
        "evidence": _evidence_payload(result.evidence),
        "threshold_evaluation": (
            None
            if result.threshold_evaluation is None
            else _threshold_payload(result.threshold_evaluation)
        ),
        "validation_target_materialization_evidence_ref": (
            None if evidence_ref is None else _wire(evidence_ref)
        ),
    }


def _case_result_shapes_valid(payload: dict[str, Any]) -> bool:
    evidence = payload["evidence"]
    if evidence is not None:
        if type(evidence) is not dict or frozenset(evidence) not in {
            frozenset(_COMPLETED_CASE_EVIDENCE_FIELDS),
            frozenset(_TERMINAL_CASE_EVIDENCE_FIELDS),
            frozenset(_PROVIDER_FAILURE_EVIDENCE_FIELDS),
            frozenset(_SAMPLE_CASE_EVIDENCE_FIELDS),
        }:
            return False
        if set(evidence) == _PROVIDER_FAILURE_EVIDENCE_FIELDS and (
            type(evidence["code"]) is not str or not evidence["code"]
        ):
            return False
        if set(evidence) == _TERMINAL_CASE_EVIDENCE_FIELDS and (
            evidence["status"] not in {"BLOCKED", "FAILED", "CANCELLED"}
            or evidence["durable_evidence_ref"] is None
        ):
            return False
        if set(evidence) == _SAMPLE_CASE_EVIDENCE_FIELDS and (
            type(evidence["untouched"]) is not bool
            or type(evidence["conflicting_records"]) is not list
        ):
            return False
    threshold = payload["threshold_evaluation"]
    return threshold is None or (
        type(threshold) is dict
        and set(threshold) == _THRESHOLD_EVALUATION_FIELDS
        and all(
            type(threshold[name]) is str and threshold[name]
            for name in ("metric_key", "observed", "operator", "threshold")
        )
        and type(threshold["passed"]) is bool
        and type(threshold["trade_count"]) is int
        and type(threshold["trade_count"]) is not bool
        and type(threshold["minimum_trade_count"]) is int
        and type(threshold["minimum_trade_count"]) is not bool
    )


def _existing_case_result(
    foundation: LocalFoundation,
    case_ref: ArtifactRef,
    evidence_ref: ArtifactRef | None,
    plan_ref: ArtifactRef,
    case_type: str,
) -> tuple[ArtifactRef, dict[str, object]] | None:
    matches = [
        (ref, payload)
        for ref, payload in _artifact_log_payloads(foundation)
        if ref.artifact_type == "validation_case_result"
        and ref.schema_version == 2
        and (
            payload.get("case_ref") == _wire(case_ref)
            or (
                evidence_ref is not None
                and payload.get("validation_target_materialization_evidence_ref")
                == _wire(evidence_ref)
            )
        )
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("multiple CaseResult@2 values exist")
    ref, payload = matches[0]
    expected_evidence = None if evidence_ref is None else _wire(evidence_ref)
    expected_case = {
        "validation_plan_ref": _wire(plan_ref),
        "case_type": case_type,
    }
    if (
        set(payload) != _CASE_RESULT_V2_FIELDS
        or payload["case_ref"] != _wire(case_ref)
        or payload["validation_target_materialization_evidence_ref"]
        != expected_evidence
        or payload["outcome"]
        not in {"PASS", "FAIL", "INCONCLUSIVE", "BLOCKED", "FAILED"}
        or type(payload["reason_codes"]) is not list
        or any(type(value) is not str or not value for value in payload["reason_codes"])
        or len(payload["reason_codes"]) != len(set(payload["reason_codes"]))
        or type(payload["limitations"]) is not list
        or any(type(value) is not str or not value for value in payload["limitations"])
        or payload["limitations"] != sorted(set(payload["limitations"]))
        or not _case_result_shapes_valid(payload)
        or _published(
            foundation,
            case_ref,
            "validation_case",
            _ARTIFACT_LOG,
            (1,),
        )
        != expected_case
        or _published(
            foundation,
            ref,
            "validation_case_result",
            _ARTIFACT_LOG,
            (2,),
        )
        != payload
    ):
        raise ValueError("CaseResult@2 is invalid")
    return ref, payload


def _verify_completed_oos_payload(
    plan: ValidationPlan,
    case_ref: ArtifactRef,
    evidence_ref: ArtifactRef,
    payload: dict[str, Any],
    backtest: Any,
) -> None:
    evidence = payload["evidence"]
    if type(evidence) is not dict:
        raise ValueError("CaseResult@2 durable evidence payload is invalid")
    if set(evidence) == _COMPLETED_CASE_EVIDENCE_FIELDS:
        completed = _load_completed(backtest, evidence["publication_ref"])
        analysis = _load_analysis(backtest, evidence["analysis_ref"])
        recovered = assess_oos(
            plan,
            OosObservation(plan, _wire(case_ref), _wire(case_ref), dict(completed)),
            AnalysisObservation(plan, _wire(case_ref), dict(analysis)),
        )
    elif set(evidence) == _TERMINAL_CASE_EVIDENCE_FIELDS:
        terminal = backtest.load_terminal(_plain(evidence["durable_evidence_ref"]))
        if type(terminal) is not dict:
            raise ValueError("CaseResult@2 terminal evidence is invalid")
        recovered = assess_oos(
            plan,
            OosObservation(plan, _wire(case_ref), _wire(case_ref), dict(terminal)),
            None,
        )
    elif set(evidence) == _PROVIDER_FAILURE_EVIDENCE_FIELDS:
        recovered = assess_oos(plan, ProviderFailure(evidence["code"]), None)
    else:
        raise ValueError("CaseResult@2 durable evidence shape is invalid")
    if _case_result_payload(case_ref, recovered, evidence_ref) != payload:
        raise ValueError("CaseResult@2 durable evidence payload is invalid")


def _run_target(
    plan: ValidationPlan,
    case_ref: ArtifactRef,
    evidence: ValidationTargetMaterializationEvidence,
    backtest: Any,
) -> CaseResult:
    try:
        request = backtest.prepare_target(
            _wire(case_ref), deepcopy(evidence.target_stream_ref)
        )
    except FoundationFailure:
        raise
    except Exception as error:
        raise ValueError("TARGET_PREPARATION_FAILED") from error
    try:
        run_ref = backtest.run(request)
    except FoundationFailure:
        raise
    except Exception as error:
        return assess_oos(plan, ProviderFailure(_failure_code(error)), None)
    try:
        if _is_terminal_ref(run_ref):
            terminal = backtest.load_terminal(run_ref)
            return assess_oos(
                plan,
                OosObservation(plan, _wire(case_ref), _wire(case_ref), terminal),
                None,
            )
        completed = _load_completed(backtest, run_ref)
    except FoundationFailure:
        raise
    except Exception as error:
        return assess_oos(plan, ProviderFailure(_failure_code(error)), None)
    observation = OosObservation(
        plan, _wire(case_ref), _wire(case_ref), dict(completed)
    )
    try:
        analysis_ref = backtest.derive(
            run_ref, _plain(plan.oos_rule.metric_profile_ref)
        )
        analysis = _load_analysis(backtest, analysis_ref)
    except FoundationFailure:
        raise
    except Exception as error:
        return assess_oos(plan, observation, ProviderFailure(_failure_code(error)))
    return assess_oos(
        plan,
        observation,
        AnalysisObservation(plan, _wire(case_ref), dict(analysis)),
    )


def _report_payload_from_results(
    plan_ref: ArtifactRef,
    evidence_result_ref: ArtifactRef,
    evidence_result: dict[str, Any],
    oos_result_ref: ArtifactRef,
    oos_result: dict[str, Any],
    assessment_ref: ArtifactRef,
    target_evidence_ref: ArtifactRef,
) -> dict[str, object] | None:
    if "FAILED" in {evidence_result["outcome"], oos_result["outcome"]}:
        return None
    outcomes = (evidence_result["outcome"], oos_result["outcome"])
    result = (
        "rejected"
        if "FAIL" in outcomes
        else (
            "inconclusive"
            if any(value in {"BLOCKED", "INCONCLUSIVE"} for value in outcomes)
            else "supported"
        )
    )
    threshold = oos_result["threshold_evaluation"]
    limitations = sorted(
        set(evidence_result["limitations"]) | set(oos_result["limitations"])
    )
    return {
        "validation_plan_ref": _wire(plan_ref),
        "result": result,
        "case_result_refs": [_wire(evidence_result_ref), _wire(oos_result_ref)],
        "threshold_evaluations": [] if threshold is None else [threshold],
        "sample_integrity_ref": _wire(assessment_ref),
        "limitations": limitations,
        "validation_target_materialization_evidence_ref": _wire(target_evidence_ref),
    }


def _require_ports(materializer: Any, backtest: Any) -> None:
    if not callable(getattr(materializer, "materialize_target", None)):
        raise TypeError(
            "materializer must expose strategy_artifact and materialize_target"
        )
    operations = (
        "run",
        "derive",
        "load_completed",
        "load_terminal",
        "load_analysis",
        "publish_target",
        "load_target",
        "prepare_target",
    )
    if not all(callable(getattr(backtest, name, None)) for name in operations):
        raise TypeError("backtest must expose the exact target Validation operations")


def validate_target_candidate(
    candidate_ref: ArtifactRef,
    policy: ValidationPolicy,
    reservation_at: str,
    foundation: LocalFoundation,
    sample_ledger: SampleConsumptionLedger,
    materializer: object,
    backtest: object,
) -> PublishedValidationReport | NoReport:
    """Validate an exact StrategyCandidate@3 through independent OOS targets."""

    if type(candidate_ref) is not ArtifactRef:
        raise TypeError("candidate_ref must be an ArtifactRef")
    candidate_ref = ArtifactRef(
        candidate_ref.artifact_type,
        candidate_ref.schema_version,
        candidate_ref.content_hash,
    )
    if (
        candidate_ref.artifact_type != "strategy_candidate"
        or candidate_ref.schema_version != 3
    ):
        raise TypeError("candidate_ref must reference exact StrategyCandidate@3")
    if type(policy) is not ValidationPolicy:
        raise TypeError("policy must be a ValidationPolicy")
    policy = ValidationPolicy(
        policy.accepted_backtest_grades,
        policy.accepted_metric_profile_refs,
        policy.holdout,
        policy.oos_rule,
        policy.decision_rule,
    )
    reservation_at = build_snapshot((), as_of=reservation_at).as_of
    if type(foundation) is not LocalFoundation:
        raise TypeError("foundation must be a LocalFoundation")
    if type(sample_ledger) is not SampleConsumptionLedger:
        raise TypeError("sample_ledger must be a SampleConsumptionLedger")
    _require_ports(materializer, backtest)

    try:
        graph, sample_entries, context = _target_context(
            candidate_ref, foundation, backtest, reservation_at
        )
        existing = _existing_plan(foundation, candidate_ref, policy, context)
        if existing is None:
            preflight = _preflight_admission(
                candidate_ref, policy, graph, sample_entries, reservation_at
            )
            if preflight.reason_codes == ("CANDIDATE_PROVENANCE_INVALID",):
                return NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
            if "SAMPLE_RESERVATION_COVERAGE_MISSING" in preflight.reason_codes:
                return NoReport(None, "SAMPLE_RESERVATION_COVERAGE_MISSING")
            snapshot_ref = sample_ledger.freeze_snapshot()
            plan, plan_payload = _target_plan_payload(
                candidate_ref, snapshot_ref, policy, context
            )
            plan_ref = _publish(foundation, "validation_plan", 2, plan_payload)
        else:
            plan_ref, plan, plan_payload = existing
            snapshot_ref = _ref_from_wire(
                plan_payload["sample_consumption_snapshot_ref"]
            )
    except FoundationFailure as error:
        if error.code == "ARTIFACT_NOT_FOUND":
            return NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
        raise
    except Exception:
        return NoReport(None, "CANDIDATE_PROVENANCE_INVALID")

    if existing is not None:
        try:
            closed_report = _closed_report_replay(
                foundation,
                candidate_ref,
                plan_ref,
                plan,
                snapshot_ref,
                policy,
                graph,
                context,
                reservation_at,
                backtest,
            )
        except FoundationFailure:
            raise
        except _OosReservationConflict:
            return NoReport(plan_ref, "SAMPLE_LEDGER_CONFLICT")
        except (KeyError, TypeError):
            return NoReport(plan_ref, "CASE_COVER_INVALID")
        except ValueError as error:
            code = str(error)
            if "CaseResult@2" in code or "ValidationReport@2" in code:
                return NoReport(plan_ref, "CASE_COVER_INVALID")
            if code in {
                "TARGET_MATERIALIZATION_INVALID",
                "TARGET_STORE_INVALID",
                "TARGET_SUBSTITUTION_INVALID",
                "TARGET_EVIDENCE_PUBLICATION_FAILED",
            }:
                return NoReport(plan_ref, code)
            return NoReport(plan_ref, "TARGET_MATERIALIZATION_INVALID")
        if closed_report is not None:
            return closed_report

    assessment_ref = sample_ledger.assess_holdout(snapshot_ref, policy.holdout)
    admission_evidence = _snapshot_evidence(
        foundation, snapshot_ref, assessment_ref, policy
    )
    evidence_case_ref = _publish(
        foundation,
        "validation_case",
        1,
        {"validation_plan_ref": _wire(plan_ref), "case_type": "evidence_integrity"},
    )
    admission = assess_admission(plan, graph, admission_evidence)
    admission_payload = _case_result_payload(evidence_case_ref, admission, None)
    try:
        admission_existing = _existing_case_result(
            foundation,
            evidence_case_ref,
            None,
            plan_ref,
            "evidence_integrity",
        )
    except ValueError:
        return NoReport(plan_ref, "CASE_COVER_INVALID")
    if admission_existing is None:
        admission_result_ref = _publish(
            foundation, "validation_case_result", 2, admission_payload
        )
    else:
        admission_result_ref, stored_admission = admission_existing
        if stored_admission != admission_payload:
            return NoReport(plan_ref, "CASE_COVER_INVALID")

    oos_case_ref = _publish(
        foundation,
        "validation_case",
        1,
        {"validation_plan_ref": _wire(plan_ref), "case_type": "out_of_sample"},
    )
    if admission.outcome != "PASS":
        blocked = CaseResult(
            plan,
            "out_of_sample",
            "FAILED" if admission.outcome == "FAILED" else "BLOCKED",
            admission.reason_codes,
            admission.limitations,
            None,
        )
        blocked_payload = _case_result_payload(oos_case_ref, blocked, None)
        try:
            existing_blocked = _existing_case_result(
                foundation,
                oos_case_ref,
                None,
                plan_ref,
                "out_of_sample",
            )
        except ValueError:
            return NoReport(plan_ref, "CASE_COVER_INVALID")
        if existing_blocked is None:
            _publish(foundation, "validation_case_result", 2, blocked_payload)
        elif existing_blocked[1] != blocked_payload:
            return NoReport(plan_ref, "CASE_COVER_INVALID")
        if blocked.outcome == "FAILED":
            return NoReport(plan_ref, blocked.reason_codes[0])
        try:
            return _publish_blocked_report(
                foundation,
                plan_ref,
                admission_result_ref,
                admission_payload,
                oos_case_ref,
                blocked_payload,
                assessment_ref,
            )
        except ValueError:
            return NoReport(plan_ref, "CASE_COVER_INVALID")

    try:
        sample_ledger.reserve(
            SampleConsumptionRecord(
                policy.holdout.dataset_revision,
                policy.holdout.interval_start,
                policy.holdout.interval_end,
                "validation",
                _consumer_id(oos_case_ref),
                reservation_at,
            ),
            oos_case_ref,
        )
        _require_reservation(foundation, oos_case_ref, policy, reservation_at)
    except FoundationFailure:
        raise
    except Exception:
        return NoReport(plan_ref, "SAMPLE_LEDGER_CONFLICT")
    try:
        recovered_evidence = _existing_evidence(
            foundation,
            backtest,
            candidate_ref,
            oos_case_ref,
            policy,
            context,
            reservation_at,
        )
        if recovered_evidence is None:
            target_evidence_ref, target_evidence = _materialize_evidence(
                foundation,
                candidate_ref,
                oos_case_ref,
                policy,
                context,
                materializer,
                backtest,
                reservation_at,
            )
        else:
            target_evidence_ref, target_evidence = recovered_evidence
        report = _existing_report(
            foundation,
            plan_ref,
            plan,
            target_evidence_ref,
            admission_result_ref,
            admission_payload,
            oos_case_ref,
            assessment_ref,
            backtest,
        )
        if report is not None:
            return report
    except FoundationFailure:
        raise
    except _OosReservationConflict:
        return NoReport(plan_ref, "SAMPLE_LEDGER_CONFLICT")
    except ValueError as error:
        code = str(error)
        if "CaseResult@2" in code or "ValidationReport@2" in code:
            code = "CASE_COVER_INVALID"
        elif code not in {
            "TARGET_MATERIALIZATION_INVALID",
            "TARGET_STORE_INVALID",
            "TARGET_SUBSTITUTION_INVALID",
            "TARGET_EVIDENCE_PUBLICATION_FAILED",
        }:
            code = "TARGET_MATERIALIZATION_INVALID"
        return NoReport(plan_ref, code)

    try:
        existing_oos = _existing_case_result(
            foundation,
            oos_case_ref,
            target_evidence_ref,
            plan_ref,
            "out_of_sample",
        )
        if existing_oos is not None:
            _verify_completed_oos_payload(
                plan,
                oos_case_ref,
                target_evidence_ref,
                existing_oos[1],
                backtest,
            )
    except ValueError:
        return NoReport(plan_ref, "CASE_COVER_INVALID")
    if existing_oos is None:
        try:
            oos = _run_target(plan, oos_case_ref, target_evidence, backtest)
        except FoundationFailure:
            raise
        except ValueError:
            return NoReport(plan_ref, "TARGET_PREPARATION_FAILED")
        oos_payload = _case_result_payload(oos_case_ref, oos, target_evidence_ref)
        oos_result_ref = _publish(foundation, "validation_case_result", 2, oos_payload)
    else:
        oos_result_ref, oos_payload = existing_oos

    report_payload = _report_payload_from_results(
        plan_ref,
        admission_result_ref,
        admission_payload,
        oos_result_ref,
        oos_payload,
        assessment_ref,
        target_evidence_ref,
    )
    if report_payload is None:
        reasons = oos_payload["reason_codes"]
        return NoReport(
            plan_ref,
            reasons[0] if type(reasons) is list and reasons else "NO_REPORT",
        )
    report_ref = _publish(foundation, "validation_report", 2, report_payload)
    return PublishedValidationReport(plan_ref, report_ref)


def _publish_blocked_report(
    foundation: LocalFoundation,
    plan_ref: ArtifactRef,
    admission_result_ref: ArtifactRef,
    admission_payload: dict[str, Any],
    oos_case_ref: ArtifactRef,
    oos_payload: dict[str, Any],
    assessment_ref: ArtifactRef,
) -> PublishedValidationReport:
    oos_existing = _existing_case_result(
        foundation,
        oos_case_ref,
        None,
        plan_ref,
        "out_of_sample",
    )
    if oos_existing is None:
        oos_result_ref = _publish(foundation, "validation_case_result", 2, oos_payload)
    else:
        oos_result_ref, _ = oos_existing
    payload = {
        "validation_plan_ref": _wire(plan_ref),
        "result": "inconclusive",
        "case_result_refs": [_wire(admission_result_ref), _wire(oos_result_ref)],
        "threshold_evaluations": [],
        "sample_integrity_ref": _wire(assessment_ref),
        "limitations": sorted(
            set(admission_payload["limitations"]) | set(oos_payload["limitations"])
        ),
        "validation_target_materialization_evidence_ref": None,
    }
    matches = [
        (ref, stored)
        for ref, stored in _artifact_log_payloads(foundation)
        if ref.artifact_type == "validation_report"
        and ref.schema_version == 2
        and stored.get("validation_plan_ref") == _wire(plan_ref)
    ]
    if len(matches) > 1:
        raise ValueError("multiple blocked ValidationReport@2 values exist")
    if matches:
        report_ref, stored = matches[0]
        if (
            stored != payload
            or _published(
                foundation,
                report_ref,
                "validation_report",
                _ARTIFACT_LOG,
                (2,),
            )
            != stored
        ):
            raise ValueError("blocked ValidationReport@2 is invalid")
    else:
        report_ref = _publish(foundation, "validation_report", 2, payload)
    return PublishedValidationReport(plan_ref, report_ref)


__all__ = [
    "ValidationTargetMaterializationEvidence",
    "validate_target_candidate",
]
