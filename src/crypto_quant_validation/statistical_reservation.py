"""Strict owner-proof projection and append-before-read statistical registration.

No Research DTO, Reader, financial result, or global untouched/source claim is
created here. Binding and sample append are separate immutable commits.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NoReturn

from crypto_quant_domain import (
    ArtifactEnvelope,
    ArtifactRef,
    canonical_bytes,
    canonical_sha256,
)
from crypto_quant_foundation import (
    LocalFoundation,
    LogCheckpoint,
    LogEntry,
    LogEntryRef,
)

from .ledger import SampleConsumptionLedger
from .sample_consumption import (
    SampleConsumptionRecord,
    assess_untouched_holdout,
    build_snapshot,
)

_RESEARCH_LOG = "research.artifacts.v1"
_ARTIFACT_LOG = "validation.artifacts.v1"
_SAMPLE_LOG = "validation.sample-consumption.v1"
_BINDING_LOG = "validation.statistical-read-bindings.v1"
_STUDY_TYPE = "statistical_study_declaration"
_SNAPSHOT_TYPE = "sample_consumption_ledger_snapshot"
_ASSESSMENT_TYPE = "sample_integrity_assessment"
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z\Z")
_MODELS = ("CLIMATOLOGY", "PRICE_ONLY", "PRICE_VOLUME", "SHUFFLED_VOLUME")
_DATA_KEYS = {"market_bundle_ref", "dataset_revision", "interval_start", "interval_end"}
_RECORD_KEYS = {
    "dataset_revision",
    "interval_start",
    "interval_end",
    "purpose",
    "consumer_id",
    "consumed_at",
}
_HASH_FIELDS = ("feature_rows_hash", "model_cards_hash", "predictions_hash")
_STUDY_HASH_FIELDS = (
    "research_design_hash",
    "core_code_hash",
    "adapter_code_hash",
    "input_manifest_hash",
    "calendar_manifest_hash",
)
# Wire interpretation only; no duplicate Research declarations/serializer.
_PHASES = {
    "TRAIN_FEATURE": (
        "TRAIN",
        "statistical_feature_read_task",
        "feature_build",
        (),
        (),
    ),
    "TRAIN_FIT": (
        "TRAIN",
        "statistical_training_read_task",
        "model_training",
        ("feature_rows_hash",),
        ("TRAIN_FEATURE",),
    ),
    "DEVELOPMENT_FEATURE": (
        "DEVELOPMENT",
        "statistical_feature_read_task",
        "feature_build",
        ("model_cards_hash",),
        ("TRAIN_FIT",),
    ),
    "HELD_OUT_FEATURE": (
        "HELD_OUT",
        "statistical_feature_read_task",
        "feature_build",
        ("model_cards_hash",),
        ("TRAIN_FIT",),
    ),
    "DEVELOPMENT_OUTCOME": (
        "DEVELOPMENT",
        "statistical_outcome_read_task",
        "validation",
        ("model_cards_hash", "predictions_hash"),
        ("DEVELOPMENT_FEATURE", "HELD_OUT_FEATURE"),
    ),
    "HELD_OUT_OUTCOME": (
        "HELD_OUT",
        "statistical_outcome_read_task",
        "validation",
        ("model_cards_hash", "predictions_hash"),
        ("DEVELOPMENT_FEATURE", "HELD_OUT_FEATURE"),
    ),
}
_PURPOSES = {
    "feature_build_task": "feature_build",
    "model_training_task": "model_training",
    "trial_declaration": "discovery",
    "selection_declaration": "selection",
    "validation_case": "validation",
    **{item[1]: item[2] for item in _PHASES.values()},
    "retrospective_input_read_task": "discovery",
    "retrospective_feature_read_task": "feature_build",
    "retrospective_training_read_task": "model_training",
    "retrospective_outcome_read_task": "validation",
}
_CODES = (
    "STATISTICAL_STUDY_INVALID",
    "STATISTICAL_READ_TASK_INVALID",
    "STATISTICAL_PUBLICATION_MISSING",
    "STATISTICAL_SCOPE_CONFLICT",
    "STATISTICAL_PREREQUISITE_INVALID",
    "STATISTICAL_HOLDOUT_EVIDENCE_REQUIRED",
    "STATISTICAL_RESERVATION_FAILED",
)
Publication = tuple[ArtifactRef, LogEntryRef, LogCheckpoint]
_SampleEntry = tuple[SampleConsumptionRecord, ArtifactRef, LogEntry]
_PhaseEntry = tuple[Mapping[str, Any], ArtifactRef, LogEntry]


class StatisticalReservationError(ValueError):
    def __init__(self, code: str) -> None:
        if code not in _CODES:
            raise ValueError("unknown statistical reservation failure")
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class StatisticalReservationReceipt:
    study_ref: ArtifactRef
    read_task_ref: ArtifactRef
    study_publication_entry_ref: LogEntryRef
    producer_publication_entry_ref: LogEntryRef
    purpose: str
    record: SampleConsumptionRecord
    reservation_entry_ref: LogEntryRef
    heldout_snapshot_ref: ArtifactRef | None
    heldout_assessment_ref: ArtifactRef | None
    heldout_binding_entry_ref: LogEntryRef | None


def _fail(code: str, error: Exception) -> NoReturn:
    raise StatisticalReservationError(code) from error


def _mapping(value: object, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("wire must have exact fields")
    return value


def _text(value: object) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("value must be an explicit nonempty string")
    return value


def _hash(value: object) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ValueError("value must be a canonical sha256 hash")
    return value


def _instant(value: object) -> str:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        raise ValueError("time must be canonical UTC with six fractional digits")
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z")
    return value


def _ref(value: object) -> ArtifactRef:
    if type(value) is not ArtifactRef:
        raise ValueError("reference must be a Domain ArtifactRef")
    if (
        ArtifactRef(value.artifact_type, value.schema_version, value.content_hash)
        != value
    ):
        raise ValueError("reference must be canonical")
    return value


def _ref_from_wire(value: object) -> ArtifactRef:
    wire = _mapping(value, {"type", "artifact_type", "schema_version", "content_hash"})
    ref = ArtifactRef(
        wire["artifact_type"], wire["schema_version"], wire["content_hash"]
    )
    if dict(wire) != ref.to_canonical_dict():
        raise ValueError("reference wire must be canonical")
    return ref


def _entry_wire(ref: LogEntryRef) -> dict[str, object]:
    if (
        type(ref) is not LogEntryRef
        or type(ref.log_sequence) is not int
        or ref.log_sequence <= 0
    ):
        raise ValueError("entry reference must have an integer positive sequence")
    return {
        "log_name": _text(ref.log_name),
        "log_sequence": ref.log_sequence,
        "receipt_hash": _hash(ref.receipt_hash),
    }


def _checkpoint_wire(checkpoint: LogCheckpoint) -> dict[str, object]:
    if (
        type(checkpoint) is not LogCheckpoint
        or type(checkpoint.upper_log_sequence) is not int
        or checkpoint.upper_log_sequence < 0
    ):
        raise ValueError("checkpoint must have an integer nonnegative sequence")
    head = checkpoint.head_receipt_hash
    if checkpoint.upper_log_sequence == 0:
        if head is not None:
            raise ValueError("empty checkpoint must not have a head")
    else:
        _hash(head)
    return {
        "log_name": _text(checkpoint.log_name),
        "as_of": _instant(checkpoint.as_of),
        "upper_log_sequence": checkpoint.upper_log_sequence,
        "head_receipt_hash": head,
    }


def _publication(value: object, log_name: str) -> Publication:
    if type(value) is not tuple or len(value) != 3:
        raise ValueError("publication must be an exact immutable three-tuple")
    ref, entry, checkpoint = value
    _ref(ref)
    _entry_wire(entry)
    _checkpoint_wire(checkpoint)
    if entry.log_name != log_name or checkpoint.log_name != log_name:
        raise ValueError("publication must use the designated owner log")
    return ref, entry, checkpoint


def _publication_wire(publication: Publication) -> dict[str, object]:
    ref, entry, checkpoint = publication
    return {
        "artifact_ref": ref.to_canonical_dict(),
        "entry_ref": _entry_wire(entry),
        "checkpoint": _checkpoint_wire(checkpoint),
    }


def _publication_from_wire(value: object, log_name: str) -> Publication:
    wire = _mapping(value, {"artifact_ref", "entry_ref", "checkpoint"})
    entry = _mapping(wire["entry_ref"], {"log_name", "log_sequence", "receipt_hash"})
    checkpoint = _mapping(
        wire["checkpoint"],
        {"log_name", "as_of", "upper_log_sequence", "head_receipt_hash"},
    )
    return _publication(
        (
            _ref_from_wire(wire["artifact_ref"]),
            LogEntryRef(**entry),
            LogCheckpoint(**checkpoint),
        ),
        log_name,
    )


def _original_publication(
    foundation: LocalFoundation,
    publication: Publication,
    source: bytes,
    event_id: str,
) -> LogEntry:
    _, entry_ref, checkpoint = publication
    # Receipt-derived checkpoint may be unissued: verify via exact EntryRef.
    entry = foundation.entries(entry_ref.log_name, entry_ref)[-1]
    expected = LogCheckpoint(
        entry_ref.log_name, entry.accepted_at, entry.log_sequence, entry.receipt_hash
    )
    if entry.event_id != event_id or entry.payload != source or checkpoint != expected:
        raise ValueError(
            "publication must match its original event/bytes/receipt cutoff"
        )
    return entry


def _artifact_event(ref: ArtifactRef, log_name: str) -> str:
    return canonical_sha256(("artifact-publication-v1", log_name, ref))


def _task_event(study_ref: ArtifactRef, phase: str) -> str:
    return canonical_sha256(("statistical-read-task-publication-v1", study_ref, phase))


def _binding_event(study_ref: ArtifactRef) -> str:
    return canonical_sha256(("statistical-heldout-binding-v1", study_ref))


def _current_entries(
    foundation: LocalFoundation, log_name: str
) -> tuple[LogCheckpoint, tuple[LogEntry, ...]]:
    checkpoint = foundation.checkpoint(log_name)
    return checkpoint, foundation.entries(log_name, checkpoint)


def _slice_wire(value: object) -> Mapping[str, Any]:
    wire = _mapping(value, _DATA_KEYS)
    source = wire["market_bundle_ref"]
    if (
        isinstance(source, Mapping)
        and source.get("type") == "backtest_market_bundle_ref"
    ):
        source = _mapping(source, {"type", "artifact_ref"})["artifact_ref"]
    ref = _ref_from_wire(source)
    if ref.artifact_type != "backtest_market_bundle" or ref.schema_version != 1:
        raise ValueError("source must bind a backtest_market_bundle@1")
    _text(wire["dataset_revision"])
    if _instant(wire["interval_start"]) >= _instant(wire["interval_end"]):
        raise ValueError("read interval must be nonempty")
    return wire


def _study(envelope: ArtifactEnvelope) -> Mapping[str, Any]:
    if (
        envelope.artifact_type != _STUDY_TYPE
        or type(envelope.schema_version) is not int
        or envelope.schema_version != 1
    ):
        raise ValueError("Study must be a schema-1 statistical declaration")
    wire = _mapping(
        envelope.payload, {"study_key", "models", "partitions", *_STUDY_HASH_FIELDS}
    )
    _text(wire["study_key"])
    for name in _STUDY_HASH_FIELDS:
        _hash(wire[name])
    if type(wire["models"]) is not tuple or wire["models"] != _MODELS:
        raise ValueError("Study must use the four fixed ordered models")
    if type(wire["partitions"]) is not tuple or len(wire["partitions"]) != 3:
        raise ValueError("Study must cover the three ordered partitions")
    previous_end = None
    for name, item in zip(
        ("TRAIN", "DEVELOPMENT", "HELD_OUT"), wire["partitions"], strict=True
    ):
        part = _mapping(
            item,
            {
                "partition",
                "data_slice",
                "decision_interval_start",
                "decision_interval_end",
            },
        )
        data = _slice_wire(part["data_slice"])
        start, end = (
            _instant(part["decision_interval_start"]),
            _instant(part["decision_interval_end"]),
        )
        if (
            part["partition"] != name
            or not data["interval_start"] <= start < end <= data["interval_end"]
        ):
            raise ValueError(
                "decision interval must be contained in its ordered partition"
            )
        if previous_end is not None and previous_end > data["interval_start"]:
            raise ValueError("partition read intervals must be ordered and disjoint")
        previous_end = data["interval_end"]
    return wire


def _task(envelope: ArtifactEnvelope) -> Mapping[str, Any]:
    wire = _mapping(
        envelope.payload, {"study_ref", "phase", "data_slice", *_HASH_FIELDS}
    )
    study_ref = _ref_from_wire(wire["study_ref"])
    if study_ref.artifact_type != _STUDY_TYPE or study_ref.schema_version != 1:
        raise ValueError("Task must name a statistical Study@1")
    phase = wire["phase"]
    if type(phase) is not str or phase not in _PHASES:
        raise ValueError("Task must use one fixed read phase")
    if (
        envelope.artifact_type != _PHASES[phase][1]
        or type(envelope.schema_version) is not int
        or envelope.schema_version != 1
    ):
        raise ValueError("Task producer type/version must agree with phase")
    _slice_wire(wire["data_slice"])
    for name in _HASH_FIELDS:
        if wire[name] is not None:
            _hash(wire[name])
    return wire


def _scope(
    study_ref: ArtifactRef, study: Mapping[str, Any], task: Mapping[str, Any]
) -> None:
    partition = _PHASES[task["phase"]][0]
    expected = next(
        item["data_slice"]
        for item in study["partitions"]
        if item["partition"] == partition
    )
    if (
        task["study_ref"] != study_ref.to_canonical_dict()
        or task["data_slice"] != expected
    ):
        raise ValueError("Task must exactly cover its Study partition/source/revision")


def _freeze_fields(task: Mapping[str, Any]) -> None:
    required = _PHASES[task["phase"]][3]
    for name in _HASH_FIELDS:
        if (name in required and task[name] is None) or (
            name not in required and task[name] is not None
        ):
            raise ValueError(
                "Task hashes must exactly match its frozen phase prerequisites"
            )


def _envelope(source: bytes) -> ArtifactEnvelope:
    try:
        decoded = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("owner entry must contain JSON Envelope bytes") from error
    wire = _mapping(
        decoded, {"artifact_type", "schema_version", "payload", "content_hash"}
    )
    envelope = ArtifactEnvelope(**wire)
    if canonical_bytes(envelope) != source:
        raise ValueError("owner entry must contain canonical Envelope bytes")
    return envelope


def _phases(
    foundation: LocalFoundation,
    study_ref: ArtifactRef,
    study: Mapping[str, Any],
    task_ref: ArtifactRef,
    task_entry: LogEntry,
    study_entry: LogEntry,
) -> dict[str, _PhaseEntry]:
    _, entries = _current_entries(foundation, _RESEARCH_LOG)
    phases = {}
    for entry in entries:
        envelope = _envelope(entry.payload)
        if envelope.artifact_type not in {item[1] for item in _PHASES.values()}:
            continue
        if (
            not isinstance(envelope.payload, Mapping)
            or envelope.payload.get("study_ref") != study_ref.to_canonical_dict()
        ):
            continue
        task = _task(envelope)
        _scope(study_ref, study, task)
        _freeze_fields(task)
        ref = ArtifactRef.from_envelope(envelope)
        if (
            entry.event_id != _task_event(study_ref, task["phase"])
            or foundation.read(ref=ref).source_bytes != entry.payload
        ):
            raise ValueError(
                "phase must have its canonical original publication and CAS bytes"
            )
        if task["phase"] in phases:
            raise ValueError("phase cannot have two publications")
        phases[task["phase"]] = task, ref, entry
    for task, _, entry in phases.values():
        if entry.ledger_sequence <= study_entry.ledger_sequence:
            raise ValueError("published phases must follow Study publication order")
        for parent in _PHASES[task["phase"]][4]:
            if (
                parent not in phases
                or phases[parent][2].ledger_sequence >= entry.ledger_sequence
            ):
                raise ValueError("published phases must respect prerequisite order")
        for name in ("model_cards_hash", "predictions_hash"):
            if task[name] is not None and any(
                prior[0][name] not in (None, task[name]) for prior in phases.values()
            ):
                raise ValueError(
                    "Study must keep one shared frozen model/prediction anchor"
                )
    if not any(
        ref == task_ref and entry == task_entry for _, ref, entry in phases.values()
    ):
        raise ValueError(
            "current prefix must include the exact requested task publication"
        )
    return phases


def _record_wire(record: SampleConsumptionRecord) -> dict[str, str]:
    return {name: getattr(record, name) for name in _RECORD_KEYS}


def _record(value: object) -> SampleConsumptionRecord:
    wire = _mapping(value, _RECORD_KEYS)
    record = SampleConsumptionRecord(**wire)
    if _record_wire(record) != wire or build_snapshot(
        (record,), as_of=record.consumed_at
    ).records != (record,):
        raise ValueError(
            "record must use canonical public SampleConsumptionRecord values"
        )
    return record


def _intended_record(
    ref: ArtifactRef, task: Mapping[str, Any], consumed_at: str
) -> SampleConsumptionRecord:
    data = task["data_slice"]
    return SampleConsumptionRecord(
        data["dataset_revision"],
        data["interval_start"],
        data["interval_end"],
        _PHASES[task["phase"]][2],
        canonical_sha256(("sample-consumer-v1", ref)),
        _instant(consumed_at),
    )


def _reservation_event(record: SampleConsumptionRecord, ref: ArtifactRef) -> str:
    return canonical_sha256(
        (
            "sample-consumption-append-v1",
            ref,
            record.dataset_revision,
            record.interval_start,
            record.interval_end,
            record.purpose,
        )
    )


def _sample_entries(entries: tuple[LogEntry, ...]) -> tuple[_SampleEntry, ...]:
    result = []
    for entry in entries:
        envelope = _envelope(entry.payload)
        if (
            envelope.artifact_type != "sample_consumption_append"
            or type(envelope.schema_version) is not int
            or envelope.schema_version != 1
        ):
            raise ValueError("sample entry must be SampleConsumptionAppend@1")
        payload = _mapping(envelope.payload, {"record", "producer_ref"})
        record, ref = (
            _record(payload["record"]),
            _ref_from_wire(payload["producer_ref"]),
        )
        if (
            ref.schema_version != 1
            or _PURPOSES.get(ref.artifact_type) != record.purpose
            or record.consumer_id != canonical_sha256(("sample-consumer-v1", ref))
        ):
            raise ValueError(
                "sample record must match its canonical producer/purpose/consumer"
            )
        if (
            entry.event_id != _reservation_event(record, ref)
            or record.consumed_at > entry.accepted_at
        ):
            raise ValueError(
                "sample record must match its original event and valid governance time"
            )
        result.append((record, ref, entry))
    return tuple(result)


def _evidence(
    foundation: LocalFoundation,
    value: object,
    data: Mapping[str, Any],
) -> tuple[Publication, Publication, LogEntry, LogEntry]:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(
            "heldout evidence must be an immutable snapshot/assessment pair"
        )
    snapshot_pub, assessment_pub = (_publication(item, _ARTIFACT_LOG) for item in value)
    snapshot = foundation.read(ref=snapshot_pub[0])
    assessment = foundation.read(ref=assessment_pub[0])
    if (
        snapshot.envelope.artifact_type != _SNAPSHOT_TYPE
        or snapshot.envelope.schema_version != 1
        or assessment.envelope.artifact_type != _ASSESSMENT_TYPE
        or assessment.envelope.schema_version != 1
    ):
        raise ValueError(
            "heldout evidence must use the existing snapshot and assessment@1"
        )
    snapshot_entry = _original_publication(
        foundation,
        snapshot_pub,
        snapshot.source_bytes,
        _artifact_event(snapshot_pub[0], _ARTIFACT_LOG),
    )
    assessment_entry = _original_publication(
        foundation,
        assessment_pub,
        assessment.source_bytes,
        _artifact_event(assessment_pub[0], _ARTIFACT_LOG),
    )
    checkpoint_wire = _mapping(
        _mapping(snapshot.envelope.payload, {"checkpoint"})["checkpoint"],
        {"log_name", "as_of", "upper_log_sequence", "head_receipt_hash"},
    )
    checkpoint = LogCheckpoint(**checkpoint_wire)
    _checkpoint_wire(checkpoint)
    if checkpoint.log_name != _SAMPLE_LOG:
        raise ValueError("snapshot must bind the sample-consumption log")
    # The stored snapshot checkpoint, unlike a receipt-derived proof, MUST be issued.
    samples = _sample_entries(foundation.entries(_SAMPLE_LOG, checkpoint))
    if (
        any(
            entry.ledger_sequence >= snapshot_entry.ledger_sequence
            for _, _, entry in samples
        )
        or snapshot_entry.ledger_sequence >= assessment_entry.ledger_sequence
    ):
        raise ValueError("snapshot/assessment must follow their original sample prefix")
    records = build_snapshot(
        tuple(record for record, _, _ in samples), as_of=checkpoint.as_of
    )
    integrity = assess_untouched_holdout(
        records,
        dataset_revision=data["dataset_revision"],
        interval_start=data["interval_start"],
        interval_end=data["interval_end"],
    )
    refs_by_record: dict[SampleConsumptionRecord, list[LogEntryRef]] = {}
    for record, _, entry in samples:
        refs_by_record.setdefault(record, []).append(entry.entry_ref)
    conflicts = tuple(
        _entry_wire(refs_by_record[record].pop(0))
        for record in integrity.conflicting_records
    )
    expected = {
        "snapshot_ref": snapshot_pub[0],
        "dataset_revision": data["dataset_revision"],
        "interval_start": data["interval_start"],
        "interval_end": data["interval_end"],
        "untouched": integrity.untouched,
        "conflicting_append_entry_refs": conflicts,
    }
    payload = _mapping(assessment.envelope.payload, set(expected))
    if (
        type(payload["untouched"]) is not bool
        or canonical_bytes(payload) != canonical_bytes(expected)
        or not integrity.untouched
    ):
        raise ValueError(
            "assessment must truthfully bind the original untouched full read window"
        )
    return snapshot_pub, assessment_pub, snapshot_entry, assessment_entry


def _phase_publication(phase: _PhaseEntry) -> Publication:
    _, ref, entry = phase
    return (
        ref,
        entry.entry_ref,
        LogCheckpoint(
            _RESEARCH_LOG, entry.accepted_at, entry.log_sequence, entry.receipt_hash
        ),
    )


def _binding_body(
    study_pub: Publication,
    feature_pub: Publication,
    data: Mapping[str, Any],
    evidence: tuple[Publication, Publication],
    record: SampleConsumptionRecord,
) -> dict[str, object]:
    return {
        "kind": "statistical_heldout_binding",
        "schema_version": 1,
        "study_publication": _publication_wire(study_pub),
        "heldout_feature_publication": _publication_wire(feature_pub),
        "data_slice": data,
        "snapshot_publication": _publication_wire(evidence[0]),
        "assessment_publication": _publication_wire(evidence[1]),
        "feature_record": _record_wire(record),
    }


def _binding(
    foundation: LocalFoundation,
    entry: LogEntry,
    study_pub: Publication,
    feature: _PhaseEntry,
) -> tuple[tuple[Publication, Publication], SampleConsumptionRecord]:
    try:
        decoded = json.loads(entry.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("binding must contain canonical JSON bytes") from error
    body = _mapping(
        decoded,
        {
            "kind",
            "schema_version",
            "study_publication",
            "heldout_feature_publication",
            "data_slice",
            "snapshot_publication",
            "assessment_publication",
            "feature_record",
        },
    )
    if (
        body["kind"] != "statistical_heldout_binding"
        or type(body["schema_version"]) is not int
        or body["schema_version"] != 1
        or canonical_bytes(body) != entry.payload
    ):
        raise ValueError("binding must be a canonical raw Binding@1 event")
    bound_study = _publication_from_wire(body["study_publication"], _RESEARCH_LOG)
    bound_feature = _publication_from_wire(
        body["heldout_feature_publication"], _RESEARCH_LOG
    )
    if (
        bound_study != study_pub
        or bound_feature != _phase_publication(feature)
        or _slice_wire(body["data_slice"]) != feature[0]["data_slice"]
    ):
        raise ValueError(
            "binding must preserve the exact original Study/Feature/source"
        )
    # Recheck the stored original proofs, not just equality of caller-supplied refs.
    for publication in (bound_study, bound_feature):
        source = foundation.read(ref=publication[0])
        event = (
            _artifact_event(publication[0], _RESEARCH_LOG)
            if publication == bound_study
            else _task_event(study_pub[0], "HELD_OUT_FEATURE")
        )
        original = _original_publication(
            foundation, publication, source.source_bytes, event
        )
        if original.ledger_sequence >= entry.ledger_sequence:
            raise ValueError(
                "binding must follow the original Study and Feature publications"
            )
    evidence = (
        _publication_from_wire(body["snapshot_publication"], _ARTIFACT_LOG),
        _publication_from_wire(body["assessment_publication"], _ARTIFACT_LOG),
    )
    _, _, snapshot_entry, assessment_entry = _evidence(
        foundation, evidence, feature[0]["data_slice"]
    )
    if (
        snapshot_entry.ledger_sequence >= entry.ledger_sequence
        or assessment_entry.ledger_sequence >= entry.ledger_sequence
    ):
        raise ValueError("binding cannot refresh evidence after its immutable commit")
    record = _record(body["feature_record"])
    if (
        record != _intended_record(feature[1], feature[0], record.consumed_at)
        or record.consumed_at > entry.accepted_at
        or entry.event_id != _binding_event(study_pub[0])
    ):
        raise ValueError(
            "binding must preserve the first valid Feature record/time and event"
        )
    return evidence, record


def _known_consumption(
    samples: tuple[_SampleEntry, ...],
    data: Mapping[str, Any],
    binding: LogEntry | None,
    feature: _PhaseEntry,
    feature_record: SampleConsumptionRecord | None,
    outcome: _PhaseEntry | None,
) -> tuple[LogEntry | None, LogEntry | None]:
    feature_entry = None
    outcome_entry = None
    for record, ref, entry in samples:
        if (
            record.dataset_revision != data["dataset_revision"]
            or record.interval_start >= data["interval_end"]
            or data["interval_start"] >= record.interval_end
        ):
            continue
        if (
            binding is not None
            and ref == feature[1]
            and record == feature_record
            and entry.ledger_sequence > binding.ledger_sequence
        ):
            feature_entry = entry
        elif (
            binding is not None
            and outcome is not None
            and ref == outcome[1]
            and record == _intended_record(outcome[1], outcome[0], record.consumed_at)
        ):
            outcome_entry = entry
        else:
            raise ValueError("known local overlap is not this exact frozen bound flow")
    if outcome_entry is not None and (
        feature_entry is None
        or outcome_entry.ledger_sequence <= feature_entry.ledger_sequence
    ):
        raise ValueError(
            "Outcome consumption must follow the bound Feature consumption"
        )
    return feature_entry, outcome_entry


def _verify_reservation(
    foundation: LocalFoundation,
    entry_ref: LogEntryRef,
    record: SampleConsumptionRecord,
    ref: ArtifactRef,
) -> LogEntry:
    _entry_wire(entry_ref)
    if entry_ref.log_name != _SAMPLE_LOG:
        raise ValueError(
            "reserve must return a sample entry in the requested Foundation"
        )
    entry = foundation.entries(_SAMPLE_LOG, entry_ref)[-1]
    expected = ArtifactEnvelope.create(
        "sample_consumption_append",
        1,
        {"record": _record_wire(record), "producer_ref": ref},
    )
    if (
        entry.payload != canonical_bytes(expected)
        or entry.event_id != _reservation_event(record, ref)
        or record.consumed_at > entry.accepted_at
    ):
        raise ValueError(
            "reserve must prove its exact original valid record in the requested storage"
        )
    return entry


def _read_declaration(
    foundation: LocalFoundation, value: object
) -> tuple[Publication, ArtifactEnvelope, bytes]:
    try:
        publication = _publication(value, _RESEARCH_LOG)
        source = foundation.read(ref=publication[0])
        return publication, source.envelope, source.source_bytes
    except Exception as error:  # public trust boundary, preserving the actual cause
        raise StatisticalReservationError("STATISTICAL_PUBLICATION_MISSING") from error


def reserve_statistical_read(
    foundation: LocalFoundation,
    ledger: SampleConsumptionLedger,
    study_publication: Publication,
    task_publication: Publication,
    consumed_at: str,
    heldout_evidence: tuple[Publication, Publication] | None = None,
) -> StatisticalReservationReceipt:
    """Register one declared read; a receipt is never permission to open a Reader."""
    if type(foundation) is not LocalFoundation:
        _fail(
            "STATISTICAL_PUBLICATION_MISSING",
            TypeError("foundation must be LocalFoundation"),
        )
    try:
        if type(ledger) is not SampleConsumptionLedger:
            raise TypeError("ledger must be the public SampleConsumptionLedger")
        ledger.require_foundation(foundation)
    except Exception as error:
        raise StatisticalReservationError(
            "STATISTICAL_RESERVATION_FAILED"
        ) from error
    study_pub, study_envelope, study_source = _read_declaration(
        foundation, study_publication
    )
    try:
        study = _study(study_envelope)
    except Exception as error:
        raise StatisticalReservationError("STATISTICAL_STUDY_INVALID") from error
    task_pub, task_envelope, task_source = _read_declaration(
        foundation, task_publication
    )
    try:
        task = _task(task_envelope)
    except Exception as error:
        raise StatisticalReservationError("STATISTICAL_READ_TASK_INVALID") from error
    try:
        study_entry = _original_publication(
            foundation,
            study_pub,
            study_source,
            _artifact_event(study_pub[0], _RESEARCH_LOG),
        )
        task_entry = _original_publication(
            foundation,
            task_pub,
            task_source,
            _task_event(_ref_from_wire(task["study_ref"]), task["phase"]),
        )
    except Exception as error:
        raise StatisticalReservationError("STATISTICAL_PUBLICATION_MISSING") from error
    try:
        _scope(study_pub[0], study, task)
    except Exception as error:
        raise StatisticalReservationError("STATISTICAL_SCOPE_CONFLICT") from error
    try:
        _freeze_fields(task)
        phases = _phases(
            foundation, study_pub[0], study, task_pub[0], task_entry, study_entry
        )
    except Exception as error:
        raise StatisticalReservationError("STATISTICAL_PREREQUISITE_INVALID") from error

    binding_entry = None
    feature_entry = None
    snapshot_ref = None
    assessment_ref = None
    current_sample_checkpoint = None
    validated_evidence: tuple[Publication, Publication] | None = None
    is_heldout = task["phase"] in ("HELD_OUT_FEATURE", "HELD_OUT_OUTCOME")
    if is_heldout:
        try:
            feature = phases["HELD_OUT_FEATURE"]
            evidence = _evidence(foundation, heldout_evidence, task["data_slice"])
            snapshot_pub, assessment_pub, snapshot_entry, assessment_entry = evidence
            validated_evidence = snapshot_pub, assessment_pub
            snapshot_ref, assessment_ref = snapshot_pub[0], assessment_pub[0]
            _, bindings = _current_entries(foundation, _BINDING_LOG)
            matches = tuple(
                entry
                for entry in bindings
                if entry.event_id == _binding_event(study_pub[0])
            )
            if len(matches) > 1:
                raise ValueError("Study can have only one binding")
            binding_entry = matches[0] if matches else None
            original_feature_record = None
            if binding_entry is not None:
                original_evidence, original_feature_record = _binding(
                    foundation, binding_entry, study_pub, feature
                )
                if task["phase"] == "HELD_OUT_OUTCOME" and original_evidence != (
                    snapshot_pub,
                    assessment_pub,
                ):
                    raise ValueError(
                        "Outcome must use the same originally bound evidence"
                    )
                if (
                    snapshot_entry.ledger_sequence >= binding_entry.ledger_sequence
                    or assessment_entry.ledger_sequence >= binding_entry.ledger_sequence
                ):
                    raise ValueError("evidence must precede the binding")
            elif task["phase"] == "HELD_OUT_OUTCOME":
                raise ValueError(
                    "Outcome cannot create or retrospectively replace a binding"
                )
            current_sample_checkpoint, current_samples = _current_entries(
                foundation, _SAMPLE_LOG
            )
            samples = _sample_entries(current_samples)
            feature_entry, _ = _known_consumption(
                samples,
                task["data_slice"],
                binding_entry,
                feature,
                original_feature_record,
                phases.get("HELD_OUT_OUTCOME"),
            )
            if task["phase"] == "HELD_OUT_OUTCOME" and feature_entry is None:
                raise ValueError(
                    "Outcome requires a valid actual Feature append after binding"
                )
        except Exception as error:
            raise StatisticalReservationError(
                "STATISTICAL_HOLDOUT_EVIDENCE_REQUIRED"
            ) from error
    elif heldout_evidence is not None:
        _fail(
            "STATISTICAL_HOLDOUT_EVIDENCE_REQUIRED",
            ValueError("non-heldout phase must not carry heldout evidence"),
        )

    try:
        record = _intended_record(task_pub[0], task, consumed_at)
        if current_sample_checkpoint is None:
            current_sample_checkpoint, _ = _current_entries(foundation, _SAMPLE_LOG)
        if record.consumed_at > current_sample_checkpoint.as_of:
            raise ValueError(
                "consumed_at must not be in the future of an actually issued sample checkpoint"
            )
        if task["phase"] == "HELD_OUT_FEATURE":
            if validated_evidence is None:
                raise ValueError("Feature requires validated original evidence")
            body = _binding_body(
                study_pub, task_pub, task["data_slice"], validated_evidence, record
            )
            receipt = foundation.append(
                _BINDING_LOG, _binding_event(study_pub[0]), canonical_bytes(body)
            )
            binding_entry = foundation.entries(_BINDING_LOG, receipt.entry_ref)[-1]
            if binding_entry.payload != canonical_bytes(body):
                raise ValueError("binding append must prove the exact requested bytes")
            # Establish the immutable commit before invoking the existing ledger.
            _binding(foundation, binding_entry, study_pub, phases["HELD_OUT_FEATURE"])
        # Same event/record replays return the original entry, never refresh time.
        reservation_ref = ledger.reserve(record, task_pub[0])
        reservation_entry = _verify_reservation(
            foundation, reservation_ref, record, task_pub[0]
        )
        if (
            binding_entry is not None
            and reservation_entry.ledger_sequence <= binding_entry.ledger_sequence
        ):
            raise ValueError("heldout reservation must follow binding")
        if (
            feature_entry is not None
            and task["phase"] == "HELD_OUT_OUTCOME"
            and reservation_entry.ledger_sequence <= feature_entry.ledger_sequence
        ):
            raise ValueError("Outcome reservation must follow Feature")
    except Exception as error:
        raise StatisticalReservationError("STATISTICAL_RESERVATION_FAILED") from error
    return StatisticalReservationReceipt(
        study_pub[0],
        task_pub[0],
        study_pub[1],
        task_pub[1],
        record.purpose,
        record,
        reservation_ref,
        snapshot_ref,
        assessment_ref,
        binding_entry.entry_ref if binding_entry is not None else None,
    )
