from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import NoReturn

from crypto_quant_domain import (
    ArtifactEnvelope,
    ArtifactRef,
    canonical_bytes,
    canonical_sha256,
)
from crypto_quant_foundation import LocalFoundation, LogCheckpoint, LogEntryRef

from .integration import Holdout, ValidationCoreFailure
from .sample_consumption import (
    SampleConsumptionRecord,
    assess_untouched_holdout,
    build_snapshot,
)

_SAMPLE_LOG = "validation.sample-consumption.v1"
_ARTIFACT_LOG = "validation.artifacts.v1"
_APPEND_TYPE = "sample_consumption_append"
_SNAPSHOT_TYPE = "sample_consumption_ledger_snapshot"
_ASSESSMENT_TYPE = "sample_integrity_assessment"
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PURPOSE_BY_PRODUCER = {
    "feature_build_task": "feature_build",
    "model_training_task": "model_training",
    "trial_declaration": "discovery",
    "selection_declaration": "selection",
    "validation_case": "validation",
}


def _conflict(error: Exception | None = None) -> NoReturn:
    raise ValidationCoreFailure("SAMPLE_LEDGER_CONFLICT") from error


def _record(value: object) -> SampleConsumptionRecord:
    if type(value) is not SampleConsumptionRecord:
        raise TypeError("record must be a SampleConsumptionRecord")
    try:
        snapshot = build_snapshot((value,), as_of=value.consumed_at)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("record must be canonical") from error
    if snapshot.records != (value,):
        raise ValueError("record must be canonical")
    return snapshot.records[0]


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


def _holdout(value: object) -> Holdout:
    if type(value) is not Holdout:
        raise TypeError("holdout must be a Holdout")
    try:
        normalized = Holdout(
            value.market_bundle_ref,
            value.dataset_revision,
            value.interval_start,
            value.interval_end,
            value.role,
            value.selection_observed,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("holdout must be canonical") from error
    if normalized != value:
        raise ValueError("holdout must be canonical")
    return normalized


def _consumer_id(producer_ref: ArtifactRef) -> str:
    return canonical_sha256(("sample-consumer-v1", producer_ref))


def _reservation_event_id(
    record: SampleConsumptionRecord, producer_ref: ArtifactRef
) -> str:
    return canonical_sha256(
        (
            "sample-consumption-append-v1",
            producer_ref,
            record.dataset_revision,
            record.interval_start,
            record.interval_end,
            record.purpose,
        )
    )


def _artifact_event_id(ref: ArtifactRef) -> str:
    return canonical_sha256(("artifact-publication-v1", _ARTIFACT_LOG, ref))


def _record_wire(record: SampleConsumptionRecord) -> dict[str, str]:
    return {
        "dataset_revision": record.dataset_revision,
        "interval_start": record.interval_start,
        "interval_end": record.interval_end,
        "purpose": record.purpose,
        "consumer_id": record.consumer_id,
        "consumed_at": record.consumed_at,
    }


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


def _validate_pair(
    record: SampleConsumptionRecord, producer_ref: ArtifactRef
) -> None:
    expected_purpose = _PURPOSE_BY_PRODUCER.get(producer_ref.artifact_type)
    if producer_ref.schema_version != 1 or expected_purpose != record.purpose:
        raise ValueError("record purpose does not match producer_ref")
    if record.consumer_id != _consumer_id(producer_ref):
        raise ValueError("record consumer_id does not match producer_ref")


def _checkpoint(value: object) -> LogCheckpoint:
    if not isinstance(value, Mapping) or set(value) != {
        "log_name",
        "as_of",
        "upper_log_sequence",
        "head_receipt_hash",
    }:
        _conflict()
    try:
        checkpoint = LogCheckpoint(
            value["log_name"],
            value["as_of"],
            value["upper_log_sequence"],
            value["head_receipt_hash"],
        )
        build_snapshot((), as_of=checkpoint.as_of)
    except (TypeError, ValueError) as error:
        _conflict(error)
    if (
        checkpoint.log_name != _SAMPLE_LOG
        or type(checkpoint.upper_log_sequence) is not int
        or checkpoint.upper_log_sequence < 0
        or (
            checkpoint.upper_log_sequence == 0
            and checkpoint.head_receipt_hash is not None
        )
        or (
            checkpoint.upper_log_sequence > 0
            and (
                type(checkpoint.head_receipt_hash) is not str
                or _HASH.fullmatch(checkpoint.head_receipt_hash) is None
            )
        )
    ):
        _conflict()
    return checkpoint


def _snapshot_checkpoint(payload: object) -> LogCheckpoint:
    if not isinstance(payload, Mapping) or set(payload) != {"checkpoint"}:
        _conflict()
    return _checkpoint(payload["checkpoint"])


def _producer_from_wire(value: object) -> ArtifactRef:
    if not isinstance(value, Mapping) or set(value) != {
        "type",
        "artifact_type",
        "schema_version",
        "content_hash",
    }:
        _conflict()
    try:
        producer_ref = ArtifactRef(
            value["artifact_type"],
            value["schema_version"],
            value["content_hash"],
        )
    except (TypeError, ValueError) as error:
        _conflict(error)
    if value["type"] != "artifact_ref" or dict(value) != producer_ref.to_canonical_dict():
        _conflict()
    return producer_ref


def _record_from_wire(value: object) -> SampleConsumptionRecord:
    if not isinstance(value, Mapping) or set(value) != {
        "dataset_revision",
        "interval_start",
        "interval_end",
        "purpose",
        "consumer_id",
        "consumed_at",
    }:
        _conflict()
    try:
        record = SampleConsumptionRecord(
            value["dataset_revision"],
            value["interval_start"],
            value["interval_end"],
            value["purpose"],
            value["consumer_id"],
            value["consumed_at"],
        )
    except (TypeError, ValueError) as error:
        _conflict(error)
    if dict(value) != _record_wire(record):
        _conflict()
    return record


def _append_record(payload: bytes, event_id: str, accepted_at: str) -> SampleConsumptionRecord:
    try:
        decoded = json.loads(payload.decode("utf-8"))
        if type(decoded) is not dict or set(decoded) != {
            "artifact_type",
            "schema_version",
            "payload",
            "content_hash",
        }:
            raise ValueError("append payload is not an Envelope")
        envelope = ArtifactEnvelope(
            decoded["artifact_type"],
            decoded["schema_version"],
            decoded["payload"],
            decoded["content_hash"],
        )
        if canonical_bytes(envelope) != payload:
            raise ValueError("append payload is not canonical")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        _conflict(error)
    if envelope.artifact_type != _APPEND_TYPE or envelope.schema_version != 1:
        _conflict()
    if not isinstance(envelope.payload, Mapping) or set(envelope.payload) != {
        "record",
        "producer_ref",
    }:
        _conflict()
    record = _record_from_wire(envelope.payload["record"])
    producer_ref = _producer_from_wire(envelope.payload["producer_ref"])
    try:
        _validate_pair(record, producer_ref)
    except ValueError as error:
        _conflict(error)
    if event_id != _reservation_event_id(record, producer_ref) or record.consumed_at > accepted_at:
        _conflict()
    return record


class SampleConsumptionLedger:
    def __init__(self, foundation: LocalFoundation) -> None:
        if type(foundation) is not LocalFoundation:
            raise TypeError("foundation must be a LocalFoundation")
        self._foundation = foundation

    def reserve(
        self, record: SampleConsumptionRecord, producer_ref: ArtifactRef
    ) -> LogEntryRef:
        record = _record(record)
        producer_ref = _artifact_ref(producer_ref, "producer_ref")
        _validate_pair(record, producer_ref)
        envelope = ArtifactEnvelope.create(
            _APPEND_TYPE,
            1,
            {"record": _record_wire(record), "producer_ref": producer_ref},
        )
        receipt = self._foundation.append(
            _SAMPLE_LOG,
            _reservation_event_id(record, producer_ref),
            canonical_bytes(envelope),
        )
        if record.consumed_at > receipt.accepted_at:
            _conflict()
        return receipt.entry_ref

    def freeze_snapshot(self) -> ArtifactRef:
        checkpoint = self._foundation.checkpoint(_SAMPLE_LOG)
        return self._publish(_SNAPSHOT_TYPE, {"checkpoint": _checkpoint_wire(checkpoint)})

    def assess_holdout(self, snapshot_ref: ArtifactRef, holdout: Holdout) -> ArtifactRef:
        holdout = _holdout(holdout)
        try:
            snapshot_ref = _artifact_ref(snapshot_ref, "snapshot_ref")
        except (TypeError, ValueError) as error:
            _conflict(error)
        if (
            snapshot_ref.artifact_type != _SNAPSHOT_TYPE
            or snapshot_ref.schema_version != 1
        ):
            _conflict()
        snapshot = self._foundation.read(ref=snapshot_ref)
        if (
            snapshot.envelope.artifact_type != _SNAPSHOT_TYPE
            or snapshot.envelope.schema_version != 1
        ):
            _conflict()
        checkpoint = _snapshot_checkpoint(snapshot.envelope.payload)
        self._require_publication(snapshot_ref, snapshot.source_bytes)
        records_and_refs = tuple(
            (
                _append_record(entry.payload, entry.event_id, entry.accepted_at),
                entry.entry_ref,
            )
            for entry in self._foundation.entries(_SAMPLE_LOG, checkpoint)
        )
        records = build_snapshot(
            tuple(record for record, _ in records_and_refs), as_of=checkpoint.as_of
        )
        integrity = assess_untouched_holdout(
            records,
            dataset_revision=holdout.dataset_revision,
            interval_start=holdout.interval_start,
            interval_end=holdout.interval_end,
        )
        refs_by_record: dict[SampleConsumptionRecord, list[LogEntryRef]] = {}
        for record, entry_ref in records_and_refs:
            refs_by_record.setdefault(record, []).append(entry_ref)
        conflicting_refs = tuple(
            refs_by_record[record].pop(0) for record in integrity.conflicting_records
        )
        return self._publish(
            _ASSESSMENT_TYPE,
            {
                "snapshot_ref": snapshot_ref,
                "dataset_revision": holdout.dataset_revision,
                "interval_start": holdout.interval_start,
                "interval_end": holdout.interval_end,
                "untouched": integrity.untouched,
                "conflicting_append_entry_refs": tuple(
                    _entry_ref_wire(ref) for ref in conflicting_refs
                ),
            },
        )

    def _publish(self, artifact_type: str, payload: dict[str, object]) -> ArtifactRef:
        envelope = ArtifactEnvelope.create(artifact_type, 1, payload)
        ref = self._foundation.put(envelope=envelope)
        self._foundation.append(
            _ARTIFACT_LOG,
            _artifact_event_id(ref),
            canonical_bytes(envelope),
        )
        return ref

    def _require_publication(self, ref: ArtifactRef, source: bytes) -> None:
        event_id = _artifact_event_id(ref)
        if not any(
            entry.event_id == event_id and entry.payload == source
            for entry in self._foundation.entries(_ARTIFACT_LOG)
        ):
            _conflict()
