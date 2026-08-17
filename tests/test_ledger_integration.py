from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from crypto_quant_domain import (
    ArtifactEnvelope,
    ArtifactRef,
    canonical_bytes,
    canonical_sha256,
)
from crypto_quant_foundation import FoundationFailure, LocalFoundation, LogCheckpoint
from crypto_quant_validation import (
    Holdout,
    SampleConsumptionLedger,
    SampleConsumptionRecord,
    ValidationCoreFailure,
)

SAMPLE_LOG = "validation.sample-consumption.v1"
ARTIFACT_LOG = "validation.artifacts.v1"
RESERVED_AT = "2026-02-01T00:00:00.000000Z"
RECEIVED_AT = "2026-02-01T00:00:01.000000Z"
HOLDOUT = Holdout(
    "market-bundle:oos",
    "eth-usdt-v1",
    "2026-03-01T00:00:00.000000Z",
    "2026-04-01T00:00:00.000000Z",
    "HOLDOUT",
    False,
)


def _producer(artifact_type: str, marker: str) -> ArtifactRef:
    return ArtifactRef(artifact_type, 1, "sha256:" + marker * 64)


def _record(
    producer_ref: ArtifactRef,
    purpose: str,
    *,
    start: str = "2026-01-01T00:00:00.000000Z",
    end: str = "2026-02-01T00:00:00.000000Z",
    consumed_at: str = RESERVED_AT,
) -> SampleConsumptionRecord:
    return SampleConsumptionRecord(
        "eth-usdt-v1",
        start,
        end,
        purpose,
        canonical_sha256(("sample-consumer-v1", producer_ref)),
        consumed_at,
    )


def _entry_ref_wire(ref: object) -> dict[str, object]:
    return {
        "log_name": ref.log_name,
        "log_sequence": ref.log_sequence,
        "receipt_hash": ref.receipt_hash,
    }


def _artifact_event_id(ref: ArtifactRef) -> str:
    return canonical_sha256(("artifact-publication-v1", ARTIFACT_LOG, ref))


def _ledger(tmp_path: Path, *times: str) -> tuple[SampleConsumptionLedger, LocalFoundation]:
    values = iter(times)
    foundation = LocalFoundation(tmp_path, clock=lambda: next(values))
    return SampleConsumptionLedger(foundation), foundation


def _failure(code: str, call: Callable[[], object]) -> None:
    with pytest.raises(ValidationCoreFailure) as raised:
        call()
    assert raised.value.code == code


@pytest.mark.parametrize(
    ("artifact_type", "purpose", "marker"),
    (
        ("trial_declaration", "discovery", "a"),
        ("selection_declaration", "selection", "b"),
        ("validation_case", "validation", "c"),
    ),
)
def test_reserve_appends_the_canonical_record_before_a_consumer_can_read(
    tmp_path: Path, artifact_type: str, purpose: str, marker: str
) -> None:
    ledger, foundation = _ledger(tmp_path, RECEIVED_AT, RECEIVED_AT)
    producer = _producer(artifact_type, marker)
    record = _record(producer, purpose)

    entry_ref = ledger.reserve(record, producer)
    entry = foundation.entries(SAMPLE_LOG, entry_ref)[0]
    envelope = json.loads(entry.payload)

    assert envelope["artifact_type"] == "sample_consumption_append"
    assert envelope["schema_version"] == 1
    assert envelope["payload"] == {
        "record": {
            "dataset_revision": record.dataset_revision,
            "interval_start": record.interval_start,
            "interval_end": record.interval_end,
            "purpose": record.purpose,
            "consumer_id": record.consumer_id,
            "consumed_at": record.consumed_at,
        },
        "producer_ref": producer.to_canonical_dict(),
    }
    assert entry.event_id == canonical_sha256(
        (
            "sample-consumption-append-v1",
            producer,
            record.dataset_revision,
            record.interval_start,
            record.interval_end,
            record.purpose,
        )
    )
    assert ledger.reserve(record, producer) == entry_ref
    assert len(foundation.entries(SAMPLE_LOG)) == 1
    assert not any((tmp_path / "artifacts" / "sha256").iterdir())

    changed_time = _record(
        producer,
        purpose,
        consumed_at="2026-02-01T00:00:00.000001Z",
    )
    with pytest.raises(FoundationFailure) as raised:
        ledger.reserve(changed_time, producer)
    assert raised.value.code == "LOG_CONFLICT"


def test_reserve_rejects_mismatched_producer_and_fails_closed_on_late_reservation(
    tmp_path: Path,
) -> None:
    ledger, foundation = _ledger(tmp_path, RECEIVED_AT)
    trial = _producer("trial_declaration", "a")
    selection = _producer("selection_declaration", "b")
    wrong_consumer = SampleConsumptionRecord(
        "eth-usdt-v1",
        "2026-01-01T00:00:00.000000Z",
        "2026-02-01T00:00:00.000000Z",
        "discovery",
        "not-derived",
        RESERVED_AT,
    )

    with pytest.raises(ValueError, match="consumer_id"):
        ledger.reserve(wrong_consumer, trial)
    with pytest.raises(ValueError, match="purpose"):
        ledger.reserve(_record(selection, "selection"), trial)
    assert not (tmp_path / "registries").exists()

    late = _record(
        trial,
        "discovery",
        consumed_at="2026-02-01T00:00:02.000000Z",
    )
    _failure("SAMPLE_LEDGER_CONFLICT", lambda: ledger.reserve(late, trial))
    assert len(foundation.entries(SAMPLE_LOG)) == 1


def test_snapshot_reconstruction_excludes_a_later_equal_time_reservation(
    tmp_path: Path,
) -> None:
    ledger, foundation = _ledger(
        tmp_path,
        "2026-02-01T00:00:01.000000Z",
        "2026-02-01T00:00:02.000000Z",
        "2026-02-01T00:00:02.000000Z",
        "2026-02-01T00:00:02.000000Z",
        "2026-02-01T00:00:03.000000Z",
    )
    trial = _producer("trial_declaration", "a")
    case = _producer("validation_case", "b")
    ledger.reserve(_record(trial, "discovery"), trial)
    snapshot_ref = ledger.freeze_snapshot()
    later_ref = ledger.reserve(
        _record(
            case,
            "validation",
            start=HOLDOUT.interval_start,
            end=HOLDOUT.interval_end,
            consumed_at="2026-02-01T00:00:01.000000Z",
        ),
        case,
    )

    assessment_ref = ledger.assess_holdout(snapshot_ref, HOLDOUT)
    snapshot = foundation.read(ref=snapshot_ref).artifact
    assessment = foundation.read(ref=assessment_ref).artifact
    snapshot_entry = foundation.entries(ARTIFACT_LOG)[0]

    assert json.loads(snapshot_entry.payload)["artifact_type"] == "sample_consumption_ledger_snapshot"
    assert snapshot_entry.event_id == _artifact_event_id(snapshot_ref)
    assert snapshot["checkpoint"]["upper_log_sequence"] == 1
    assert foundation.entries(SAMPLE_LOG)[-1].entry_ref == later_ref
    assert assessment["untouched"] is True
    assert assessment["conflicting_append_entry_refs"] == ()


def test_assessment_publishes_canonical_conflicting_append_entry_refs(
    tmp_path: Path,
) -> None:
    ledger, foundation = _ledger(
        tmp_path,
        "2026-02-01T00:00:01.000000Z",
        "2026-02-01T00:00:02.000000Z",
        "2026-02-01T00:00:03.000000Z",
        "2026-02-01T00:00:04.000000Z",
    )
    case = _producer("validation_case", "c")
    conflict_ref = ledger.reserve(
        _record(
            case,
            "validation",
            start=HOLDOUT.interval_start,
            end=HOLDOUT.interval_end,
        ),
        case,
    )
    snapshot_ref = ledger.freeze_snapshot()

    assessment_ref = ledger.assess_holdout(snapshot_ref, HOLDOUT)
    assessment = foundation.read(ref=assessment_ref).artifact
    entry = foundation.entries(ARTIFACT_LOG)[-1]

    assert assessment["snapshot_ref"] == snapshot_ref.to_canonical_dict()
    assert assessment["untouched"] is False
    assert assessment["conflicting_append_entry_refs"] == (_entry_ref_wire(conflict_ref),)
    assert json.loads(entry.payload)["artifact_type"] == "sample_integrity_assessment"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda path: path.write_bytes(path.read_bytes()[:-1]),
        lambda path: path.write_bytes(b"{\n"),
    ),
    ids=("truncated", "tampered"),
)
def test_corrupt_sample_prefix_does_not_publish_an_assessment(
    tmp_path: Path, mutate: Callable[[Path], None]
) -> None:
    ledger, _ = _ledger(
        tmp_path,
        "2026-02-01T00:00:01.000000Z",
        "2026-02-01T00:00:02.000000Z",
        "2026-02-01T00:00:03.000000Z",
    )
    trial = _producer("trial_declaration", "a")
    ledger.reserve(_record(trial, "discovery"), trial)
    snapshot_ref = ledger.freeze_snapshot()
    mutate(tmp_path / "registries" / f"{SAMPLE_LOG}.jsonl")

    with pytest.raises(FoundationFailure) as raised:
        ledger.assess_holdout(snapshot_ref, HOLDOUT)
    assert raised.value.code == "LOG_INTEGRITY"
    assert (tmp_path / "registries" / f"{ARTIFACT_LOG}.jsonl").read_text().count("\n") == 1


def test_wrong_log_snapshot_fails_without_an_assessment(tmp_path: Path) -> None:
    ledger, foundation = _ledger(tmp_path, "2026-02-01T00:00:01.000000Z")
    envelope = ArtifactEnvelope.create(
        "sample_consumption_ledger_snapshot",
        1,
        {
            "checkpoint": {
                "log_name": "other.v1",
                "as_of": "2026-02-01T00:00:00.000000Z",
                "upper_log_sequence": 0,
                "head_receipt_hash": None,
            }
        },
    )
    snapshot_ref = foundation.put(envelope=envelope)
    foundation.append(ARTIFACT_LOG, _artifact_event_id(snapshot_ref), canonical_bytes(envelope))

    _failure("SAMPLE_LEDGER_CONFLICT", lambda: ledger.assess_holdout(snapshot_ref, HOLDOUT))
    assert len(foundation.entries(ARTIFACT_LOG)) == 1


def test_forged_issued_checkpoint_cannot_hide_a_holdout_reservation(
    tmp_path: Path,
) -> None:
    ledger, foundation = _ledger(tmp_path, *([RECEIVED_AT] * 4))
    case = _producer("validation_case", "c")
    ledger.reserve(
        _record(
            case,
            "validation",
            start=HOLDOUT.interval_start,
            end=HOLDOUT.interval_end,
        ),
        case,
    )
    forged = LogCheckpoint(SAMPLE_LOG, RECEIVED_AT, 0, None)
    envelope = ArtifactEnvelope.create(
        "sample_consumption_ledger_snapshot",
        1,
        {
            "checkpoint": {
                "log_name": forged.log_name,
                "as_of": forged.as_of,
                "upper_log_sequence": forged.upper_log_sequence,
                "head_receipt_hash": forged.head_receipt_hash,
            }
        },
    )
    snapshot_ref = foundation.put(envelope=envelope)
    foundation.append(ARTIFACT_LOG, _artifact_event_id(snapshot_ref), canonical_bytes(envelope))

    with pytest.raises(FoundationFailure) as raised:
        ledger.assess_holdout(snapshot_ref, HOLDOUT)
    assert raised.value.code == "LOG_INTEGRITY"
    assert len(foundation.entries(ARTIFACT_LOG)) == 1
