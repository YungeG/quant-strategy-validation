"""Synthetic public-path governance tests, not source/Reader qualification."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime, timedelta
from functools import partial
from itertools import count
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
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
from crypto_quant_research import (
    DataSlice,
    StatisticalReadTask,
    StatisticalStudyDeclaration,
    publish_statistical_read_task,
    publish_statistical_study,
)
from crypto_quant_validation import (
    Holdout,
    SampleConsumptionLedger,
    SampleConsumptionRecord,
    StatisticalReservationError,
    StatisticalReservationReceipt,
    ValidationCoreFailure,
    reserve_statistical_read,
)

RESEARCH_LOG = "research.artifacts.v1"
ARTIFACT_LOG = "validation.artifacts.v1"
SAMPLE_LOG = "validation.sample-consumption.v1"
BINDING_LOG = "validation.statistical-read-bindings.v1"
AT = "2026-01-31T00:00:00.000000Z"
LATER_AT = "2026-01-31T00:00:00.000001Z"
CLOCK = "2026-02-01T00:00:00.000000Z"
MODEL_HASH = "sha256:" + "7" * 64
PREDICTION_HASH = "sha256:" + "8" * 64
SOURCE = ArtifactRef("backtest_market_bundle", 1, "sha256:" + "a" * 64)
PHASES = (
    "TRAIN_FEATURE",
    "TRAIN_FIT",
    "DEVELOPMENT_FEATURE",
    "HELD_OUT_FEATURE",
    "DEVELOPMENT_OUTCOME",
    "HELD_OUT_OUTCOME",
)
Publication = tuple[ArtifactRef, LogEntryRef, LogCheckpoint]
Evidence = tuple[Publication, Publication]


@dataclass
class Case:
    foundation: LocalFoundation
    ledger: SampleConsumptionLedger
    study: StatisticalStudyDeclaration
    study_pub: Publication
    tasks: dict[str, Publication]


def _case(tmp_path: Path, *, full: bool = True, constant_clock: bool = False) -> Case:
    ticks = count()
    instant = datetime(2026, 2, 1, tzinfo=UTC)
    foundation = LocalFoundation(
        tmp_path,
        clock=lambda: (
            CLOCK
            if constant_clock
            else (instant + timedelta(microseconds=next(ticks)))
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        ),
    )
    partitions = tuple(
        (
            name,
            DataSlice(
                SOURCE.to_canonical_dict(),
                "synthetic-registration-only-v1",
                f"2000-{month:02d}-01T00:00:00.000000Z",
                f"2000-{month + 1:02d}-01T00:00:00.000000Z",
            ),
            f"2000-{month:02d}-10T00:00:00.000000Z",
            f"2000-{month:02d}-20T00:00:00.000000Z",
        )
        for name, month in (("TRAIN", 1), ("DEVELOPMENT", 3), ("HELD_OUT", 5))
    )
    study = StatisticalStudyDeclaration(
        "synthetic-statistical-only",
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
        "sha256:" + "3" * 64,
        "sha256:" + "4" * 64,
        "sha256:" + "5" * 64,
        ("CLIMATOLOGY", "PRICE_ONLY", "PRICE_VOLUME", "SHUFFLED_VOLUME"),
        partitions,
    )
    result = Case(
        foundation,
        SampleConsumptionLedger(foundation),
        study,
        publish_statistical_study(study, foundation).publication,
        {},
    )
    if full:
        for phase in PHASES:
            result.tasks[phase] = publish_statistical_read_task(
                _task(result, phase), foundation
            ).publication
    return result


def _task(case: Case, phase: str) -> StatisticalReadTask:
    partition = (
        0 if phase.startswith("TRAIN") else 1 if phase.startswith("DEVELOPMENT") else 2
    )
    return StatisticalReadTask(
        case.study_pub[0],
        phase,
        case.study.partitions[partition][1],
        "sha256:" + "6" * 64 if phase == "TRAIN_FIT" else None,
        MODEL_HASH if partition != 0 else None,
        PREDICTION_HASH if phase.endswith("OUTCOME") else None,
    )


def _wire(value: object) -> Any:
    return json.loads(canonical_bytes(value))


def _entries(case: Case, log_name: str):
    return case.foundation.entries(log_name, case.foundation.checkpoint(log_name))


def _proof(case: Case, ref: ArtifactRef, log_name: str = ARTIFACT_LOG) -> Publication:
    event = canonical_sha256(("artifact-publication-v1", log_name, ref))
    entry = next(entry for entry in _entries(case, log_name) if entry.event_id == event)
    return (
        ref,
        entry.entry_ref,
        LogCheckpoint(
            log_name, entry.accepted_at, entry.log_sequence, entry.receipt_hash
        ),
    )


def _holdout(case: Case) -> Holdout:
    data = case.study.partitions[2][1]
    return Holdout(
        _wire(data.market_bundle_ref),
        data.dataset_revision,
        data.interval_start,
        data.interval_end,
        "HOLDOUT",
        False,
    )


def _evidence(case: Case) -> Evidence:
    snapshot = case.ledger.freeze_snapshot()
    assessment = case.ledger.assess_holdout(snapshot, _holdout(case))
    return _proof(case, snapshot), _proof(case, assessment)


def _raw_publication(
    case: Case,
    artifact_type: str,
    payload: Any,
    *,
    version: int = 1,
    log_name: str = RESEARCH_LOG,
    event: str | None = None,
) -> Publication:
    envelope = ArtifactEnvelope.create(artifact_type, version, payload)
    ref = case.foundation.put(envelope=envelope)
    if event is None:
        event = (
            canonical_sha256(
                (
                    "statistical-read-task-publication-v1",
                    payload["study_ref"],
                    payload["phase"],
                )
            )
            if artifact_type.startswith("statistical_")
            and artifact_type != "statistical_study_declaration"
            else canonical_sha256(("artifact-publication-v1", log_name, ref))
        )
    receipt = case.foundation.append(log_name, event, canonical_bytes(envelope))
    return (
        ref,
        receipt.entry_ref,
        LogCheckpoint(
            log_name, receipt.accepted_at, receipt.log_sequence, receipt.receipt_hash
        ),
    )


def _raw_tasks(
    case: Case,
    *,
    order: tuple[str, ...] = PHASES,
    changes: dict[str, dict[str, Any]] | None = None,
) -> None:
    for phase in order:
        task = _task(case, phase)
        payload = _wire(task.payload)
        payload.update((changes or {}).get(phase, {}))
        case.tasks[phase] = _raw_publication(case, task.artifact_type, payload)


def _reserve(
    case: Case, phase: str, evidence: Any = None, consumed_at: Any = AT
) -> StatisticalReservationReceipt:
    return reserve_statistical_read(
        case.foundation,
        case.ledger,
        case.study_pub,
        case.tasks[phase],
        consumed_at,
        evidence,
    )


def _rejected(code: str, call: Callable[[], object]) -> StatisticalReservationError:
    reader = Mock(name="never-open-a-real-reader")
    with pytest.raises(StatisticalReservationError) as raised:
        call()
        reader()
    reader.assert_not_called()
    assert raised.value.code == "STATISTICAL_" + code
    assert raised.value.__cause__ is not None
    return raised.value


def _record(
    case: Case, producer: ArtifactRef, *, purpose: str = "feature_build", **changes: Any
) -> SampleConsumptionRecord:
    data = case.study.partitions[2][1]
    values = {
        "dataset_revision": data.dataset_revision,
        "interval_start": data.interval_start,
        "interval_end": data.interval_end,
        "purpose": purpose,
        "consumer_id": canonical_sha256(("sample-consumer-v1", producer)),
        "consumed_at": AT,
    }
    values.update(changes)
    return SampleConsumptionRecord(**values)


def _sample_event(record: SampleConsumptionRecord, producer: ArtifactRef) -> str:
    return canonical_sha256(
        (
            "sample-consumption-append-v1",
            producer,
            record.dataset_revision,
            record.interval_start,
            record.interval_end,
            record.purpose,
        )
    )


def _binding_event(case: Case) -> str:
    return canonical_sha256(("statistical-heldout-binding-v1", case.study_pub[0]))


def _capture_binding(
    case: Case, evidence: Evidence, monkeypatch: pytest.MonkeyPatch
) -> bytes:
    original = case.foundation.append
    payloads = []

    def capture(log_name: str, event: str, payload: bytes):
        if log_name == BINDING_LOG:
            payloads.append(payload)
            raise RuntimeError("capture before binding commit")
        return original(log_name, event, payload)

    with monkeypatch.context() as patch:
        patch.setattr(case.foundation, "append", capture)
        _rejected(
            "RESERVATION_FAILED", lambda: _reserve(case, "HELD_OUT_FEATURE", evidence)
        )
    assert len(payloads) == 1
    assert _entries(case, BINDING_LOG) == ()
    assert _entries(case, SAMPLE_LOG) == ()
    return payloads[0]


def test_all_six_public_phases_have_exact_receipts_and_binding_before_consumption(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, constant_clock=True)
    evidence = _evidence(case)
    receipts = {}
    for phase in PHASES:
        heldout = phase.startswith("HELD_OUT")
        receipt = _reserve(case, phase, evidence if heldout else None)
        receipts[phase] = receipt
        task = _task(case, phase)
        assert receipt.study_ref == case.study_pub[0]
        assert receipt.read_task_ref == case.tasks[phase][0]
        assert receipt.study_publication_entry_ref == case.study_pub[1]
        assert receipt.producer_publication_entry_ref == case.tasks[phase][1]
        assert receipt.purpose == task.purpose
        assert receipt.record.dataset_revision == task.data_slice.dataset_revision
        assert (receipt.record.interval_start, receipt.record.interval_end) == (
            task.data_slice.interval_start,
            task.data_slice.interval_end,
        )
        assert receipt.record.consumer_id == canonical_sha256(
            ("sample-consumer-v1", receipt.read_task_ref)
        )
        assert receipt.record.consumed_at == AT
        entry = case.foundation.entries(SAMPLE_LOG, receipt.reservation_entry_ref)[-1]
        assert entry.event_id == _sample_event(receipt.record, receipt.read_task_ref)
        assert entry.payload == canonical_bytes(
            ArtifactEnvelope.create(
                "sample_consumption_append",
                1,
                {
                    "record": {
                        field.name: getattr(receipt.record, field.name)
                        for field in fields(receipt.record)
                    },
                    "producer_ref": receipt.read_task_ref,
                },
            )
        )
        assert (receipt.heldout_snapshot_ref, receipt.heldout_assessment_ref) == (
            (evidence[0][0], evidence[1][0]) if heldout else (None, None)
        )
        assert (receipt.heldout_binding_entry_ref is not None) == heldout
        assert _reserve(case, phase, evidence if heldout else None) == receipt
    assert tuple(field.name for field in fields(StatisticalReservationReceipt)) == (
        "study_ref",
        "read_task_ref",
        "study_publication_entry_ref",
        "producer_publication_entry_ref",
        "purpose",
        "record",
        "reservation_entry_ref",
        "heldout_snapshot_ref",
        "heldout_assessment_ref",
        "heldout_binding_entry_ref",
    )
    bindings = _entries(case, BINDING_LOG)
    assert len(bindings) == 1
    binding = bindings[0]
    body = json.loads(binding.payload)
    assert binding.event_id == _binding_event(case)
    assert binding.payload == canonical_bytes(body)
    assert set(body) == {
        "kind",
        "schema_version",
        "study_publication",
        "heldout_feature_publication",
        "data_slice",
        "snapshot_publication",
        "assessment_publication",
        "feature_record",
    }
    assert body["kind"] == "statistical_heldout_binding" and body["schema_version"] == 1
    assert set(body["study_publication"]) == {"artifact_ref", "entry_ref", "checkpoint"}
    assert set(body["study_publication"]["artifact_ref"]) == {
        "type",
        "artifact_type",
        "schema_version",
        "content_hash",
    }
    assert set(body["study_publication"]["entry_ref"]) == {
        "log_name",
        "log_sequence",
        "receipt_hash",
    }
    assert set(body["study_publication"]["checkpoint"]) == {
        "log_name",
        "as_of",
        "upper_log_sequence",
        "head_receipt_hash",
    }
    assert body["data_slice"] == _wire(case.study.partitions[2][1].payload)
    assert body["feature_record"] == {
        field.name: getattr(receipts["HELD_OUT_FEATURE"].record, field.name)
        for field in fields(SampleConsumptionRecord)
    }
    feature = case.foundation.entries(
        SAMPLE_LOG, receipts["HELD_OUT_FEATURE"].reservation_entry_ref
    )[-1]
    outcome = case.foundation.entries(
        SAMPLE_LOG, receipts["HELD_OUT_OUTCOME"].reservation_entry_ref
    )[-1]
    assert binding.accepted_at == feature.accepted_at == outcome.accepted_at == CLOCK
    assert binding.ledger_sequence < feature.ledger_sequence < outcome.ledger_sequence
    assert (
        receipts["HELD_OUT_FEATURE"].heldout_binding_entry_ref
        == receipts["HELD_OUT_OUTCOME"].heldout_binding_entry_ref
        == binding.entry_ref
    )
    assert _reserve(case, "HELD_OUT_FEATURE", evidence) == receipts["HELD_OUT_FEATURE"]
    assert len(_entries(case, SAMPLE_LOG)) == 6
    # Binding is raw log bytes, not a fifth CAS artifact or an assessment refresh.
    assert len(_entries(case, ARTIFACT_LOG)) == 2


@pytest.mark.parametrize(
    "field,value",
    (
        ("models", ["UNKNOWN"]),
        ("study_key", ""),
        ("core_code_hash", "bad"),
        ("calendar_manifest_hash", True),
        ("extra", "forbidden"),
    ),
)
def test_study_shape_precedes_missing_owner_proof_and_invalid_task(
    tmp_path: Path, field: str, value: Any
) -> None:
    case = _case(tmp_path, full=False)
    payload = _wire(case.study.payload)
    payload[field] = value
    bad = case.foundation.put(
        envelope=ArtifactEnvelope.create("statistical_study_declaration", 1, payload)
    )
    study_pub = (
        bad,
        case.study_pub[1],
        case.study_pub[2],
    )  # CAS-only, wrong actual bytes too.
    invalid_task: Any = ()
    _rejected(
        "STUDY_INVALID",
        lambda: reserve_statistical_read(
            case.foundation, case.ledger, study_pub, invalid_task, AT
        ),
    )
    assert _entries(case, SAMPLE_LOG) == ()


@pytest.mark.parametrize("version", (2, 3))
def test_study_and_task_versions_are_not_coerced(tmp_path: Path, version: int) -> None:
    case = _case(tmp_path, full=False)
    bad = _raw_publication(
        case,
        "statistical_study_declaration",
        _wire(case.study.payload),
        version=version,
    )
    invalid_task: Any = ()
    _rejected(
        "STUDY_INVALID",
        lambda: reserve_statistical_read(
            case.foundation, case.ledger, bad, invalid_task, AT
        ),
    )
    task = _task(case, "TRAIN_FEATURE")
    case.tasks[task.phase] = _raw_publication(
        case, task.artifact_type, _wire(task.payload), version=version
    )
    _rejected("READ_TASK_INVALID", lambda: _reserve(case, task.phase))


@pytest.mark.parametrize(
    "path,value",
    (
        (("phase",), "UNKNOWN"),
        (("extra",), True),
        (("model_cards_hash",), "bad"),
        (("study_ref", "type"), "wrong"),
        (("study_ref", "schema_version"), True),
        (("data_slice", "market_bundle_ref"), "UNKNOWN"),
        (("data_slice", "extra"), True),
        (("data_slice", "interval_start"), "2000-01-01"),
        (("data_slice", "interval_end"), "2000-01-01T00:00:00.000000Z"),
        (("data_slice", "dataset_revision"), "  "),
    ),
)
def test_task_shape_precedes_owner_scope_freeze_and_holdout(
    tmp_path: Path, path: tuple[str, ...], value: Any
) -> None:
    case = _case(tmp_path, full=False)
    task = _task(case, "TRAIN_FEATURE")
    payload = _wire(task.payload)
    _set(payload, path, value)
    # An orphan CAS is enough to inspect shape; it is NOT accepted as publication.
    ref = case.foundation.put(
        envelope=ArtifactEnvelope.create(task.artifact_type, 1, payload)
    )
    case.tasks[task.phase] = (ref, case.study_pub[1], case.study_pub[2])
    _rejected("READ_TASK_INVALID", lambda: _reserve(case, task.phase, "bad-evidence"))
    assert _entries(case, SAMPLE_LOG) == ()


def _set(body: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = body
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value


@pytest.mark.parametrize(
    "case_name",
    (
        "missing-ref",
        "cas-only",
        "wrong-entry",
        "wrong-log",
        "wrong-event",
        "wrong-bytes",
        "wrong-head",
        "wrong-time",
        "later-prefix",
        "bool-sequence",
        "bool-upper",
        "list",
        "bare-ref",
    ),
)
def test_original_owner_proof_cannot_be_substituted(
    tmp_path: Path, case_name: str
) -> None:
    case = _case(tmp_path)
    ref, entry, cp = case.tasks["TRAIN_FEATURE"]
    publication: Any = (ref, entry, cp)
    if case_name == "missing-ref":
        publication = (
            ArtifactRef(ref.artifact_type, 1, "sha256:" + "f" * 64),
            entry,
            cp,
        )
    elif case_name == "cas-only":
        payload = _wire(_task(case, "TRAIN_FEATURE").payload)
        payload["extra"] = "orphan"
        # Keep valid shape by making another actual Study's canonical task instead.
        other = publish_statistical_study(
            replace(case.study, study_key="orphan-parent"), case.foundation
        )
        orphan = case.foundation.put(
            envelope=ArtifactEnvelope.create(
                ref.artifact_type,
                1,
                replace(_task(case, "TRAIN_FEATURE"), study_ref=other.ref).payload,
            )
        )
        publication = (orphan, entry, cp)
    elif case_name == "wrong-entry":
        publication = (ref, replace(entry, receipt_hash="sha256:" + "f" * 64), cp)
    elif case_name == "wrong-log":
        publication = (
            ref,
            replace(entry, log_name=ARTIFACT_LOG),
            replace(cp, log_name=ARTIFACT_LOG),
        )
    elif case_name in ("wrong-event", "wrong-bytes"):
        actual = case.foundation.append(
            RESEARCH_LOG,
            "noncanonical-task-event",
            canonical_bytes(ArtifactEnvelope.create("other", 1, {}))
            if case_name == "wrong-bytes"
            else case.foundation.read(ref=ref).source_bytes,
        )
        publication = (
            ref,
            actual.entry_ref,
            LogCheckpoint(
                RESEARCH_LOG,
                actual.accepted_at,
                actual.log_sequence,
                actual.receipt_hash,
            ),
        )
    elif case_name == "wrong-head":
        publication = (ref, entry, replace(cp, head_receipt_hash="sha256:" + "f" * 64))
    elif case_name == "wrong-time":
        publication = (ref, entry, replace(cp, as_of="2026-02-02T00:00:00.000000Z"))
    elif case_name == "later-prefix":
        publication = (ref, entry, case.foundation.checkpoint(RESEARCH_LOG))
    elif case_name == "bool-sequence":
        publication = (ref, replace(entry, log_sequence=True), cp)
    elif case_name == "bool-upper":
        publication = (ref, entry, replace(cp, upper_log_sequence=True))
    elif case_name == "list":
        publication = list(publication)
    elif case_name == "bare-ref":
        publication = ref
    case.tasks["TRAIN_FEATURE"] = publication
    _rejected(
        "PUBLICATION_MISSING", lambda: _reserve(case, "TRAIN_FEATURE", "bad-evidence")
    )
    assert _entries(case, SAMPLE_LOG) == ()
    assert _entries(case, BINDING_LOG) == ()


@pytest.mark.parametrize(
    "field,value",
    (
        ("dataset_revision", "different-revision"),
        ("interval_start", "2000-01-10T00:00:00.000000Z"),
        ("interval_end", "2000-01-20T00:00:00.000000Z"),
        (
            "market_bundle_ref",
            ArtifactRef(
                "backtest_market_bundle", 1, "sha256:" + "b" * 64
            ).to_canonical_dict(),
        ),
    ),
)
def test_scope_precedes_prerequisite_and_evidence(
    tmp_path: Path, field: str, value: Any
) -> None:
    case = _case(tmp_path, full=False)
    payload = _wire(_task(case, "TRAIN_FIT").payload)
    payload["data_slice"][field] = value
    payload["feature_rows_hash"] = None  # Also invalid prerequisite, lower precedence.
    case.tasks["TRAIN_FIT"] = _raw_publication(
        case, "statistical_training_read_task", payload
    )
    _rejected("SCOPE_CONFLICT", lambda: _reserve(case, "TRAIN_FIT", "bad-evidence"))
    assert _entries(case, SAMPLE_LOG) == ()


def test_genuine_other_study_task_is_scope_conflict_not_missing_publication(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, full=False)
    other = publish_statistical_study(
        replace(case.study, study_key="other-real-owner-declaration"), case.foundation
    )
    task = replace(_task(case, "TRAIN_FEATURE"), study_ref=other.ref)
    case.tasks[task.phase] = publish_statistical_read_task(
        task, case.foundation
    ).publication
    _rejected("SCOPE_CONFLICT", lambda: _reserve(case, task.phase))


@pytest.mark.parametrize("phase", PHASES)
def test_required_and_forbidden_hash_presence_is_prerequisite_failure(
    tmp_path: Path, phase: str
) -> None:
    case = _case(tmp_path, full=False)
    task = _task(case, phase)
    payload = _wire(task.payload)
    required = next(
        (
            field
            for field in ("feature_rows_hash", "model_cards_hash", "predictions_hash")
            if payload[field] is not None
        ),
        None,
    )
    payload[required or "predictions_hash"] = None if required else PREDICTION_HASH
    case.tasks[phase] = _raw_publication(case, task.artifact_type, payload)
    _rejected("PREREQUISITE_INVALID", lambda: _reserve(case, phase, "bad-evidence"))


@pytest.mark.parametrize(
    "order,changes",
    (
        (("TRAIN_FIT", "TRAIN_FEATURE"), {}),
        (
            (
                "TRAIN_FEATURE",
                "TRAIN_FIT",
                "DEVELOPMENT_FEATURE",
                "DEVELOPMENT_OUTCOME",
            ),
            {},
        ),
        (PHASES, {"HELD_OUT_FEATURE": {"model_cards_hash": "sha256:" + "f" * 64}}),
        (PHASES, {"HELD_OUT_OUTCOME": {"predictions_hash": "sha256:" + "f" * 64}}),
    ),
)
def test_actual_owner_order_and_shared_frozen_anchors_are_rechecked(
    tmp_path: Path, order: tuple[str, ...], changes: dict[str, dict[str, Any]]
) -> None:
    case = _case(tmp_path, full=False)
    _raw_tasks(case, order=order, changes=changes)
    _rejected("PREREQUISITE_INVALID", lambda: _reserve(case, order[-1], "bad-evidence"))
    assert _entries(case, SAMPLE_LOG) == ()


@pytest.mark.parametrize(
    "phase,evidence",
    (
        ("TRAIN_FEATURE", ()),
        ("HELD_OUT_FEATURE", None),
        ("HELD_OUT_OUTCOME", None),
        ("HELD_OUT_FEATURE", []),
        ("HELD_OUT_FEATURE", (SOURCE, SOURCE)),
    ),
)
def test_heldout_evidence_is_required_only_on_the_two_heldout_phases(
    tmp_path: Path, phase: str, evidence: Any
) -> None:
    case = _case(tmp_path)
    _rejected("HOLDOUT_EVIDENCE_REQUIRED", lambda: _reserve(case, phase, evidence))
    assert _entries(case, SAMPLE_LOG) == ()
    assert _entries(case, BINDING_LOG) == ()


@pytest.mark.parametrize("phase", ("TRAIN_FEATURE", "HELD_OUT_FEATURE"))
@pytest.mark.parametrize(
    "consumed_at",
    ("2026-01-31", "2026-01-31T00:00:00Z", "2030-01-31T00:00:00.000000Z", True, None),
)
def test_malformed_or_future_governance_time_never_appends(
    tmp_path: Path, phase: str, consumed_at: Any
) -> None:
    case = _case(tmp_path)
    evidence = _evidence(case) if phase.startswith("HELD_OUT") else None
    _rejected(
        "RESERVATION_FAILED", lambda: _reserve(case, phase, evidence, consumed_at)
    )
    assert _entries(case, SAMPLE_LOG) == ()
    assert _entries(case, BINDING_LOG) == ()


def test_two_valid_pre_read_pairs_cannot_replace_the_first_binding(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    first, second = _evidence(case), _evidence(case)
    assert first != second
    assert all(
        case.foundation.read(ref=pair[1][0]).artifact["untouched"] is True
        for pair in (first, second)
    )
    feature = _reserve(case, "HELD_OUT_FEATURE", first)
    error = _rejected(
        "RESERVATION_FAILED", lambda: _reserve(case, "HELD_OUT_FEATURE", second)
    )
    assert (
        isinstance(error.__cause__, FoundationFailure)
        and error.__cause__.code == "LOG_CONFLICT"
    )
    _rejected(
        "HOLDOUT_EVIDENCE_REQUIRED", lambda: _reserve(case, "HELD_OUT_OUTCOME", second)
    )
    outcome = _reserve(case, "HELD_OUT_OUTCOME", first, LATER_AT)
    assert outcome.record.consumed_at == LATER_AT
    assert feature.heldout_binding_entry_ref == outcome.heldout_binding_entry_ref
    assert _reserve(case, "HELD_OUT_FEATURE", first) == feature
    assert _reserve(case, "HELD_OUT_OUTCOME", first, LATER_AT) == outcome
    changed = _rejected(
        "RESERVATION_FAILED", lambda: _reserve(case, "HELD_OUT_OUTCOME", first, AT)
    )
    assert (
        isinstance(changed.__cause__, FoundationFailure)
        and changed.__cause__.code == "LOG_CONFLICT"
    )
    assert (
        len(_entries(case, SAMPLE_LOG)) == 2 and len(_entries(case, BINDING_LOG)) == 1
    )
    assert len(_entries(case, ARTIFACT_LOG)) == 4


def test_outcome_without_binding_never_creates_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    evidence = _evidence(case)
    append = Mock(wraps=case.foundation.append)
    monkeypatch.setattr(case.foundation, "append", append)
    _rejected(
        "HOLDOUT_EVIDENCE_REQUIRED",
        lambda: _reserve(case, "HELD_OUT_OUTCOME", evidence),
    )
    append.assert_not_called()
    assert _entries(case, BINDING_LOG) == () and _entries(case, SAMPLE_LOG) == ()


def test_binding_append_failure_leaves_no_sample_or_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    evidence = _evidence(case)
    failure = FoundationFailure("LOG_PUBLICATION_FAILED")
    append = Mock(side_effect=failure)
    reserve = Mock(wraps=case.ledger.reserve)
    monkeypatch.setattr(case.foundation, "append", append)
    monkeypatch.setattr(case.ledger, "reserve", reserve)
    error = _rejected(
        "RESERVATION_FAILED", lambda: _reserve(case, "HELD_OUT_FEATURE", evidence)
    )
    assert error.__cause__ is failure
    reserve.assert_not_called()
    assert append.call_args.args[0] == BINDING_LOG
    assert _entries(case, BINDING_LOG) == () and _entries(case, SAMPLE_LOG) == ()


def test_binding_only_retry_requires_original_evidence_and_feature_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    first, second = _evidence(case), _evidence(case)
    failure = RuntimeError("before Sample append")
    with monkeypatch.context() as patch:
        patch.setattr(case.ledger, "reserve", Mock(side_effect=failure))
        error = _rejected(
            "RESERVATION_FAILED", lambda: _reserve(case, "HELD_OUT_FEATURE", first)
        )
    assert error.__cause__ is failure
    binding = _entries(case, BINDING_LOG)[0]
    assert _entries(case, SAMPLE_LOG) == ()
    _rejected(
        "HOLDOUT_EVIDENCE_REQUIRED", lambda: _reserve(case, "HELD_OUT_OUTCOME", first)
    )
    for evidence, time in ((second, AT), (first, LATER_AT)):
        changed = _rejected(
            "RESERVATION_FAILED",
            partial(_reserve, case, "HELD_OUT_FEATURE", evidence, time),
        )
        assert (
            isinstance(changed.__cause__, FoundationFailure)
            and changed.__cause__.code == "LOG_CONFLICT"
        )
    feature = _reserve(case, "HELD_OUT_FEATURE", first)
    assert feature.heldout_binding_entry_ref == binding.entry_ref
    assert feature.record.consumed_at == AT
    assert _entries(case, BINDING_LOG) == (binding,)
    assert (
        _reserve(case, "HELD_OUT_OUTCOME", first).heldout_binding_entry_ref
        == binding.entry_ref
    )


@pytest.mark.parametrize("phase", ("HELD_OUT_FEATURE", "HELD_OUT_OUTCOME"))
def test_exception_after_valid_sample_commit_does_not_mean_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    case = _case(tmp_path)
    evidence = _evidence(case)
    if phase.endswith("OUTCOME"):
        _reserve(case, "HELD_OUT_FEATURE", evidence)
    original = case.ledger.reserve
    failure = RuntimeError("after valid Sample commit")
    committed = []

    def commit_then_fail(record: SampleConsumptionRecord, producer: ArtifactRef):
        committed.append(original(record, producer))
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(case.ledger, "reserve", commit_then_fail)
        error = _rejected("RESERVATION_FAILED", lambda: _reserve(case, phase, evidence))
    assert error.__cause__ is failure
    before = _entries(case, SAMPLE_LOG)
    binding = _entries(case, BINDING_LOG)[0]
    receipt = _reserve(case, phase, evidence)
    assert receipt.reservation_entry_ref == committed[0]
    assert receipt.heldout_binding_entry_ref == binding.entry_ref
    assert _entries(case, SAMPLE_LOG) == before
    assert _entries(case, BINDING_LOG) == (binding,)


def test_illegal_committed_sample_time_is_preserved_and_blocks_all_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    evidence = _evidence(case)
    original = case.ledger.reserve

    def commit_illegal_time(record: SampleConsumptionRecord, producer: ArtifactRef):
        return original(
            replace(record, consumed_at="2030-01-01T00:00:00.000000Z"), producer
        )

    with monkeypatch.context() as patch:
        patch.setattr(case.ledger, "reserve", commit_illegal_time)
        error = _rejected(
            "RESERVATION_FAILED", lambda: _reserve(case, "HELD_OUT_FEATURE", evidence)
        )
    assert (
        isinstance(error.__cause__, ValidationCoreFailure)
        and error.__cause__.code == "SAMPLE_LEDGER_CONFLICT"
    )
    samples, bindings = _entries(case, SAMPLE_LOG), _entries(case, BINDING_LOG)
    assert len(samples) == len(bindings) == 1
    assert json.loads(samples[0].payload)["payload"]["record"][
        "consumed_at"
    ].startswith("2030")
    for phase in ("HELD_OUT_FEATURE", "HELD_OUT_OUTCOME"):
        _rejected("HOLDOUT_EVIDENCE_REQUIRED", partial(_reserve, case, phase, evidence))
    assert (
        _entries(case, SAMPLE_LOG) == samples
        and _entries(case, BINDING_LOG) == bindings
    )
    assert len(_entries(case, ARTIFACT_LOG)) == 2


@pytest.mark.parametrize(
    "producer_kind", ("other", "retrospective-feature", "retrospective-outcome")
)
def test_old_empty_snapshot_cannot_hide_known_later_overlap(
    tmp_path: Path, producer_kind: str
) -> None:
    case = _case(tmp_path)
    evidence = _evidence(case)
    producer = (
        ArtifactRef("validation_case", 1, "sha256:" + "b" * 64)
        if producer_kind == "other"
        else case.tasks[
            "HELD_OUT_FEATURE"
            if producer_kind.endswith("feature")
            else "HELD_OUT_OUTCOME"
        ][0]
    )
    purpose = "feature_build" if producer_kind.endswith("feature") else "validation"
    old = case.ledger.reserve(_record(case, producer, purpose=purpose), producer)
    assert case.foundation.read(ref=evidence[1][0]).artifact["untouched"] is True
    _rejected(
        "HOLDOUT_EVIDENCE_REQUIRED",
        lambda: _reserve(case, "HELD_OUT_FEATURE", evidence),
    )
    _rejected(
        "HOLDOUT_EVIDENCE_REQUIRED",
        lambda: _reserve(case, "HELD_OUT_OUTCOME", evidence),
    )
    assert _entries(case, BINDING_LOG) == ()
    assert _entries(case, SAMPLE_LOG)[-1].entry_ref == old


@pytest.mark.parametrize(
    "changes",
    (
        {"dataset_revision": "other-revision"},
        {
            "interval_start": "2000-04-01T00:00:00.000000Z",
            "interval_end": "2000-05-01T00:00:00.000000Z",
        },
        {
            "interval_start": "2000-06-01T00:00:00.000000Z",
            "interval_end": "2000-07-01T00:00:00.000000Z",
        },
    ),
)
def test_revision_and_half_open_nonconflicts_remain_compatible(
    tmp_path: Path, changes: dict[str, str]
) -> None:
    case = _case(tmp_path)
    producer = ArtifactRef("validation_case", 1, "sha256:" + "b" * 64)
    case.ledger.reserve(
        _record(case, producer, purpose="validation", **changes), producer
    )
    evidence = _evidence(case)
    _reserve(case, "HELD_OUT_FEATURE", evidence)
    _reserve(case, "HELD_OUT_OUTCOME", evidence)
    assert len(_entries(case, SAMPLE_LOG)) == 3


@pytest.mark.parametrize(
    "corruption",
    (
        "wrong-purpose",
        "wrong-consumer",
        "wrong-event",
        "extra-record",
        "future-time",
        "wrong-version",
    ),
)
def test_only_exact_canonical_sample_flow_can_be_excluded_from_overlap(
    tmp_path: Path, corruption: str
) -> None:
    case = _case(tmp_path)
    evidence = _evidence(case)
    producer = case.tasks["HELD_OUT_FEATURE"][0]
    record = _record(case, producer)
    wire = {field.name: getattr(record, field.name) for field in fields(record)}
    if corruption == "wrong-purpose":
        wire["purpose"] = "validation"
    elif corruption == "wrong-consumer":
        wire["consumer_id"] = "not-derived"
    elif corruption == "extra-record":
        wire["extra"] = "no"
    elif corruption == "future-time":
        wire["consumed_at"] = "2030-01-01T00:00:00.000000Z"
    envelope = ArtifactEnvelope.create(
        "sample_consumption_append",
        2 if corruption == "wrong-version" else 1,
        {"record": wire, "producer_ref": producer},
    )
    event = (
        "wrong-sample-event"
        if corruption == "wrong-event"
        else _sample_event(record, producer)
    )
    case.foundation.append(SAMPLE_LOG, event, canonical_bytes(envelope))
    _rejected(
        "HOLDOUT_EVIDENCE_REQUIRED",
        lambda: _reserve(case, "HELD_OUT_FEATURE", evidence),
    )
    assert len(_entries(case, SAMPLE_LOG)) == 1 and _entries(case, BINDING_LOG) == ()


@pytest.mark.parametrize(
    "case_name",
    (
        "snapshot-extra",
        "checkpoint-extra",
        "unissued-checkpoint",
        "checkpoint-bool",
        "wrong-sample-log",
        "snapshot-version",
        "assessment-version",
        "assessment-extra",
        "wrong-snapshot",
        "wrong-revision",
        "decision-only",
        "false",
        "bool-as-int",
        "wrong-conflicts",
        "cas-only",
        "wrong-artifact-log",
        "wrong-event",
        "late-proof",
        "bare-refs",
    ),
)
def test_evidence_requires_exact_original_issued_scope_and_truth(
    tmp_path: Path, case_name: str
) -> None:
    case = _case(tmp_path)
    original = _evidence(case)
    snapshot_pub, assessment_pub = original
    snapshot_payload = _wire(case.foundation.read(ref=snapshot_pub[0]).envelope.payload)
    assessment_payload = _wire(
        case.foundation.read(ref=assessment_pub[0]).envelope.payload
    )
    snapshot_version, assessment_version = 1, 1
    if case_name == "snapshot-extra":
        snapshot_payload["extra"] = True
    elif case_name == "checkpoint-extra":
        snapshot_payload["checkpoint"]["extra"] = True
    elif case_name == "unissued-checkpoint":
        snapshot_payload["checkpoint"]["as_of"] = "2026-02-02T00:00:00.000000Z"
    elif case_name == "checkpoint-bool":
        snapshot_payload["checkpoint"]["upper_log_sequence"] = False
    elif case_name == "wrong-sample-log":
        snapshot_payload["checkpoint"]["log_name"] = "different.v1"
    elif case_name == "snapshot-version":
        snapshot_version = 2
    if case_name.startswith("snapshot-") or case_name in (
        "checkpoint-extra",
        "unissued-checkpoint",
        "checkpoint-bool",
        "wrong-sample-log",
    ):
        snapshot_pub = _raw_publication(
            case,
            "sample_consumption_ledger_snapshot",
            snapshot_payload,
            version=snapshot_version,
            log_name=ARTIFACT_LOG,
        )
        assessment_payload["snapshot_ref"] = snapshot_pub[0].to_canonical_dict()
    if case_name == "assessment-version":
        assessment_version = 2
    elif case_name == "assessment-extra":
        assessment_payload["extra"] = True
    elif case_name == "wrong-snapshot":
        assessment_payload["snapshot_ref"] = ArtifactRef(
            "sample_consumption_ledger_snapshot", 1, "sha256:" + "b" * 64
        ).to_canonical_dict()
    elif case_name == "wrong-revision":
        assessment_payload["dataset_revision"] = "other-revision"
    elif case_name == "decision-only":
        assessment_payload["interval_start"] = case.study.partitions[2][2]
        assessment_payload["interval_end"] = case.study.partitions[2][3]
    elif case_name == "false":
        assessment_payload["untouched"] = False
    elif case_name == "bool-as-int":
        assessment_payload["untouched"] = 1
    elif case_name == "wrong-conflicts":
        assessment_payload["conflicting_append_entry_refs"] = [
            {
                "log_name": SAMPLE_LOG,
                "log_sequence": 1,
                "receipt_hash": "sha256:" + "b" * 64,
            }
        ]
    if case_name not in (
        "cas-only",
        "wrong-artifact-log",
        "wrong-event",
        "late-proof",
        "bare-refs",
    ):
        assessment_pub = _raw_publication(
            case,
            "sample_integrity_assessment",
            assessment_payload,
            version=assessment_version,
            log_name=ARTIFACT_LOG,
        )
    elif case_name == "cas-only":
        assessment_payload["dataset_revision"] = "unused-new-CAS-only"
        ref = case.foundation.put(
            envelope=ArtifactEnvelope.create(
                "sample_integrity_assessment", 1, assessment_payload
            )
        )
        assessment_pub = (ref, assessment_pub[1], assessment_pub[2])
    elif case_name == "wrong-artifact-log":
        assessment_pub = _raw_publication(
            case,
            "sample_integrity_assessment",
            assessment_payload,
            log_name="wrong.artifacts.v1",
        )
    elif case_name == "wrong-event":
        assessment_pub = _raw_publication(
            case,
            "sample_integrity_assessment",
            assessment_payload,
            log_name=ARTIFACT_LOG,
            event="wrong-evidence-event",
        )
    elif case_name == "late-proof":
        assessment_pub = (
            assessment_pub[0],
            assessment_pub[1],
            case.foundation.checkpoint(ARTIFACT_LOG),
        )
    evidence: Any = (snapshot_pub, assessment_pub)
    if case_name == "bare-refs":
        evidence = (snapshot_pub[0], assessment_pub[0])
    error = _rejected(
        "HOLDOUT_EVIDENCE_REQUIRED",
        lambda: _reserve(case, "HELD_OUT_FEATURE", evidence),
    )
    if case_name == "unissued-checkpoint":
        assert (
            isinstance(error.__cause__, FoundationFailure)
            and error.__cause__.code == "LOG_INTEGRITY"
        )
    assert _entries(case, BINDING_LOG) == () and _entries(case, SAMPLE_LOG) == ()


def test_forged_true_assessment_cannot_hide_conflicts_or_future_records(
    tmp_path: Path,
) -> None:
    for future in (False, True):
        case = _case(tmp_path / str(future))
        producer = ArtifactRef("validation_case", 1, "sha256:" + "b" * 64)
        record = _record(
            case,
            producer,
            purpose="validation",
            **({"consumed_at": "2030-01-01T00:00:00.000000Z"} if future else {}),
        )
        envelope = ArtifactEnvelope.create(
            "sample_consumption_append",
            1,
            {
                "record": {
                    field.name: getattr(record, field.name) for field in fields(record)
                },
                "producer_ref": producer,
            },
        )
        case.foundation.append(
            SAMPLE_LOG, _sample_event(record, producer), canonical_bytes(envelope)
        )
        checkpoint = case.foundation.checkpoint(SAMPLE_LOG)
        snapshot = _raw_publication(
            case,
            "sample_consumption_ledger_snapshot",
            {
                "checkpoint": {
                    field.name: getattr(checkpoint, field.name)
                    for field in fields(checkpoint)
                }
            },
            log_name=ARTIFACT_LOG,
        )
        data = case.study.partitions[2][1]
        assessment = _raw_publication(
            case,
            "sample_integrity_assessment",
            {
                "snapshot_ref": snapshot[0],
                "dataset_revision": data.dataset_revision,
                "interval_start": data.interval_start,
                "interval_end": data.interval_end,
                "untouched": True,
                "conflicting_append_entry_refs": (),
            },
            log_name=ARTIFACT_LOG,
        )
        _rejected(
            "HOLDOUT_EVIDENCE_REQUIRED",
            partial(_reserve, case, "HELD_OUT_FEATURE", (snapshot, assessment)),
        )
        assert (
            _entries(case, BINDING_LOG) == () and len(_entries(case, SAMPLE_LOG)) == 1
        )


def test_refresh_after_feature_never_replaces_bound_pre_read_evidence(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    original = _evidence(case)
    feature = _reserve(case, "HELD_OUT_FEATURE", original)
    refreshed = _evidence(case)
    assert case.foundation.read(ref=refreshed[1][0]).artifact["untouched"] is False
    for phase in ("HELD_OUT_FEATURE", "HELD_OUT_OUTCOME"):
        _rejected(
            "HOLDOUT_EVIDENCE_REQUIRED", partial(_reserve, case, phase, refreshed)
        )
    outcome = _reserve(case, "HELD_OUT_OUTCOME", original)
    assert outcome.heldout_binding_entry_ref == feature.heldout_binding_entry_ref
    assert len(_entries(case, BINDING_LOG)) == 1


@pytest.mark.parametrize(
    "path,value",
    (
        (("extra",), True),
        (("kind",), "different"),
        (("schema_version",), True),
        (("study_publication", "extra"), True),
        (("study_publication", "artifact_ref", "extra"), True),
        (("study_publication", "artifact_ref", "content_hash"), "sha256:" + "f" * 64),
        (("study_publication", "entry_ref", "extra"), True),
        (("study_publication", "entry_ref", "log_sequence"), True),
        (("study_publication", "checkpoint", "extra"), True),
        (("study_publication", "checkpoint", "upper_log_sequence"), True),
        (("heldout_feature_publication", "entry_ref", "log_name"), "wrong.v1"),
        (
            ("heldout_feature_publication", "checkpoint", "head_receipt_hash"),
            "sha256:" + "f" * 64,
        ),
        (
            ("snapshot_publication", "checkpoint", "as_of"),
            "2026-02-02T00:00:00.000000Z",
        ),
        (("assessment_publication", "artifact_ref", "schema_version"), True),
        (("data_slice", "extra"), True),
        (("data_slice", "dataset_revision"), "other-revision"),
        (("data_slice", "interval_start"), "2000-05-10T00:00:00.000000Z"),
        (("feature_record", "extra"), True),
        (("feature_record", "purpose"), "validation"),
        (("feature_record", "consumer_id"), "not-derived"),
        (("feature_record", "consumed_at"), "2030-01-01T00:00:00.000000Z"),
        (("feature_record", "consumed_at"), "2026-01-31T00:00:00Z"),
    ),
)
def test_binding_has_strict_nested_fields_and_original_proofs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: tuple[str, ...], value: Any
) -> None:
    case = _case(tmp_path)
    evidence = _evidence(case)
    body = json.loads(_capture_binding(case, evidence, monkeypatch))
    _set(body, path, value)
    case.foundation.append(BINDING_LOG, _binding_event(case), canonical_bytes(body))
    # Valid actual Feature entry after binding: a missing-Feature failure cannot
    # accidentally make a malformed binding test pass.
    case.ledger.reserve(
        _record(case, case.tasks["HELD_OUT_FEATURE"][0]),
        case.tasks["HELD_OUT_FEATURE"][0],
    )
    _rejected(
        "HOLDOUT_EVIDENCE_REQUIRED",
        lambda: _reserve(case, "HELD_OUT_OUTCOME", evidence),
    )
    assert len(_entries(case, SAMPLE_LOG)) == len(_entries(case, BINDING_LOG)) == 1


@pytest.mark.parametrize(
    "corruption",
    (
        "noncanonical-json",
        "float",
        "wrong-event",
        "missing-root-key",
        "missing-record-key",
        "invalid-json",
    ),
)
def test_binding_raw_bytes_cannot_be_reinterpreted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    case = _case(tmp_path)
    evidence = _evidence(case)
    source = _capture_binding(case, evidence, monkeypatch)
    body = json.loads(source)
    event = _binding_event(case)
    if corruption == "float":
        body["schema_version"] = 1.0
        source = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    elif corruption == "noncanonical-json":
        source = json.dumps(body, indent=2).encode()
    elif corruption == "wrong-event":
        event = "wrong-binding-event"
    elif corruption == "missing-root-key":
        del body["data_slice"]
        source = canonical_bytes(body)
    elif corruption == "missing-record-key":
        del body["feature_record"]["consumed_at"]
        source = canonical_bytes(body)
    elif corruption == "invalid-json":
        source = b"{"
    case.foundation.append(BINDING_LOG, event, source)
    case.ledger.reserve(
        _record(case, case.tasks["HELD_OUT_FEATURE"][0]),
        case.tasks["HELD_OUT_FEATURE"][0],
    )
    _rejected(
        "HOLDOUT_EVIDENCE_REQUIRED",
        lambda: _reserve(case, "HELD_OUT_OUTCOME", evidence),
    )
    assert len(_entries(case, SAMPLE_LOG)) == 1


@pytest.mark.parametrize(
    "order",
    ("missing-feature", "feature-before-binding", "outcome-before-feature", "correct"),
)
def test_actual_global_order_not_wall_clock_controls_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: str
) -> None:
    case = _case(tmp_path, constant_clock=True)
    evidence = _evidence(case)
    body = _capture_binding(case, evidence, monkeypatch)
    feature_ref, outcome_ref = (
        case.tasks["HELD_OUT_FEATURE"][0],
        case.tasks["HELD_OUT_OUTCOME"][0],
    )
    if order == "feature-before-binding":
        case.ledger.reserve(_record(case, feature_ref), feature_ref)
    case.foundation.append(BINDING_LOG, _binding_event(case), body)
    if order == "outcome-before-feature":
        case.ledger.reserve(
            _record(case, outcome_ref, purpose="validation"), outcome_ref
        )
    if order in ("correct", "outcome-before-feature"):
        case.ledger.reserve(_record(case, feature_ref), feature_ref)
    append = Mock(wraps=case.foundation.append)
    monkeypatch.setattr(case.foundation, "append", append)
    if order == "correct":
        assert (
            _reserve(case, "HELD_OUT_OUTCOME", evidence).heldout_binding_entry_ref
            == _entries(case, BINDING_LOG)[0].entry_ref
        )
        assert all(call.args[0] == SAMPLE_LOG for call in append.call_args_list)
    else:
        before = _entries(case, SAMPLE_LOG)
        _rejected(
            "HOLDOUT_EVIDENCE_REQUIRED",
            lambda: _reserve(case, "HELD_OUT_OUTCOME", evidence),
        )
        append.assert_not_called()
        assert _entries(case, SAMPLE_LOG) == before


@pytest.mark.parametrize("corruption", ("wrong-bytes", "wrong-event"))
def test_returned_binding_receipt_must_prove_requested_bytes_before_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    case = _case(tmp_path)
    evidence = _evidence(case)
    original = case.foundation.append

    def wrong_binding(log_name: str, event: str, payload: bytes):
        if log_name == BINDING_LOG:
            return original(
                log_name,
                event if corruption == "wrong-bytes" else "wrong-returned-event",
                canonical_bytes({}) if corruption == "wrong-bytes" else payload,
            )
        return original(log_name, event, payload)

    reserve = Mock(wraps=case.ledger.reserve)
    monkeypatch.setattr(case.foundation, "append", wrong_binding)
    monkeypatch.setattr(case.ledger, "reserve", reserve)
    _rejected(
        "RESERVATION_FAILED", lambda: _reserve(case, "HELD_OUT_FEATURE", evidence)
    )
    reserve.assert_not_called()
    assert len(_entries(case, BINDING_LOG)) == 1 and _entries(case, SAMPLE_LOG) == ()


@pytest.mark.parametrize("wrong_result", ("wrong-log", "wrong-sample"))
def test_returned_sample_reference_requires_actual_exact_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrong_result: str
) -> None:
    case = _case(tmp_path)
    evidence = _evidence(case)
    returned = case.study_pub[1]
    if wrong_result == "wrong-sample":
        producer = ArtifactRef("validation_case", 1, "sha256:" + "b" * 64)
        returned = case.ledger.reserve(
            _record(
                case,
                producer,
                purpose="validation",
                dataset_revision="different-revision",
            ),
            producer,
        )
    monkeypatch.setattr(case.ledger, "reserve", Mock(return_value=returned))
    _rejected(
        "RESERVATION_FAILED", lambda: _reserve(case, "HELD_OUT_FEATURE", evidence)
    )
    assert len(_entries(case, BINDING_LOG)) == 1
    assert len(_entries(case, SAMPLE_LOG)) == (
        1 if wrong_result == "wrong-sample" else 0
    )


def test_different_foundation_and_ledger_cannot_return_a_success_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path / "requested")
    other = LocalFoundation(tmp_path / "other", clock=lambda: CLOCK)
    ledger = SampleConsumptionLedger(other)
    evidence = _evidence(case)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("io")
        raise AssertionError("foreign ledger must fail before declaration or binding I/O")

    with monkeypatch.context() as guarded:
        for name in ("read", "entries", "put", "append", "checkpoint"):
            guarded.setattr(LocalFoundation, name, forbidden)
        error = _rejected(
            "RESERVATION_FAILED",
            lambda: reserve_statistical_read(
                case.foundation,
                ledger,
                case.study_pub,
                case.tasks["HELD_OUT_FEATURE"],
                AT,
                evidence,
            ),
        )
    assert isinstance(error.__cause__, ValidationCoreFailure)
    assert error.__cause__.code == "SAMPLE_LEDGER_CONFLICT"
    assert calls == []
    assert _entries(case, BINDING_LOG) == _entries(case, SAMPLE_LOG) == ()
    assert other.entries(SAMPLE_LOG, other.checkpoint(SAMPLE_LOG)) == ()


def test_every_new_lookup_has_a_cutoff_and_no_evidence_is_refreshed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    evidence = _evidence(
        case
    )  # Old ledger publication finder is deliberately unchanged.
    with pytest.raises(FoundationFailure) as raised:
        case.foundation.entries(RESEARCH_LOG, case.study_pub[2])
    assert raised.value.code == "LOG_INTEGRITY"
    original = case.foundation.entries
    seen = []

    def bounded(log_name: str, through: Any = None):
        assert through is not None
        assert type(through) in (LogCheckpoint, LogEntryRef)
        seen.append((log_name, through))
        return original(log_name, through)

    monkeypatch.setattr(case.foundation, "entries", bounded)
    for phase in PHASES:
        held = phase.startswith("HELD_OUT")
        _reserve(case, phase, evidence if held else None)
    _reserve(case, "HELD_OUT_FEATURE", evidence)
    _reserve(case, "HELD_OUT_OUTCOME", evidence)
    assert {name for name, _ in seen} == {
        RESEARCH_LOG,
        ARTIFACT_LOG,
        SAMPLE_LOG,
        BINDING_LOG,
    }
    assert len(_entries(case, ARTIFACT_LOG)) == 2
    assert (
        len(_entries(case, BINDING_LOG)) == 1 and len(_entries(case, SAMPLE_LOG)) == 6
    )


def test_public_failure_codes_are_separate_from_financial_failure() -> None:
    error = StatisticalReservationError("STATISTICAL_RESERVATION_FAILED")
    assert str(error) == error.code
    assert not isinstance(error, ValidationCoreFailure)
    with pytest.raises(ValueError, match="unknown"):
        StatisticalReservationError("FAILED")


def test_a_genuine_task_published_before_its_study_is_not_a_valid_phase(
    tmp_path: Path,
) -> None:
    template = _case(tmp_path / "template", full=False)
    foundation = LocalFoundation(tmp_path / "target", clock=lambda: CLOCK)
    case = Case(
        foundation,
        SampleConsumptionLedger(foundation),
        template.study,
        template.study_pub,
        {},
    )
    task = _task(case, "TRAIN_FEATURE")
    case.tasks[task.phase] = _raw_publication(
        case, task.artifact_type, _wire(task.payload)
    )
    case.study_pub = publish_statistical_study(case.study, foundation).publication
    _rejected("PREREQUISITE_INVALID", lambda: _reserve(case, task.phase))
    assert _entries(case, SAMPLE_LOG) == ()


@pytest.mark.parametrize(
    "field,value",
    (
        ("entry", "wrong-head"),
        ("checkpoint", "late-time"),
        ("ref", "missing-CAS"),
    ),
)
def test_study_original_publication_is_checked_before_task_scope(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    case = _case(tmp_path)
    ref, entry, checkpoint = case.study_pub
    if field == "entry":
        entry = replace(entry, receipt_hash="sha256:" + "f" * 64)
    elif field == "checkpoint":
        checkpoint = replace(checkpoint, as_of="2026-02-02T00:00:00.000000Z")
    else:
        ref = ArtifactRef(ref.artifact_type, 1, "sha256:" + "f" * 64)
    case.study_pub = ref, entry, checkpoint
    _rejected("PUBLICATION_MISSING", lambda: _reserve(case, "TRAIN_FEATURE"))
    assert _entries(case, SAMPLE_LOG) == ()


def test_all_four_retrospective_public_producers_coexist_with_legacy_heldout(
    tmp_path: Path,
) -> None:
    """Distinct synthetic revisions; this proves no source-alias completeness."""
    from crypto_quant_research import (
        RetrospectiveReadTask,
        RetrospectiveSourceSlice,
        RetrospectiveStudyDeclaration,
        RetrospectiveStudyProtocol,
        publish_retrospective_protocol,
        publish_retrospective_read_task,
        publish_retrospective_study,
    )
    from crypto_quant_validation import reserve_retrospective_read

    case = _case(tmp_path, constant_clock=True)
    evidence = _evidence(case)
    f = case.foundation
    h = tuple("sha256:" + c * 64 for c in "123456789abcdef")
    p = RetrospectiveStudyProtocol(
        "UNRELATED-SYNTHETIC-LOCAL-CAPTURE",
        h[0],
        h[1],
        h[2],
        h[3],
        h[4],
        (("capture", "2000-01-01T00:00:00.000000Z", "2000-07-01T00:00:00.000000Z"),),
    )
    pp = publish_retrospective_protocol(p, f)
    producers = set()

    def append(phase, source, sp=None, **hashes):
        task = RetrospectiveReadTask(
            pp.ref if sp is None else sp.ref, phase, source, **hashes
        )
        tp = publish_retrospective_read_task(task, f)
        reserve_retrospective_read(
            f,
            case.ledger,
            pp.publication,
            tp.publication,
            CLOCK,
            None if sp is None else sp.publication,
        )
        producers.add(tp.ref.artifact_type)

    append("INPUT_QUALIFICATION", p.source_slices[0])
    append("SNAPSHOT_FREEZE", p.source_slices[0], input_check_hash=h[9])
    parts = tuple(
        (
            role,
            RetrospectiveSourceSlice(
                "capture",
                p.capture_revision,
                p.source_scope_hash,
                f"2000-{m:02d}-01T00:00:00.000000Z",
                f"2000-{m + 1:02d}-01T00:00:00.000000Z",
                h[5],
            ),
            f"2000-{m:02d}-10T00:00:00.000000Z",
            f"2000-{m:02d}-20T00:00:00.000000Z",
        )
        for role, m in (("TRAIN", 1), ("DEVELOPMENT", 3), ("RETROSPECTIVE_EVAL", 5))
    )
    sp = publish_retrospective_study(
        RetrospectiveStudyDeclaration(pp.ref, h[5], h[6], h[7], h[8], parts), f
    )
    for phase, index, hashes in (
        ("TRAIN_FEATURE", 0, {}),
        ("TRAIN_FIT", 0, {"feature_rows_hash": h[10]}),
        ("DEVELOPMENT_FEATURE", 1, {"model_cards_hash": h[11]}),
        ("RETROSPECTIVE_EVAL_FEATURE", 2, {"model_cards_hash": h[11]}),
        (
            "DEVELOPMENT_OUTCOME",
            1,
            {"model_cards_hash": h[11], "predictions_hash": h[12]},
        ),
        (
            "RETROSPECTIVE_EVAL_OUTCOME",
            2,
            {"model_cards_hash": h[11], "predictions_hash": h[12]},
        ),
    ):
        append(phase, parts[index][1], sp, **hashes)
    assert len(_entries(case, SAMPLE_LOG)) == 8
    assert len(producers) == 4
    feature = _reserve(case, "HELD_OUT_FEATURE", evidence)
    outcome = _reserve(case, "HELD_OUT_OUTCOME", evidence)
    assert feature.heldout_binding_entry_ref == outcome.heldout_binding_entry_ref
    assert len(_entries(case, SAMPLE_LOG)) == 10


@pytest.mark.parametrize(
    "producer,purpose",
    (
        ("retrospective_input_read_task", "discovery"),
        ("retrospective_feature_read_task", "feature_build"),
        ("retrospective_training_read_task", "model_training"),
        ("retrospective_outcome_read_task", "validation"),
    ),
)
def test_retrospective_same_revision_overlap_still_blocks_old_heldout(
    tmp_path, producer, purpose
):
    case = _case(tmp_path)
    evidence = _evidence(case)
    ref = ArtifactRef(producer, 1, "sha256:" + "e" * 64)
    case.ledger.reserve(_record(case, ref, purpose=purpose), ref)
    _rejected(
        "HOLDOUT_EVIDENCE_REQUIRED",
        lambda: _reserve(case, "HELD_OUT_FEATURE", evidence),
    )
    assert not _entries(case, BINDING_LOG)


def test_assessment_cannot_precede_its_actual_snapshot_publication(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    checkpoint = case.foundation.checkpoint(SAMPLE_LOG)
    payload = {
        "checkpoint": {
            field.name: getattr(checkpoint, field.name) for field in fields(checkpoint)
        }
    }
    snapshot_ref = case.foundation.put(
        envelope=ArtifactEnvelope.create(
            "sample_consumption_ledger_snapshot", 1, payload
        )
    )
    data = case.study.partitions[2][1]
    assessment = _raw_publication(
        case,
        "sample_integrity_assessment",
        {
            "snapshot_ref": snapshot_ref,
            "dataset_revision": data.dataset_revision,
            "interval_start": data.interval_start,
            "interval_end": data.interval_end,
            "untouched": True,
            "conflicting_append_entry_refs": (),
        },
        log_name=ARTIFACT_LOG,
    )
    snapshot = _raw_publication(
        case, "sample_consumption_ledger_snapshot", payload, log_name=ARTIFACT_LOG
    )
    _rejected(
        "HOLDOUT_EVIDENCE_REQUIRED",
        lambda: _reserve(case, "HELD_OUT_FEATURE", (snapshot, assessment)),
    )
    assert _entries(case, BINDING_LOG) == () and _entries(case, SAMPLE_LOG) == ()
