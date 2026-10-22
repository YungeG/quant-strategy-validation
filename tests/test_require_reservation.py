from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn, get_type_hints

import pytest

import crypto_quant_validation as validation
from crypto_quant_domain import ArtifactEnvelope, ArtifactRef, canonical_bytes, canonical_sha256
from crypto_quant_foundation import FoundationFailure, LocalFoundation, LogCheckpoint, LogEntryRef
from crypto_quant_validation import (
    SampleConsumptionLedger, SampleConsumptionRecord, ValidationCoreFailure,
)


def test_public_require_reservation_surface() -> None:
    expected = Path(__file__).resolve().parents[1] / "src/crypto_quant_validation/ledger.py"
    assert Path(inspect.getfile(SampleConsumptionLedger)).resolve() == expected
    assert "SampleConsumptionLedger" in validation.__all__
    method = getattr(SampleConsumptionLedger, "require_reservation", None)
    assert callable(method), "public require_reservation method is missing"
    signature = inspect.signature(method)
    assert tuple(signature.parameters) == (
        "self", "record", "producer_ref", "before_entry_ref",
    )
    assert signature.parameters["before_entry_ref"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["before_entry_ref"].default is None
    assert get_type_hints(method) == {
        "record": SampleConsumptionRecord,
        "producer_ref": ArtifactRef,
        "before_entry_ref": LogEntryRef | None,
        "return": LogEntryRef,
    }


SAMPLE_LOG = "validation.sample-consumption.v1"
BOUND_LOG = "fixture.reservation-bound.v1"
RESERVED_AT = "2026-02-01T00:00:00.000000Z"
ACCEPTED_AT = "2026-02-01T00:00:01.000000Z"
PRODUCERS = (
    ("feature_build_task", "feature_build"),
    ("model_training_task", "model_training"),
    ("trial_declaration", "discovery"),
    ("selection_declaration", "selection"),
    ("validation_case", "validation"),
    ("statistical_feature_read_task", "feature_build"),
    ("statistical_training_read_task", "model_training"),
    ("statistical_outcome_read_task", "validation"),
    ("retrospective_input_read_task", "discovery"),
    ("retrospective_feature_read_task", "feature_build"),
    ("retrospective_training_read_task", "model_training"),
    ("retrospective_outcome_read_task", "validation"),
)


def _producer(kind: str = "trial_declaration", marker: str = "a") -> ArtifactRef:
    return ArtifactRef(kind, 1, "sha256:" + marker * 64)


def _record(producer: ArtifactRef, purpose: str = "discovery") -> SampleConsumptionRecord:
    return SampleConsumptionRecord(
        "l0-synthetic-v1",
        "2026-01-01T00:00:00.000000Z",
        "2026-02-01T00:00:00.000000Z",
        purpose,
        canonical_sha256(("sample-consumer-v1", producer)),
        RESERVED_AT,
    )


def _forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("read path must not publish, reserve, checkpoint or call the clock")


def _retained_files(root: Path) -> dict[str, tuple[bytes, int, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize(("kind", "purpose"), PRODUCERS, ids=[pair[0] for pair in PRODUCERS])
def test_require_reservation_returns_original_entry_without_publication(
    tmp_path: Path, kind: str, purpose: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation = LocalFoundation(tmp_path, clock=lambda: ACCEPTED_AT)
    ledger = SampleConsumptionLedger(foundation)
    producer = _producer(kind)
    record = _record(producer, purpose)
    original = ledger.reserve(record, producer)
    bound = foundation.append(BOUND_LOG, "after-reservation", b"accepted boundary").entry_ref
    entries_before = foundation.entries(SAMPLE_LOG)
    files_before = _retained_files(tmp_path)
    reopened = SampleConsumptionLedger(LocalFoundation(tmp_path, clock=_forbidden))
    for name in ("put", "append", "checkpoint"):
        monkeypatch.setattr(LocalFoundation, name, _forbidden)
    monkeypatch.setattr(SampleConsumptionLedger, "reserve", _forbidden)
    for _ in range(2):
        assert reopened.require_reservation(record, producer) == original
        assert reopened.require_reservation(record, producer, before_entry_ref=bound) == original
    assert foundation.entries(SAMPLE_LOG) == entries_before
    assert _retained_files(tmp_path) == files_before
    assert not (tmp_path / ".foundation.write.lock").exists()


def _reserved(root: Path):
    foundation = LocalFoundation(root, clock=lambda: ACCEPTED_AT)
    ledger = SampleConsumptionLedger(foundation)
    producer = _producer()
    record = _record(producer)
    original = ledger.reserve(record, producer)
    return foundation, ledger, producer, record, original


def _sample_conflict(call: Callable[[], object]) -> ValidationCoreFailure:
    with pytest.raises(ValidationCoreFailure) as raised:
        call()
    assert raised.value.code == "SAMPLE_LEDGER_CONFLICT"
    return raised.value


@pytest.mark.parametrize("mutation", (
    "missing", "revision", "start", "end", "producer", "consumed_at",
    "wrong-log", "cas-only", "duplicate",
))
def test_missing_or_wrong_reservation_does_not_append(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation = LocalFoundation(tmp_path / "subject", clock=lambda: ACCEPTED_AT)
    ledger = SampleConsumptionLedger(foundation)
    producer, record = _producer(), _record(_producer())
    if mutation in {"wrong-log", "cas-only"}:
        donor, _, _, _, _ = _reserved(tmp_path / "donor")
        entry = donor.entries(SAMPLE_LOG)[0]
        if mutation == "wrong-log":
            foundation.append("fixture.wrong-owner.v1", entry.event_id, entry.payload)
        else:
            foundation.put(envelope=ArtifactEnvelope(**json.loads(entry.payload)))
    elif mutation != "missing":
        ledger.reserve(record, producer)
    if mutation == "revision":
        record = replace(record, dataset_revision="other-revision")
    elif mutation == "start":
        record = replace(record, interval_start="2026-01-02T00:00:00.000000Z")
    elif mutation == "end":
        record = replace(record, interval_end="2026-01-31T00:00:00.000000Z")
    elif mutation == "producer":
        producer = _producer(marker="b")
        record = _record(producer)
    elif mutation == "consumed_at":
        record = replace(record, consumed_at="2026-02-01T00:00:00.000001Z")
    elif mutation == "duplicate":
        entries = foundation.entries(SAMPLE_LOG)
        monkeypatch.setattr(foundation, "entries", lambda *args: entries + entries)
    before = _retained_files(tmp_path)
    for name in ("put", "append", "checkpoint"):
        monkeypatch.setattr(LocalFoundation, name, _forbidden)
    monkeypatch.setattr(SampleConsumptionLedger, "reserve", _forbidden)
    _sample_conflict(lambda: ledger.require_reservation(record, producer))
    assert _retained_files(tmp_path) == before


@pytest.mark.parametrize("mutation", (
    "record-mapping", "record-subclass", "record-revision", "record-time",
    "producer-mapping", "producer-subclass", "producer-bool", "producer-float",
    "producer-version", "producer-hash", "purpose", "consumer",
    "bound-mapping", "bound-subclass", "bound-checkpoint", "bound-int",
    "bound-log", "bound-zero", "bound-bool", "bound-float", "bound-hash",
))
def test_invalid_arguments_precede_storage(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation = LocalFoundation(tmp_path / "never-opened", clock=_forbidden)
    ledger = SampleConsumptionLedger(foundation)
    producer: Any = _producer()
    record: Any = _record(producer)
    bound: Any = None
    if mutation == "record-mapping":
        record = {}
    elif mutation == "record-subclass":
        record = type("RecordSubclass", (SampleConsumptionRecord,), {})(
            record.dataset_revision, record.interval_start, record.interval_end,
            record.purpose, record.consumer_id, record.consumed_at,
        )
    elif mutation.startswith("record-"):
        object.__setattr__(record, "dataset_revision" if mutation == "record-revision" else "consumed_at", "")
    elif mutation == "producer-mapping":
        producer = {}
    elif mutation == "producer-subclass":
        producer = type("RefSubclass", (ArtifactRef,), {})(
            producer.artifact_type, producer.schema_version, producer.content_hash,
        )
    elif mutation in {"producer-bool", "producer-float", "producer-version"}:
        object.__setattr__(producer, "schema_version", {
            "producer-bool": True, "producer-float": 1.0, "producer-version": 2,
        }[mutation])
    elif mutation == "producer-hash":
        object.__setattr__(producer, "content_hash", "broken")
    elif mutation == "purpose":
        record = replace(record, purpose="validation")
    elif mutation == "consumer":
        record = replace(record, consumer_id="not-derived")
    else:
        bound = LogEntryRef(BOUND_LOG, 1, "sha256:" + "0" * 64)
        if mutation == "bound-mapping":
            bound = {"log_name": BOUND_LOG, "log_sequence": 1, "receipt_hash": bound.receipt_hash}
        elif mutation == "bound-subclass":
            bound = type("EntrySubclass", (LogEntryRef,), {})(BOUND_LOG, 1, bound.receipt_hash)
        elif mutation == "bound-checkpoint":
            bound = LogCheckpoint(BOUND_LOG, ACCEPTED_AT, 0, None)
        elif mutation == "bound-int":
            bound = 100
        elif mutation == "bound-log":
            bound = replace(bound, log_name="../not-a-log")
        elif mutation in {"bound-zero", "bound-bool", "bound-float"}:
            bound = replace(bound, log_sequence={"bound-zero": 0, "bound-bool": True, "bound-float": 1.0}[mutation])
        elif mutation == "bound-hash":
            bound = replace(bound, receipt_hash="broken")
    malformed_bound = mutation in {"bound-log", "bound-zero", "bound-bool", "bound-float", "bound-hash"}
    if malformed_bound:
        monkeypatch.setattr(Path, "mkdir", _forbidden)
    else:
        monkeypatch.setattr(foundation, "entries", _forbidden)
    type_cases = {
        "record-mapping", "record-subclass", "producer-mapping", "producer-subclass",
        "bound-mapping", "bound-subclass", "bound-checkpoint", "bound-int",
        "bound-bool", "bound-float",
    }
    with pytest.raises(TypeError if mutation in type_cases else ValueError):
        ledger.require_reservation(record, producer, before_entry_ref=bound)
    assert not (tmp_path / "never-opened").exists()


def test_before_entry_uses_verified_global_sequence(tmp_path: Path) -> None:
    foundation = LocalFoundation(tmp_path, clock=lambda: ACCEPTED_AT)
    before = foundation.append("fixture.earlier.v1", "before", b"before").entry_ref
    ledger = SampleConsumptionLedger(foundation)
    producer, record = _producer(), _record(_producer())
    original = ledger.reserve(record, producer)
    after = foundation.append(BOUND_LOG, "after", b"after").entry_ref
    observed = [foundation.entries(ref.log_name, ref)[-1] for ref in (before, original, after)]
    assert [entry.log_sequence for entry in observed] == [1, 1, 1]
    assert [entry.ledger_sequence for entry in observed] == [1, 2, 3]
    assert {entry.accepted_at for entry in observed} == {ACCEPTED_AT}
    assert ledger.require_reservation(record, producer, before_entry_ref=after) == original
    for bound in (before, original):
        _sample_conflict(lambda: ledger.require_reservation(record, producer, before_entry_ref=bound))


@pytest.mark.parametrize("mutation", ("missing-log", "future-sequence", "forged-receipt", "copied-log"))
def test_missing_or_forged_bound_preserves_foundation_failure(tmp_path: Path, mutation: str) -> None:
    foundation, ledger, producer, record, _ = _reserved(tmp_path)
    bound = foundation.append(BOUND_LOG, "bound", b"bound").entry_ref
    if mutation == "missing-log":
        bound = replace(bound, log_name="fixture.absent.v1")
    elif mutation == "future-sequence":
        bound = replace(bound, log_sequence=2)
    elif mutation == "forged-receipt":
        bound = replace(bound, receipt_hash="sha256:" + "0" * 64)
    else:
        foundation.append("fixture.other-bound.v1", "bound", b"bound")
        bound = replace(bound, log_name="fixture.other-bound.v1")
    before = _retained_files(tmp_path)
    with pytest.raises(FoundationFailure) as raised:
        ledger.require_reservation(record, producer, before_entry_ref=bound)
    assert raised.value.code == "LOG_INTEGRITY"
    assert _retained_files(tmp_path) == before


@pytest.mark.parametrize("mutation", (
    "json", "noncanonical", "envelope-kind", "version", "keys", "event",
    "purpose", "consumer", "late-time",
))
def test_sample_semantic_corruption_is_not_skipped(tmp_path: Path, mutation: str) -> None:
    foundation, ledger, producer, record, original = _reserved(tmp_path / "store")
    assert ledger.require_reservation(record, producer) == original
    donor = LocalFoundation(tmp_path / "donor", clock=lambda: ACCEPTED_AT)
    tail_producer = _producer(marker="b")
    SampleConsumptionLedger(donor).reserve(_record(tail_producer), tail_producer)
    entry = donor.entries(SAMPLE_LOG)[0]
    decoded = json.loads(entry.payload)
    payload = copy.deepcopy(decoded["payload"])
    kind, version = "sample_consumption_append", 1
    event = entry.event_id
    if mutation == "envelope-kind":
        kind = "not_a_reservation"
    elif mutation == "version":
        version = 2
    elif mutation == "keys":
        payload["extra"] = None
    elif mutation == "purpose":
        payload["record"]["purpose"] = "validation"
    elif mutation == "consumer":
        payload["record"]["consumer_id"] = "not-derived"
    elif mutation == "late-time":
        payload["record"]["consumed_at"] = "2026-02-01T00:00:02.000000Z"
    corrupt = canonical_bytes(ArtifactEnvelope.create(kind, version, payload))
    if mutation == "json":
        corrupt = b"not-json"
    elif mutation == "noncanonical":
        corrupt = b" " + entry.payload
    # An unrelated semantically invalid tail must not be skipped after a good match.
    if mutation == "event":
        event = "corrupt-tail-event"
    foundation.append(SAMPLE_LOG, event, corrupt)
    before = _retained_files(tmp_path)
    _sample_conflict(lambda: ledger.require_reservation(record, producer))
    assert _retained_files(tmp_path) == before


@pytest.mark.parametrize("mutation, expected", (
    ("tamper", "LOG_INTEGRITY"),
    ("foreign-lock", "WRITE_LOCK_UNAVAILABLE"),
    ("unavailable-read", "LOG_INTEGRITY"),
    ("root-symlink", "UNSUPPORTED_FILESYSTEM"),
))
def test_structural_storage_failures_are_preserved(
    tmp_path: Path, mutation: str, expected: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation, ledger, producer, record, _ = _reserved(tmp_path / "store")
    log_path = tmp_path / "store" / "registries" / (SAMPLE_LOG + ".jsonl")
    if mutation == "tamper":
        log_path.write_bytes(log_path.read_bytes()[:-1])
    elif mutation == "foreign-lock":
        (tmp_path / "store" / ".foundation.write.lock").write_bytes(b"other writer")
    elif mutation == "root-symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path / "store", target_is_directory=True)
        ledger = SampleConsumptionLedger(LocalFoundation(alias, clock=_forbidden))
    before = _retained_files(tmp_path)
    with monkeypatch.context() as patched:
        if mutation == "unavailable-read":
            read_bytes = Path.read_bytes
            def fail_registry(path: Path) -> bytes:
                if path == log_path:
                    raise PermissionError("synthetic unavailable registry")
                return read_bytes(path)
            patched.setattr(Path, "read_bytes", fail_registry)
        with pytest.raises(FoundationFailure) as raised:
            ledger.require_reservation(record, producer)
    assert raised.value.code == expected
    assert _retained_files(tmp_path) == before


def test_failure_precedence_is_ordered(tmp_path: Path) -> None:
    foundation = LocalFoundation(tmp_path, clock=lambda: ACCEPTED_AT)
    foundation.append(SAMPLE_LOG, "malformed-sample", b"not-json")
    ledger = SampleConsumptionLedger(foundation)
    producer, record = _producer(), _record(_producer())
    absent_bound = LogEntryRef(BOUND_LOG, 1, "sha256:" + "0" * 64)
    with pytest.raises(FoundationFailure) as raised:
        ledger.require_reservation(record, producer, before_entry_ref=absent_bound)
    assert raised.value.code == "LOG_INTEGRITY"
    error = _sample_conflict(lambda: ledger.require_reservation(record, producer))
    assert isinstance(error.__cause__, json.JSONDecodeError)
    with pytest.raises(ValueError, match="consumer_id"):
        ledger.require_reservation(replace(record, consumer_id="wrong"), producer, before_entry_ref=absent_bound)
    (tmp_path / "registries" / (SAMPLE_LOG + ".jsonl")).write_bytes(b"broken-prefix")
    with pytest.raises(FoundationFailure) as raised:
        ledger.require_reservation(record, producer)
    assert raised.value.code == "LOG_INTEGRITY"


def test_reads_do_not_publish_or_refresh_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    foundation, ledger, producer, record, original = _reserved(tmp_path / "retained")
    foundation.checkpoint(SAMPLE_LOG)
    bound = foundation.append(BOUND_LOG, "bound", b"bound").entry_ref
    before = _retained_files(tmp_path)
    clock = iter(())
    reopened = SampleConsumptionLedger(LocalFoundation(tmp_path / "retained", clock=lambda: next(clock)))
    missing = SampleConsumptionLedger(LocalFoundation(tmp_path / "missing", clock=_forbidden))
    for name in ("put", "append", "checkpoint"):
        monkeypatch.setattr(LocalFoundation, name, _forbidden)
    monkeypatch.setattr(SampleConsumptionLedger, "reserve", _forbidden)
    assert reopened.require_reservation(record, producer, before_entry_ref=bound) == original
    _sample_conflict(lambda: missing.require_reservation(record, producer))
    assert _retained_files(tmp_path) == before
    assert not list((tmp_path / "missing" / "registries").iterdir())
    assert not list(tmp_path.rglob(".foundation.write.lock"))
    # An original-only read may not reconstruct evidence once its owner log is lost.
    (tmp_path / "retained" / "registries" / (SAMPLE_LOG + ".jsonl")).unlink()
    damaged = _retained_files(tmp_path)
    with pytest.raises(FoundationFailure) as raised:
        reopened.require_reservation(record, producer)
    assert raised.value.code == "LOG_INTEGRITY"
    assert _retained_files(tmp_path) == damaged
