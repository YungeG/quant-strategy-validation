"""Public synthetic input-read protocol; no source, reader, or financial authority."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from crypto_quant_domain import ArtifactEnvelope, ArtifactRef, canonical_bytes, canonical_sha256
from crypto_quant_foundation import FoundationFailure, LocalFoundation, LogCheckpoint, LogEntryRef
from crypto_quant_research import (
    InputQualificationDeclarationV1, InputQualificationReadTaskV1,
    InputQualificationCaptureBindingV1, publish_input_qualification_declaration,
    publish_input_qualification_read_task, publish_input_qualification_capture_binding,
)
from crypto_quant_validation import (
    InputQualificationError, InputQualificationReadBindingV1, InputQualificationVerifierV1,
    SampleConsumptionLedger,
)

AT = "2026-02-01T00:00:00.000000Z"
START = "2000-01-01T00:00:00.000000Z"
END = "2000-02-01T00:00:00.000000Z"
H = canonical_sha256("synthetic-input-qualification")
OTHER = canonical_sha256("different-synthetic-identity")
LOG = "research.artifacts.v1"
SAMPLE = "validation.sample-consumption.v1"


def raw_publication(f, envelope, *, event=None, log=LOG):
    ref = f.put(envelope=envelope)
    if event is None:
        if envelope.artifact_type == "input_qualification_read_task":
            p = envelope.payload
            event = canonical_sha256(("input-qualification-read-task-v1", p["declaration_ref"], p["phase"], p["scope_key"]))
        elif envelope.artifact_type == "input_qualification_capture_binding":
            revision = "local-capture:" + canonical_sha256(("input-qualification-capture-v1", envelope.payload["declaration_ref"]))
            event = canonical_sha256(("input-qualification-capture-binding-v1", revision))
        else:
            event = canonical_sha256(("artifact-publication-v1", log, ref))
    receipt = f.append(log, event, canonical_bytes(envelope))
    return ref, receipt.entry_ref, LogCheckpoint(log, receipt.accepted_at, receipt.log_sequence, receipt.receipt_hash)


def new_case(tmp_path, *, scopes=None, foundation=None, ledger=None):
    f = foundation or LocalFoundation(tmp_path, clock=lambda: AT)
    ledger = ledger or SampleConsumptionLedger(f)
    verifier = InputQualificationVerifierV1(f, ledger)
    declaration = InputQualificationDeclarationV1("synthetic", H, H, H, H, H, scopes or (("prices", START, END),))
    dp = publish_input_qualification_declaration(declaration, f)
    revision = "local-capture:" + canonical_sha256(("input-qualification-capture-v1", dp.ref))
    return SimpleNamespace(f=f, ledger=ledger, verifier=verifier, declaration=declaration, dp=dp, revision=revision)


def task(c, phase="QUALIFY", scope="prices"):
    return InputQualificationReadTaskV1(c.dp.ref, scope, phase, None if phase == "QUALIFY" else H)


def actual(c, phase="QUALIFY", scope="prices"):
    _, start, end = next(item for item in c.declaration.source_intervals if item[0] == scope)
    return InputQualificationReadBindingV1(c.revision, scope, phase, H, H, H, start, end)


def register(c, phase="QUALIFY", scope="prices"):
    t = task(c, phase, scope)
    p = publish_input_qualification_read_task(t, c.f, c.verifier)
    r = c.verifier.reserve(c.dp.publication, p.publication, AT, actual(c, phase, scope))
    return t, p, r


def capture(c):
    proofs = []
    for scope, _, _ in c.declaration.source_intervals:
        register(c, scope=scope)
        _, p, r = register(c, "CAPTURE", scope)
        proofs.append((scope, p.ref, r.reservation_entry_ref))
    manifest = {"type": "market_bundle_manifest", "bundle_key": "synthetic", "schema_version": 1,
        "coverage_start": START, "coverage_end_exclusive": END, "instrument_catalog_hash": H,
        "capabilities": (), "streams": (), "content_hash": H}
    ref = {"type": "market_bundle_ref", "bundle_key": "synthetic", "manifest_hash": canonical_sha256(manifest)}
    return InputQualificationCaptureBindingV1(c.dp.ref, tuple(proofs), H, ref, manifest)


def test_T1_exact_foundation_identity_and_nominal_types_before_any_io(tmp_path, monkeypatch):
    f = LocalFoundation(tmp_path, clock=Mock(side_effect=AssertionError("clock")))
    other = LocalFoundation(tmp_path, clock=Mock(side_effect=AssertionError("clock")))
    ledger = SampleConsumptionLedger(f)
    for name in ("put", "read", "append", "entries", "checkpoint"):
        monkeypatch.setattr(LocalFoundation, name, Mock(side_effect=AssertionError("I/O")))
    for foundation, supplied_ledger in ((other, ledger), (object(), ledger), (f, object())):
        with pytest.raises(InputQualificationError, match="FOUNDATION_MISMATCH"):
            InputQualificationVerifierV1(foundation, supplied_ledger)
    verifier = InputQualificationVerifierV1(f, ledger)
    with pytest.raises(InputQualificationError, match="FOUNDATION_MISMATCH"):
        verifier.require_foundation(other)
    with pytest.raises(InputQualificationError, match="FOUNDATION_MISMATCH"):
        publish_input_qualification_read_task(object(), other, verifier)
    with pytest.raises(InputQualificationError, match="FOUNDATION_MISMATCH"):
        publish_input_qualification_capture_binding(object(), f, object())
    with pytest.raises(TypeError):
        publish_input_qualification_declaration(object(), object())
    verifier._ledger = SampleConsumptionLedger(other)
    for call in (lambda: verifier.reserve(None, None, None, None),
                 lambda: verifier.verify_read_task_inputs(None), lambda: verifier.verify_capture_inputs(None),
                 lambda: verifier.verify_capture(None), lambda: verifier.require_trial_binding(None, None)):
        with pytest.raises(InputQualificationError, match="FOUNDATION_MISMATCH"):
            call()
    assert not tmp_path.exists() or not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("mutation", ["orphan", "log", "checkpoint", "entry", "bytes", "ref"])
def test_T2_original_owner_publication_not_just_CAS(tmp_path, mutation):
    c = new_case(tmp_path)
    t = task(c)
    envelope = ArtifactEnvelope.create("input_qualification_read_task", 1, t.payload)
    ref = c.f.put(envelope=envelope)
    if mutation == "orphan":
        pub = (ref, c.dp.entry_ref, c.dp.checkpoint)
    elif mutation == "log":
        pub = raw_publication(c.f, envelope, log="wrong.owner.v1")
    elif mutation == "bytes":
        event = canonical_sha256(("input-qualification-read-task-v1", c.dp.ref, "QUALIFY", "prices"))
        receipt = c.f.append(LOG, event, canonical_bytes(ArtifactEnvelope.create("unrelated", 1, {})))
        pub = (ref, receipt.entry_ref, LogCheckpoint(LOG, receipt.accepted_at, receipt.log_sequence, receipt.receipt_hash))
    else:
        pub = publish_input_qualification_read_task(t, c.f, c.verifier).publication
        if mutation == "checkpoint":
            pub = (pub[0], pub[1], c.dp.checkpoint)
        elif mutation == "entry":
            pub = (pub[0], c.dp.entry_ref, pub[2])
        else:
            pub = (ArtifactRef("input_qualification_read_task", 1, H), pub[1], pub[2])
            # Addressable but different original bytes/ref, not a missing CAS failure.
            other = task(c, scope="other")
            other_ref = c.f.put(envelope=ArtifactEnvelope.create("input_qualification_read_task", 1, other.payload))
            pub = (other_ref, pub[1], pub[2])
    with pytest.raises(InputQualificationError, match="PUBLICATION_INVALID"):
        c.verifier.reserve(c.dp.publication, pub, AT, actual(c))
    assert c.f.entries(SAMPLE) == ()


@pytest.mark.parametrize("field,value", [
    ("capture_revision", "local-capture:" + OTHER), ("scope_key", "other"), ("phase", "CAPTURE"),
    ("source_scope_hash", OTHER), ("reader_code_hash", OTHER), ("normalization_code_hash", OTHER),
    ("interval_start", "2000-01-02T00:00:00.000000Z"), ("interval_end", "2000-02-02T00:00:00.000000Z"),
])
def test_T3_actual_reader_configuration_exact_match(tmp_path, field, value):
    c = new_case(tmp_path)
    p = publish_input_qualification_read_task(task(c), c.f, c.verifier)
    before = c.f.entries(LOG)
    with pytest.raises(InputQualificationError, match="SCOPE_CONFLICT"):
        c.verifier.reserve(c.dp.publication, p.publication, AT, replace(actual(c), **{field: value}))
    assert c.f.entries(SAMPLE) == () and before == c.f.entries(LOG)


def test_T4_capture_cannot_be_published_before_qualify_actual_reservation(tmp_path, monkeypatch):
    c = new_case(tmp_path)
    publish_input_qualification_read_task(task(c), c.f, c.verifier)
    put = Mock(wraps=c.f.put)
    monkeypatch.setattr(c.f, "put", put)
    with pytest.raises(InputQualificationError, match="PREREQUISITE_INVALID"):
        publish_input_qualification_read_task(task(c, "CAPTURE"), c.f, c.verifier)
    assert put.call_count == 0 and c.f.entries(SAMPLE) == ()


def test_T4_late_parent_append_does_not_repair_capture_publication_order(tmp_path):
    c = new_case(tmp_path)
    q = publish_input_qualification_read_task(task(c), c.f, c.verifier)
    cap = task(c, "CAPTURE")
    cp = raw_publication(c.f, ArtifactEnvelope.create("input_qualification_read_task", 1, cap.payload))
    c.verifier.reserve(c.dp.publication, q.publication, AT, actual(c))
    with pytest.raises(InputQualificationError, match="PREREQUISITE_INVALID"):
        c.verifier.reserve(c.dp.publication, cp, AT, actual(c, "CAPTURE"))
    with pytest.raises(InputQualificationError, match="PREREQUISITE_INVALID"):
        publish_input_qualification_read_task(cap, c.f, c.verifier)
    assert len(c.f.entries(SAMPLE)) == 1


def test_T9_failed_and_zero_row_reads_keep_original_receipt_and_retry_time(tmp_path):
    now = [AT]
    c = new_case(tmp_path, foundation=LocalFoundation(tmp_path, clock=lambda: now[0]))
    _, p, original = register(c)
    before = c.f.entries(SAMPLE)
    # Physical read outcome is deliberately not a ledger operation, nor is it erased.
    outcomes = []
    for physical_outcome in ("failed", "zero-row"):
        replay = c.verifier.reserve(c.dp.publication, p.publication, AT, actual(c))
        if physical_outcome == "failed":
            with pytest.raises(RuntimeError, match="synthetic reader failure"):
                outcomes.append("failed")
                raise RuntimeError("synthetic reader failure")
        else:
            rows = ()
            outcomes.append(("zero-row", len(rows)))
        assert replay == original and c.f.entries(SAMPLE) == before
    assert outcomes == ["failed", ("zero-row", 0)]
    now[0] = "2026-02-02T00:00:00.000000Z"
    # A refreshed time is in the current clock window but still conflicts with the original record.
    with pytest.raises(InputQualificationError, match="RESERVATION_INVALID"):
        c.verifier.reserve(c.dp.publication, p.publication, "2026-02-01T00:00:00.000001Z", actual(c))
    with pytest.raises(InputQualificationError, match="RESERVATION_INVALID"):
        c.verifier.reserve(c.dp.publication, p.publication, START, actual(c))
    assert c.f.entries(SAMPLE) == before


def test_T9_foundation_append_and_storage_failures_propagate_unmodified(tmp_path, monkeypatch):
    c = new_case(tmp_path)
    p = publish_input_qualification_read_task(task(c), c.f, c.verifier)
    failure = FoundationFailure("LOG_PUBLICATION_FAILED")
    monkeypatch.setattr(c.f, "append", Mock(side_effect=failure))
    with pytest.raises(FoundationFailure) as error:
        c.verifier.reserve(c.dp.publication, p.publication, AT, actual(c))
    assert error.value is failure
    monkeypatch.setattr(c.f, "entries", Mock(side_effect=FoundationFailure("LOG_INTEGRITY")))
    with pytest.raises(FoundationFailure, match="LOG_INTEGRITY"):
        c.verifier.reserve(c.dp.publication, p.publication, AT, actual(c))


@pytest.mark.parametrize("corruption", ["field", "version", "float", "size", "bool", "tuple"])
def test_T1_validation_decodes_its_own_strict_wire_before_storage(tmp_path, monkeypatch, corruption):
    c = new_case(tmp_path)
    payload = dict(task(c).payload)
    version = 1
    if corruption == "field":
        payload["authorized"] = True
    elif corruption == "version":
        version = 2
    elif corruption == "float":
        payload["input_check_hash"] = 1.5
    elif corruption == "size":
        payload["scope_key"] = "a" * (2 * 1024 * 1024)
    elif corruption == "bool":
        payload["phase"] = True
    else:
        payload["declaration_ref"] = ("not", "a", "ref")
    # Domain itself refuses floats; the public verifier rejects other wire errors without I/O.
    if corruption == "float":
        with pytest.raises(ValueError):
            ArtifactEnvelope.create("input_qualification_read_task", version, payload)
        return
    envelope = ArtifactEnvelope.create("input_qualification_read_task", version, payload)
    for name in ("read", "put", "entries", "checkpoint", "append"):
        monkeypatch.setattr(c.f, name, Mock(side_effect=AssertionError("I/O")))
    with pytest.raises((TypeError, ValueError)):
        c.verifier.verify_read_task_inputs(envelope)


def test_T2_capture_binding_CAS_orphan_is_not_an_owner_attestation(tmp_path):
    c = new_case(tmp_path)
    b = capture(c)
    ref = c.f.put(envelope=ArtifactEnvelope.create("input_qualification_capture_binding", 1, b.payload))
    with pytest.raises(InputQualificationError, match="PUBLICATION_INVALID"):
        c.verifier.verify_capture((ref, c.dp.entry_ref, c.dp.checkpoint))


def test_T9_partial_publication_failure_leaves_only_orphan_then_exact_retry(tmp_path, monkeypatch):
    c = new_case(tmp_path)
    t = task(c)
    original_append = c.f.append
    failure = FoundationFailure("LOG_PUBLICATION_FAILED")
    monkeypatch.setattr(c.f, "append", Mock(side_effect=failure))
    with pytest.raises(FoundationFailure) as error:
        publish_input_qualification_read_task(t, c.f, c.verifier)
    assert error.value is failure
    assert len(c.f.entries(LOG)) == 1 and c.f.entries(SAMPLE) == ()
    orphan = c.f.put(envelope=ArtifactEnvelope.create("input_qualification_read_task", 1, t.payload))
    with pytest.raises(InputQualificationError, match="PUBLICATION_INVALID"):
        c.verifier.reserve(c.dp.publication, (orphan, c.dp.entry_ref, c.dp.checkpoint), AT, actual(c))
    monkeypatch.setattr(c.f, "append", original_append)
    _, p, receipt = register(c)
    assert p.ref == orphan
    assert c.verifier.reserve(c.dp.publication, p.publication, AT, actual(c)) == receipt


def test_T1_validation_output_values_have_only_the_frozen_fields():
    from dataclasses import fields, MISSING
    from crypto_quant_validation import InputQualificationReservationReceiptV1, InputQualificationCaptureViewV1
    for cls, expected in (
        (InputQualificationReservationReceiptV1, ("declaration_ref", "read_task_ref", "declaration_publication_entry_ref", "producer_publication_entry_ref", "phase", "scope_key", "record", "reservation_entry_ref")),
        (InputQualificationCaptureViewV1, ("binding_ref", "declaration_ref", "capture_revision", "market_bundle_ref", "source_manifest_hash", "source_intervals")),
    ):
        assert tuple(field.name for field in fields(cls)) == expected
        assert all(field.default is MISSING and field.default_factory is MISSING for field in fields(cls))


def test_T9_real_foundation_clock_failure_is_not_translated_to_qualification_failure(tmp_path):
    now = [AT]
    c = new_case(tmp_path, foundation=LocalFoundation(tmp_path, clock=lambda: now[0]))
    p = publish_input_qualification_read_task(task(c), c.f, c.verifier)
    now[0] = START
    with pytest.raises(FoundationFailure, match="CLOCK_NOT_MONOTONIC"):
        c.verifier.reserve(c.dp.publication, p.publication, AT, actual(c))
    assert c.f.entries(SAMPLE) == ()


def test_T9_ledger_returning_another_original_append_is_not_accepted(tmp_path, monkeypatch):
    from crypto_quant_validation import SampleConsumptionRecord
    c = new_case(tmp_path)
    p = publish_input_qualification_read_task(task(c), c.f, c.verifier)
    other = ArtifactRef("trial_declaration", 1, OTHER)
    other_record = SampleConsumptionRecord("other", START, END, "discovery", canonical_sha256(("sample-consumer-v1", other)), AT)
    other_entry = c.ledger.reserve(other_record, other)
    monkeypatch.setattr(c.ledger, "reserve", lambda *args: other_entry)
    with pytest.raises(InputQualificationError, match="RESERVATION_INVALID"):
        c.verifier.reserve(c.dp.publication, p.publication, AT, actual(c))
    assert len(c.f.entries(SAMPLE)) == 1
