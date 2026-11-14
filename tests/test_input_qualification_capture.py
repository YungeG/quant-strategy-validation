"""Exact capture cover, immutable slots, and strict shared-log compatibility."""
from dataclasses import replace

import pytest
from crypto_quant_domain import ArtifactEnvelope, ArtifactRef, canonical_bytes, canonical_sha256
from crypto_quant_foundation import FoundationFailure, LogCheckpoint
from crypto_quant_research import (
    DataSlice, StatisticalStudyDeclaration, StatisticalReadTask,
    RetrospectiveStudyProtocol, RetrospectiveReadTask,
    publish_statistical_study, publish_statistical_read_task,
    publish_retrospective_protocol, publish_retrospective_read_task,
    publish_input_qualification_read_task, publish_input_qualification_capture_binding,
)
from crypto_quant_validation import (
    Holdout, InputQualificationError, SampleConsumptionRecord, StatisticalReservationError,
    RetrospectiveReservationError, ValidationCoreFailure, reserve_statistical_read, reserve_retrospective_read,
)
from .test_input_qualification_reservation import (
    AT, START, END, H, OTHER, LOG, SAMPLE, actual, new_case, task, register, capture, raw_publication,
)


def test_T4_binding_requires_all_original_capture_proofs_and_exact_scope_cover(tmp_path):
    c = new_case(tmp_path, scopes=(("a", START, END), ("b", START, END)))
    b = capture(c)
    extra = (*b.capture_proofs, ("c", b.capture_proofs[0][1], b.capture_proofs[0][2]))
    for proofs in ((), b.capture_proofs[:1], extra, (("a", b.capture_proofs[1][1], b.capture_proofs[1][2]), b.capture_proofs[1])):
        with pytest.raises(InputQualificationError, match="PREREQUISITE_INVALID"):
            publish_input_qualification_capture_binding(replace(b, capture_proofs=proofs), c.f, c.verifier)
    q_entry = c.f.entries(SAMPLE)[0].entry_ref
    proofs = (("a", b.capture_proofs[0][1], q_entry), b.capture_proofs[1])
    with pytest.raises(InputQualificationError, match="PREREQUISITE_INVALID"):
        publish_input_qualification_capture_binding(replace(b, capture_proofs=proofs), c.f, c.verifier)
    with pytest.raises(ValueError):
        replace(b, capture_proofs=(b.capture_proofs[0], b.capture_proofs[0]))
    with pytest.raises(ValueError):
        replace(b, capture_proofs=tuple(reversed(b.capture_proofs)))


def test_T4_binding_published_before_reservation_cannot_be_repaired_later(tmp_path):
    c = new_case(tmp_path)
    register(c)
    t = task(c, "CAPTURE")
    p = publish_input_qualification_read_task(t, c.f, c.verifier)
    # A raw early binding uses a real, but wrong original Q entry as its proof.
    manifest = {"type": "market_bundle_manifest", "bundle_key": "synthetic", "schema_version": 1,
        "coverage_start": START, "coverage_end_exclusive": END, "instrument_catalog_hash": H,
        "capabilities": (), "streams": (), "content_hash": H}
    ref = {"type": "market_bundle_ref", "bundle_key": "synthetic", "manifest_hash": canonical_sha256(manifest)}
    from crypto_quant_research import InputQualificationCaptureBindingV1
    b = InputQualificationCaptureBindingV1(c.dp.ref, (("prices", p.ref, c.f.entries(SAMPLE)[0].entry_ref),), H, ref, manifest)
    pub = raw_publication(c.f, ArtifactEnvelope.create("input_qualification_capture_binding", 1, b.payload))
    c.verifier.reserve(c.dp.publication, p.publication, AT, actual(c, "CAPTURE"))
    with pytest.raises(InputQualificationError, match="PREREQUISITE_INVALID"):
        c.verifier.verify_capture(pub)
    with pytest.raises(InputQualificationError, match="PREREQUISITE_INVALID"):
        publish_input_qualification_capture_binding(b, c.f, c.verifier)


@pytest.mark.parametrize("change", ["source", "bundle", "manifest"])
def test_T5_one_C_one_immutable_capture_binding(tmp_path, change):
    c = new_case(tmp_path)
    b = capture(c)
    pub = publish_input_qualification_capture_binding(b, c.f, c.verifier)
    before = c.f.entries(LOG)
    if change == "source":
        different = replace(b, source_manifest_hash=OTHER)
    else:
        manifest = {**dict(b.market_bundle_manifest), ("bundle_key" if change == "bundle" else "content_hash"): ("another" if change == "bundle" else OTHER)}
        ref = {**dict(b.market_bundle_ref), "bundle_key": manifest["bundle_key"], "manifest_hash": canonical_sha256(manifest)}
        different = replace(b, market_bundle_ref=ref, market_bundle_manifest=manifest)
    with pytest.raises(InputQualificationError, match="CAPTURE_CONFLICT"):
        publish_input_qualification_capture_binding(different, c.f, c.verifier)
    assert before == c.f.entries(LOG)
    assert publish_input_qualification_capture_binding(b, c.f, c.verifier) == pub


def test_T5_phase_slot_cannot_be_reopened_by_changing_check_hash(tmp_path):
    c = new_case(tmp_path)
    register(c)
    t, p, r = register(c, "CAPTURE")
    with pytest.raises(InputQualificationError, match="CAPTURE_CONFLICT"):
        publish_input_qualification_read_task(replace(t, input_check_hash=OTHER), c.f, c.verifier)
    assert publish_input_qualification_read_task(t, c.f, c.verifier) == p


@pytest.mark.parametrize("kind", ["task", "binding"])
def test_T5_controlled_append_race_preserves_foundation_LOG_CONFLICT(tmp_path, monkeypatch, kind):
    c = new_case(tmp_path)
    if kind == "task":
        register(c)
        first = task(c, "CAPTURE")
        competitor = replace(first, input_check_hash=OTHER)
        name = "input_qualification_read_task"
        publish = publish_input_qualification_read_task
    else:
        first = capture(c)
        competitor = replace(first, source_manifest_hash=OTHER)
        name = "input_qualification_capture_binding"
        publish = publish_input_qualification_capture_binding
    original_append = c.f.append
    winner = []
    def append(log, event, payload):
        if not winner:
            envelope = ArtifactEnvelope.create(name, 1, competitor.payload)
            c.f.put(envelope=envelope)
            winner.append(original_append(log, event, canonical_bytes(envelope)))
        return original_append(log, event, payload)
    monkeypatch.setattr(c.f, "append", append)
    with pytest.raises(FoundationFailure, match="LOG_CONFLICT"):
        publish(first, c.f, c.verifier)
    monkeypatch.setattr(c.f, "append", original_append)
    assert len(winner) == 1
    published = publish(competitor, c.f, c.verifier)
    assert published.entry_ref == winner[0].entry_ref
    if kind == "binding":
        assert c.verifier.verify_capture(published.publication).source_manifest_hash == OTHER


@pytest.mark.parametrize("change", ["content_hash", "key", "nested_wire"])
def test_T6_whole_native_manifest_hash_not_content_hash(tmp_path, change):
    c = new_case(tmp_path)
    b = capture(c)
    if change == "content_hash":
        ref = {**dict(b.market_bundle_ref), "manifest_hash": b.market_bundle_manifest["content_hash"]}
        different = replace(b, market_bundle_ref=ref)
    elif change == "key":
        different = replace(b, market_bundle_ref={**dict(b.market_bundle_ref), "bundle_key": "another"})
    else:
        different = replace(b, market_bundle_manifest={**dict(b.market_bundle_manifest), "streams": ({"opaque": "changed"},)})
    with pytest.raises(InputQualificationError, match="CAPTURE_CONFLICT"):
        publish_input_qualification_capture_binding(different, c.f, c.verifier)
    assert canonical_sha256(b.market_bundle_manifest) != b.market_bundle_manifest["content_hash"]
    p = publish_input_qualification_capture_binding(b, c.f, c.verifier)
    assert c.verifier.verify_capture(p.publication).market_bundle_ref == b.market_bundle_ref


def statistical(c, *, overlap=False):
    source = ArtifactRef("backtest_market_bundle", 1, H).to_canonical_dict()
    partitions = []
    for role, month in (("TRAIN", 1), ("DEVELOPMENT", 3), ("HELD_OUT", 5)):
        start, end = (START, END) if overlap and role == "HELD_OUT" else (f"1999-{month:02d}-01T00:00:00.000000Z", f"1999-{month+1:02d}-01T00:00:00.000000Z")
        data = DataSlice(source, c.revision if overlap else "unrelated-statistical", start, end)
        partitions.append((role, data, start, end))
    s = StatisticalStudyDeclaration("mixed-statistical", H, H, H, H, H,
        ("CLIMATOLOGY", "PRICE_ONLY", "PRICE_VOLUME", "SHUFFLED_VOLUME"), tuple(partitions))
    sp = publish_statistical_study(s, c.f)
    pubs = {}
    for phase, index, feature_hash, model_hash in (("TRAIN_FEATURE", 0, None, None), ("TRAIN_FIT", 0, H, None), ("HELD_OUT_FEATURE", 2, None, H)):
        t = StatisticalReadTask(sp.ref, phase, s.partitions[index][1], feature_hash, model_hash, None)
        pubs[phase] = publish_statistical_read_task(t, c.f)
    return s, sp, pubs


def retrospective(c):
    p = RetrospectiveStudyProtocol("mixed-retrospective", H, H, H, H, H, (("capture", START, END),))
    pp = publish_retrospective_protocol(p, c.f)
    t = RetrospectiveReadTask(pp.ref, "INPUT_QUALIFICATION", p.source_slices[0])
    tp = publish_retrospective_read_task(t, c.f)
    return pp, tp


def heldout_evidence(c, study):
    snap = c.ledger.freeze_snapshot()
    data = study.partitions[2][1]
    assessment = c.ledger.assess_holdout(snap, Holdout(dict(data.market_bundle_ref), data.dataset_revision,
        data.interval_start, data.interval_end, "HOLDOUT", False))
    def proof(ref):
        entry = next(e for e in c.f.entries("validation.artifacts.v1") if e.event_id == canonical_sha256(("artifact-publication-v1", "validation.artifacts.v1", ref)))
        return ref, entry.entry_ref, LogCheckpoint("validation.artifacts.v1", entry.accepted_at, entry.log_sequence, entry.receipt_hash)
    return proof(snap), proof(assessment)


def test_T11_new_actual_reservations_then_genuine_statistical_and_retrospective(tmp_path):
    c = new_case(tmp_path)
    register(c)
    study, sp, tasks = statistical(c)
    first = reserve_statistical_read(c.f, c.ledger, sp.publication, tasks["TRAIN_FEATURE"].publication, AT)
    assert reserve_statistical_read(c.f, c.ledger, sp.publication, tasks["TRAIN_FEATURE"].publication, AT) == first
    reserve_statistical_read(c.f, c.ledger, sp.publication, tasks["TRAIN_FIT"].publication, AT)
    reserve_statistical_read(c.f, c.ledger, sp.publication, tasks["HELD_OUT_FEATURE"].publication, AT, heldout_evidence(c, study))
    pp, tp = retrospective(c)
    retro = reserve_retrospective_read(c.f, c.ledger, pp.publication, tp.publication, AT)
    assert reserve_retrospective_read(c.f, c.ledger, pp.publication, tp.publication, AT) == retro
    assert len(c.f.entries(SAMPLE)) == 5
    # The ordinary ledger projection recognizes the same exact pair too.
    snapshot = c.ledger.freeze_snapshot()
    holdout = Holdout(ArtifactRef("backtest_market_bundle", 1, H).to_canonical_dict(), "unrelated", START, END, "HOLDOUT", False)
    assessment = c.ledger.assess_holdout(snapshot, holdout)
    assert c.f.read(ref=assessment).envelope.payload["untouched"] is True


def test_T11_same_C_overlap_still_contaminates_heldout(tmp_path):
    c = new_case(tmp_path)
    register(c)
    s, sp, tasks = statistical(c, overlap=True)
    for phase in ("TRAIN_FEATURE", "TRAIN_FIT"):
        reserve_statistical_read(c.f, c.ledger, sp.publication, tasks[phase].publication, AT)
    snap = c.ledger.freeze_snapshot()
    data = s.partitions[2][1]
    assessment = c.ledger.assess_holdout(snap, Holdout(dict(data.market_bundle_ref), data.dataset_revision, START, END, "HOLDOUT", False))
    def proof(ref):
        entry = next(e for e in c.f.entries("validation.artifacts.v1") if e.event_id == canonical_sha256(("artifact-publication-v1", "validation.artifacts.v1", ref)))
        return ref, entry.entry_ref, LogCheckpoint("validation.artifacts.v1", entry.accepted_at, entry.log_sequence, entry.receipt_hash)
    assert c.f.read(ref=assessment).envelope.payload["untouched"] is False
    with pytest.raises(StatisticalReservationError, match="HOLDOUT_EVIDENCE_REQUIRED"):
        reserve_statistical_read(c.f, c.ledger, sp.publication, tasks["HELD_OUT_FEATURE"].publication, AT, (proof(snap), proof(assessment)))


@pytest.mark.parametrize("corruption", ["purpose", "consumer", "schema", "unknown", "clock"])
def test_T11_shared_parsers_never_skip_invalid_new_or_unknown_entries(tmp_path, corruption):
    c = new_case(tmp_path)
    register(c)
    study, sp, tasks = statistical(c)
    evidence = heldout_evidence(c, study)
    pp, tp = retrospective(c)
    producer = ArtifactRef("unknown_read_task" if corruption == "unknown" else "input_qualification_read_task", 2 if corruption == "schema" else 1, OTHER)
    record = SampleConsumptionRecord("unrelated", START, END, "feature_build" if corruption == "purpose" else "discovery",
        OTHER if corruption == "consumer" else canonical_sha256(("sample-consumer-v1", producer)),
        "2026-02-01T00:00:00.000001Z" if corruption == "clock" else AT)
    envelope = ArtifactEnvelope.create("sample_consumption_append", 1, {"record": {name: getattr(record, name) for name in ("dataset_revision", "interval_start", "interval_end", "purpose", "consumer_id", "consumed_at")}, "producer_ref": producer})
    event = canonical_sha256(("sample-consumption-append-v1", producer, record.dataset_revision, START, END, record.purpose))
    c.f.append(SAMPLE, event, canonical_bytes(envelope))
    with pytest.raises(StatisticalReservationError):
        reserve_statistical_read(c.f, c.ledger, sp.publication, tasks["HELD_OUT_FEATURE"].publication, AT, evidence)
    with pytest.raises(RetrospectiveReservationError):
        reserve_retrospective_read(c.f, c.ledger, pp.publication, tp.publication, AT)
    snap = c.ledger.freeze_snapshot()
    with pytest.raises(ValidationCoreFailure, match="SAMPLE_LEDGER_CONFLICT"):
        c.ledger.assess_holdout(snap, Holdout(ArtifactRef("backtest_market_bundle", 1, H).to_canonical_dict(), "unrelated", START, END, "HOLDOUT", False))
