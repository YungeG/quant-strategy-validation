"""Public-path synthetic governance; no real source, source qualification or Reader."""

from __future__ import annotations

from pathlib import Path

import pytest
from crypto_quant_domain import ArtifactEnvelope, canonical_bytes, canonical_sha256
from crypto_quant_foundation import LocalFoundation, LogCheckpoint
from crypto_quant_research import (
    RetrospectiveReadTask,
    RetrospectiveSourceSlice,
    RetrospectiveStudyDeclaration,
    RetrospectiveStudyProtocol,
    publish_retrospective_protocol,
    publish_retrospective_read_task,
    publish_retrospective_study,
)
from crypto_quant_validation import (
    RetrospectiveReservationError,
    SampleConsumptionLedger,
    reserve_retrospective_read,
)

AT = "2026-02-01T00:00:00.000000Z"
H = tuple("sha256:" + c * 64 for c in "123456789abcdef")
LOG = "research.artifacts.v1"
SAMPLE = "validation.sample-consumption.v1"


def case(root: Path):
    f = LocalFoundation(root, clock=lambda: AT)
    p = RetrospectiveStudyProtocol(
        "SYNTHETIC-ONLY",
        H[0],
        H[1],
        H[2],
        H[3],
        H[4],
        (("capture", "2000-01-01T00:00:00.000000Z", "2000-07-01T00:00:00.000000Z"),),
    )
    pp = publish_retrospective_protocol(p, f)
    return f, SampleConsumptionLedger(f), p, pp


def register(f, ledger, p, pp, phase, source, sp=None, **hashes):
    t = RetrospectiveReadTask(pp.ref if sp is None else sp.ref, phase, source, **hashes)
    tp = publish_retrospective_read_task(t, f)
    receipt = reserve_retrospective_read(
        f,
        ledger,
        pp.publication,
        tp.publication,
        AT,
        None if sp is None else sp.publication,
    )
    return t, tp, receipt


def full(root):
    f, ledger, p, pp = case(root)
    tasks = {}
    for phase, hashes in (
        ("INPUT_QUALIFICATION", {}),
        ("SNAPSHOT_FREEZE", {"input_check_hash": H[9]}),
    ):
        tasks[phase] = register(f, ledger, p, pp, phase, p.source_slices[0], **hashes)
    parts = tuple(
        (
            role,
            RetrospectiveSourceSlice(
                "capture",
                p.capture_revision,
                p.source_scope_hash,
                f"2000-{m:02d}-01T00:00:00.000000Z",
                f"2000-{m + 1:02d}-01T00:00:00.000000Z",
                H[5],
            ),
            f"2000-{m:02d}-10T00:00:00.000000Z",
            f"2000-{m:02d}-20T00:00:00.000000Z",
        )
        for role, m in (("TRAIN", 1), ("DEVELOPMENT", 3), ("RETROSPECTIVE_EVAL", 5))
    )
    s = RetrospectiveStudyDeclaration(pp.ref, H[5], H[6], H[7], H[8], parts)
    sp = publish_retrospective_study(s, f)
    for phase, index, hashes in (
        ("TRAIN_FEATURE", 0, {}),
        ("TRAIN_FIT", 0, {"feature_rows_hash": H[10]}),
        ("DEVELOPMENT_FEATURE", 1, {"model_cards_hash": H[11]}),
        ("RETROSPECTIVE_EVAL_FEATURE", 2, {"model_cards_hash": H[11]}),
        (
            "DEVELOPMENT_OUTCOME",
            1,
            {"model_cards_hash": H[11], "predictions_hash": H[12]},
        ),
        (
            "RETROSPECTIVE_EVAL_OUTCOME",
            2,
            {"model_cards_hash": H[11], "predictions_hash": H[12]},
        ),
    ):
        tasks[phase] = register(
            f, ledger, p, pp, phase, s.partitions[index][1], sp, **hashes
        )
    return f, ledger, p, pp, s, sp, tasks


def test_pre_freeze_public_append_does_not_need_final_study(tmp_path):
    f, ledger, p, pp = case(tmp_path)
    _t, tp, r = register(f, ledger, p, pp, "INPUT_QUALIFICATION", p.source_slices[0])
    assert (
        r.study_ref is None and r.protocol_ref == pp.ref and r.read_task_ref == tp.ref
    )
    assert r.record.dataset_revision == p.capture_revision
    assert r.purpose == "discovery" and r.untouched_claim is r.causal_pit_claim is False
    assert (
        len(f.entries(SAMPLE)) == 1
        and r.reservation_entry_ref == f.entries(SAMPLE)[0].entry_ref
    )
    assert (
        reserve_retrospective_read(f, ledger, pp.publication, tp.publication, AT) == r
    )
    assert len(f.entries(SAMPLE)) == 1


def test_full_fixed_public_flow_and_reentry(tmp_path):
    f, ledger, _p, pp, _s, sp, tasks = full(tmp_path)
    assert len(f.entries(SAMPLE)) == 8
    assert {item[1].ref.artifact_type for item in tasks.values()} == {
        "retrospective_input_read_task",
        "retrospective_feature_read_task",
        "retrospective_training_read_task",
        "retrospective_outcome_read_task",
    }
    _t, tp, r = tasks["RETROSPECTIVE_EVAL_OUTCOME"]
    assert r.record.purpose == "validation" and r.study_ref == sp.ref
    assert (
        reserve_retrospective_read(
            f, ledger, pp.publication, tp.publication, AT, sp.publication
        )
        == r
    )
    assert len(f.entries(SAMPLE)) == 8
    with pytest.raises(RetrospectiveReservationError, match="RESERVATION_FAILED"):
        reserve_retrospective_read(
            f,
            ledger,
            pp.publication,
            tp.publication,
            "2026-02-01T00:00:00.000001Z",
            sp.publication,
        )


def test_orphan_cas_and_forged_receipt_proof(tmp_path):
    f, ledger, p, pp = case(tmp_path)
    t = RetrospectiveReadTask(pp.ref, "INPUT_QUALIFICATION", p.source_slices[0])
    ref = f.put(envelope=ArtifactEnvelope.create(t.artifact_type, 1, t.payload))
    bad = (ref, pp.entry_ref, pp.checkpoint)
    with pytest.raises(RetrospectiveReservationError, match="PUBLICATION_MISSING"):
        reserve_retrospective_read(f, ledger, pp.publication, bad, AT)
    tp = publish_retrospective_read_task(t, f)
    fake = (tp.ref, tp.entry_ref, LogCheckpoint(LOG, AT, 0, None))
    with pytest.raises(RetrospectiveReservationError, match="PUBLICATION_MISSING"):
        reserve_retrospective_read(f, ledger, pp.publication, fake, AT)
    assert not f.entries(SAMPLE)


def test_direct_parent_append_cannot_hide_missing_grandparent_append(tmp_path):
    from crypto_quant_validation import SampleConsumptionRecord

    f, ledger, p, pp = case(tmp_path)
    for phase, hashes in (
        ("INPUT_QUALIFICATION", {}),
        ("SNAPSHOT_FREEZE", {"input_check_hash": H[9]}),
    ):
        register(f, ledger, p, pp, phase, p.source_slices[0], **hashes)
    parts = tuple(
        (
            role,
            RetrospectiveSourceSlice(
                "capture",
                p.capture_revision,
                p.source_scope_hash,
                f"2000-{m:02d}-01T00:00:00.000000Z",
                f"2000-{m + 1:02d}-01T00:00:00.000000Z",
                H[5],
            ),
            f"2000-{m:02d}-10T00:00:00.000000Z",
            f"2000-{m:02d}-20T00:00:00.000000Z",
        )
        for role, m in (("TRAIN", 1), ("DEVELOPMENT", 3), ("RETROSPECTIVE_EVAL", 5))
    )
    sp = publish_retrospective_study(
        RetrospectiveStudyDeclaration(pp.ref, H[5], H[6], H[7], H[8], parts), f
    )
    publish_retrospective_read_task(
        RetrospectiveReadTask(sp.ref, "TRAIN_FEATURE", parts[0][1]), f
    )
    fit = RetrospectiveReadTask(
        sp.ref, "TRAIN_FIT", parts[0][1], feature_rows_hash=H[10]
    )
    fitp = publish_retrospective_read_task(fit, f)
    source = fit.source_slice
    # Public Ledger validates producer/purpose, not this complete phase graph.
    ledger.reserve(
        SampleConsumptionRecord(
            source.capture_revision,
            source.interval_start,
            source.interval_end,
            fit.purpose,
            canonical_sha256(("sample-consumer-v1", fitp.ref)),
            AT,
        ),
        fitp.ref,
    )
    dev = publish_retrospective_read_task(
        RetrospectiveReadTask(
            sp.ref, "DEVELOPMENT_FEATURE", parts[1][1], model_cards_hash=H[11]
        ),
        f,
    )
    with pytest.raises(RetrospectiveReservationError, match="PREREQUISITE_INVALID"):
        reserve_retrospective_read(
            f, ledger, pp.publication, dev.publication, AT, sp.publication
        )


@pytest.mark.parametrize(
    "mutation,code",
    (
        ("protocol_claim", "PROTOCOL_INVALID"),
        ("protocol_hash", "PROTOCOL_INVALID"),
        ("task_extra", "READ_TASK_INVALID"),
        ("task_phase", "READ_TASK_INVALID"),
        ("task_capture", "READ_TASK_INVALID"),
        ("task_scope", "SCOPE_CONFLICT"),
    ),
)
def test_untrusted_canonical_owner_wire_is_rechecked_before_any_append(
    tmp_path, mutation, code
):
    import json

    from crypto_quant_domain import ArtifactRef

    f, ledger, p, pp = case(tmp_path)
    t = RetrospectiveReadTask(pp.ref, "INPUT_QUALIFICATION", p.source_slices[0])
    task_payload = json.loads(canonical_bytes(t.payload))
    protocol_publication = pp.publication
    if mutation.startswith("protocol_"):
        wire = json.loads(canonical_bytes(p.payload))
        if mutation == "protocol_claim":
            wire["untouched_claim"] = True
        else:
            wire["prior_observation_hash"] = "not-a-hash"
        envelope = ArtifactEnvelope.create("retrospective_study_protocol", 1, wire)
        ref = f.put(envelope=envelope)
        receipt = f.append(
            LOG,
            canonical_sha256(("artifact-publication-v1", LOG, ref)),
            canonical_bytes(envelope),
        )
        protocol_publication = (
            ref,
            receipt.entry_ref,
            LogCheckpoint(
                LOG, receipt.accepted_at, receipt.log_sequence, receipt.receipt_hash
            ),
        )
    if mutation == "task_extra":
        task_payload["unexpected"] = True
    elif mutation == "task_phase":
        task_payload["phase"] = "HELD_OUT_FEATURE"
    elif mutation == "task_capture":
        task_payload["source_slice"]["capture_revision"] = "pretend-provider-v1"
    elif mutation == "task_scope":
        task_payload["source_slice"]["interval_end"] = "2000-06-01T00:00:00.000000Z"
    envelope = ArtifactEnvelope.create(t.artifact_type, 1, task_payload)
    ref = f.put(envelope=envelope)
    context = ArtifactRef(
        **{
            key: task_payload["context_ref"][key]
            for key in ("artifact_type", "schema_version", "content_hash")
        }
    )
    event = canonical_sha256(
        (
            "retrospective-read-task-publication-v1",
            context,
            task_payload["phase"],
            "capture",
        )
    )
    receipt = f.append(LOG, event, canonical_bytes(envelope))
    task_publication = (
        ref,
        receipt.entry_ref,
        LogCheckpoint(
            LOG, receipt.accepted_at, receipt.log_sequence, receipt.receipt_hash
        ),
    )
    with pytest.raises(RetrospectiveReservationError, match=code):
        reserve_retrospective_read(
            f, ledger, protocol_publication, task_publication, AT
        )
    assert not f.entries(SAMPLE)


def test_parent_publication_without_actual_append_is_insufficient(tmp_path):
    f, ledger, p, pp = case(tmp_path)
    publish_retrospective_read_task(
        RetrospectiveReadTask(pp.ref, "INPUT_QUALIFICATION", p.source_slices[0]), f
    )
    freeze = publish_retrospective_read_task(
        RetrospectiveReadTask(
            pp.ref, "SNAPSHOT_FREEZE", p.source_slices[0], input_check_hash=H[9]
        ),
        f,
    )
    with pytest.raises(RetrospectiveReservationError, match="PREREQUISITE_INVALID"):
        reserve_retrospective_read(f, ledger, pp.publication, freeze.publication, AT)
    assert not f.entries(SAMPLE)


def test_parent_append_after_child_publication_is_rejected(tmp_path):
    f, ledger, p, pp = case(tmp_path)
    tp = publish_retrospective_read_task(
        RetrospectiveReadTask(pp.ref, "INPUT_QUALIFICATION", p.source_slices[0]), f
    )
    freeze = publish_retrospective_read_task(
        RetrospectiveReadTask(
            pp.ref, "SNAPSHOT_FREEZE", p.source_slices[0], input_check_hash=H[9]
        ),
        f,
    )
    reserve_retrospective_read(f, ledger, pp.publication, tp.publication, AT)
    with pytest.raises(RetrospectiveReservationError, match="PREREQUISITE_INVALID"):
        reserve_retrospective_read(f, ledger, pp.publication, freeze.publication, AT)


def test_foreign_ledger_and_governance_time_fail_closed(tmp_path):
    f, ledger, p, pp = case(tmp_path / "a")
    tp = publish_retrospective_read_task(
        RetrospectiveReadTask(pp.ref, "INPUT_QUALIFICATION", p.source_slices[0]), f
    )
    other = LocalFoundation(tmp_path / "b", clock=lambda: AT)
    with pytest.raises(RetrospectiveReservationError, match="RESERVATION_FAILED"):
        reserve_retrospective_read(
            f, SampleConsumptionLedger(other), pp.publication, tp.publication, AT
        )
    for at in ("2026-01-31T00:00:00.000000Z", "2026-02-02T00:00:00.000000Z"):
        with pytest.raises(RetrospectiveReservationError, match="RESERVATION_FAILED"):
            reserve_retrospective_read(f, ledger, pp.publication, tp.publication, at)
    assert not f.entries(SAMPLE)


def test_wrong_context_and_extraneous_study_are_not_admitted(tmp_path):
    f, ledger, _p, pp, _s, sp, tasks = full(tmp_path)
    input_task = tasks["INPUT_QUALIFICATION"][1]
    with pytest.raises(RetrospectiveReservationError, match="SCOPE_CONFLICT"):
        reserve_retrospective_read(
            f, ledger, pp.publication, input_task.publication, AT, sp.publication
        )
    ft = tasks["TRAIN_FEATURE"][1]
    with pytest.raises(RetrospectiveReservationError, match="STUDY_INVALID"):
        reserve_retrospective_read(f, ledger, pp.publication, ft.publication, AT)


def test_reservation_partial_failure_replays_original_append(tmp_path, monkeypatch):
    import crypto_quant_validation.retrospective_reservation as module

    f, ledger, p, pp = case(tmp_path)
    tp = publish_retrospective_read_task(
        RetrospectiveReadTask(pp.ref, "INPUT_QUALIFICATION", p.source_slices[0]), f
    )
    original = module._verify_reservation

    def failed(*args, **kwargs):
        raise ValueError("synthetic post-append failure")

    monkeypatch.setattr(module, "_verify_reservation", failed)
    with pytest.raises(RetrospectiveReservationError, match="RESERVATION_FAILED"):
        reserve_retrospective_read(f, ledger, pp.publication, tp.publication, AT)
    assert len(f.entries(SAMPLE)) == 1
    monkeypatch.setattr(module, "_verify_reservation", original)
    r = reserve_retrospective_read(f, ledger, pp.publication, tp.publication, AT)
    assert (
        r.reservation_entry_ref == f.entries(SAMPLE)[0].entry_ref
        and len(f.entries(SAMPLE)) == 1
    )
