"""Qualification publications coexist with retained Validation owner-log readers."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
from crypto_quant_domain import ArtifactEnvelope, ArtifactRef, canonical_bytes, canonical_sha256
from crypto_quant_research import (
    InputQualificationCaptureBindingV1, publish_input_qualification_capture_binding,
    publish_input_qualification_read_task, verify_candidate_provenance,
)
from crypto_quant_validation import PublishedValidationReport, validate_candidate
from crypto_quant_validation import target_runtime

from . import test_input_qualification_reservation as qualification
from . import test_validation_outcome_view as outcomes


RESEARCH_LOG = "research.artifacts.v1"
VALIDATION_LOG = "validation.artifacts.v1"
KINDS = (
    "input_qualification_declaration",
    "input_qualification_read_task",
    "input_qualification_capture_binding",
)


def _publish_capture(root, foundation, ledger):
    case = qualification.new_case(root, foundation=foundation, ledger=ledger)
    for phase in ("QUALIFY", "CAPTURE"):
        task = publish_input_qualification_read_task(qualification.task(case, phase), foundation, case.verifier)
        receipt = case.verifier.reserve(
            case.dp.publication, task.publication, task.checkpoint.as_of,
            qualification.actual(case, phase),
        )
    manifest = {
        "type": "market_bundle_manifest", "bundle_key": "synthetic", "schema_version": 1,
        "coverage_start": qualification.START, "coverage_end_exclusive": qualification.END,
        "instrument_catalog_hash": qualification.H, "capabilities": (), "streams": (),
        "content_hash": qualification.H,
    }
    binding = InputQualificationCaptureBindingV1(
        case.dp.ref, (("prices", task.ref, receipt.reservation_entry_ref),), qualification.H,
        {"type": "market_bundle_ref", "bundle_key": "synthetic", "manifest_hash": canonical_sha256(manifest)},
        manifest,
    )
    publication = publish_input_qualification_capture_binding(binding, foundation, case.verifier)
    assert case.verifier.verify_capture(publication.publication).binding_ref == publication.ref


def _completed_plan(root, mode):
    if mode == "ordinary":
        context = outcomes.shell._setup(root)
        foundation, ledger, port = context[:3]
        result = outcomes._validate(context)
    elif mode == "model":
        integrated = outcomes._fixture("test_integrated_validation")
        foundation, ledger, port, candidate, _, policy = integrated._model_runtime(root)
        result = validate_candidate(
            candidate, policy, {"binding_key": "validation:oos:model"},
            integrated._RESERVED_AT, foundation, ledger, port,
            candidate_provenance_verifier=verify_candidate_provenance,
        )
    else:
        context = outcomes.target._runtime(root)
        foundation, ledger, _, port, _ = context
        result = outcomes._target_validate(context)
    assert type(result) is PublishedValidationReport
    return foundation, ledger, port, result


@pytest.mark.parametrize("mode", ["ordinary", "model", "target"])
def test_public_outcome_view_preserves_existing_plan_with_mixed_qualification_log(tmp_path, monkeypatch, mode):
    foundation, ledger, port, result = _completed_plan(tmp_path, mode)
    original = outcomes._view(foundation, port, result.validation_plan_ref)
    _publish_capture(tmp_path, foundation, ledger)
    before = outcomes._frozen(tmp_path)
    outcomes._readonly(monkeypatch, foundation, ledger, port)
    view = outcomes._view(foundation, port, result.validation_plan_ref)
    assert view["source_log_heads"][RESEARCH_LOG] != original["source_log_heads"][RESEARCH_LOG]
    assert {key: value for key, value in view.items() if key != "source_log_heads"} == {
        key: value for key, value in original.items() if key != "source_log_heads"
    }
    assert outcomes._frozen(tmp_path) == before


def test_public_target_execution_and_closed_replay_accept_mixed_qualification_log(tmp_path, monkeypatch):
    context = outcomes.target._runtime(tmp_path)
    foundation, ledger, materializer, port, _ = context
    _publish_capture(tmp_path, foundation, ledger)
    result = outcomes._target_validate(context)
    assert type(result) is PublishedValidationReport
    before = outcomes._frozen(tmp_path)
    outcomes._readonly(monkeypatch, foundation, ledger, port, materializer)
    assert outcomes._target_validate(context) == result
    assert outcomes._view(foundation, port, result.validation_plan_ref)["report_ref"] == result.validation_report_ref.to_canonical_dict()
    assert outcomes._frozen(tmp_path) == before


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mutation", ["fields", "schema", "event", "wrong_log"])
def test_malformed_qualification_publication_is_rejected_without_writes(tmp_path, monkeypatch, kind, mutation):
    foundation, ledger, port, result = _completed_plan(tmp_path, "ordinary")
    _publish_capture(tmp_path, foundation, ledger)
    entries = foundation.entries
    selected = next(entry for entry in entries(RESEARCH_LOG) if json.loads(entry.payload)["artifact_type"] == kind)
    decoded = json.loads(selected.payload)
    payload = deepcopy(decoded["payload"])
    version = decoded["schema_version"]
    if mutation == "fields":
        payload["unrecognized"] = True
    elif mutation == "schema":
        version = 2
    envelope = ArtifactEnvelope.create(kind, version, payload)
    ref = ArtifactRef.from_envelope(envelope)
    event = selected.event_id
    if mutation in ("fields", "schema", "event"):
        # Generic publication must not bypass strict qualification decoding or slots.
        event = canonical_sha256(("artifact-publication-v1", RESEARCH_LOG, ref))
        if kind == "input_qualification_declaration" and mutation == "event":
            event = canonical_sha256("not-an-owner-publication")
    malformed = replace(selected, payload=canonical_bytes(envelope), event_id=event)
    log = VALIDATION_LOG if mutation == "wrong_log" else RESEARCH_LOG

    def malformed_entries(log_name, through=None):
        values = entries(log_name, through)
        if through is not None:
            return values
        if mutation == "wrong_log":
            return values + (malformed,) if log_name == VALIDATION_LOG else values
        return tuple(malformed if entry == selected else entry for entry in values) if log_name == RESEARCH_LOG else values

    monkeypatch.setattr(foundation, "entries", malformed_entries)
    before = outcomes._frozen(tmp_path)
    outcomes._readonly(monkeypatch, foundation, ledger, port)
    with pytest.raises(ValueError, match="owner publication is invalid"):
        target_runtime._entries(foundation, log)
    with pytest.raises(ValueError):
        outcomes._view(foundation, port, result.validation_plan_ref)
    assert outcomes._frozen(tmp_path) == before


def test_unknown_type_does_not_gain_qualification_event_exception(tmp_path, monkeypatch):
    foundation, ledger, port, result = _completed_plan(tmp_path, "ordinary")
    envelope = ArtifactEnvelope.create("unknown_qualification_type", 1, {})
    foundation.append(RESEARCH_LOG, canonical_sha256("unknown-event"), canonical_bytes(envelope))
    before = outcomes._frozen(tmp_path)
    outcomes._readonly(monkeypatch, foundation, ledger, port)
    with pytest.raises(ValueError, match="owner publication is invalid"):
        target_runtime._entries(foundation, RESEARCH_LOG)
    with pytest.raises(ValueError):
        outcomes._view(foundation, port, result.validation_plan_ref)
    assert outcomes._frozen(tmp_path) == before
