from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from crypto_quant_domain import ArtifactRef
from crypto_quant_research import verify_candidate_provenance
from crypto_quant_validation import NoReport, PublishedValidationReport, VerifiedCandidateProvenance, validate_candidate
from crypto_quant_validation import runtime

_spec = importlib.util.spec_from_file_location("provenance_validation_shell", Path(__file__).with_name("test_validation_shell.py"))
shell = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shell)


def _validate(context, *, verifier=verify_candidate_provenance):
    foundation, ledger, port, candidate, policy = context
    return validate_candidate(candidate, policy, {"fixture_case": "adverse_completed"}, shell.RESERVED_AT, foundation, ledger, port, candidate_provenance_verifier=verifier)


def _read_only(context, monkeypatch):
    foundation, ledger, port, _, _ = context
    def forbidden(*args, **kwargs):
        pytest.fail("closed report replay performed a write, clock, or economic operation")
    for name in ("put", "append", "checkpoint", "_clock"):
        monkeypatch.setattr(foundation, name, forbidden)
    for name in ("reserve", "freeze_snapshot", "assess_holdout"):
        monkeypatch.setattr(ledger, name, forbidden)
    for name in ("run", "derive"):
        monkeypatch.setattr(port, name, forbidden)


def test_valid_replay_reverifies_once_without_any_side_effects(tmp_path, monkeypatch):
    context = shell._setup(tmp_path)
    calls = []
    def verify(candidate, foundation, port):
        calls.append(candidate)
        return verify_candidate_provenance(candidate, foundation, port)
    first = _validate(context, verifier=verify)
    assert type(first) is PublishedValidationReport
    _read_only(context, monkeypatch)
    assert _validate(context, verifier=verify) == first
    assert calls == [context[3], context[3]]


@pytest.mark.parametrize("field", ["candidate_ref", "candidate_family_ref", "experiment_ref", "execution_manifest_ref", "selection_declaration_ref", "manifest_cutoff_ref"])
def test_verifier_identity_cannot_be_reused_for_another_graph(field, tmp_path):
    context = shell._setup(tmp_path)
    def wrong(candidate, foundation, port):
        value = verify_candidate_provenance(candidate, foundation, port)
        wire = dict(getattr(value, field))
        wire["receipt_hash" if field == "manifest_cutoff_ref" else "content_hash"] = "sha256:" + "f" * 64
        fields = {name: dict(getattr(value, name)) for name in value.__dataclass_fields__}
        fields[field] = wire
        return VerifiedCandidateProvenance(**fields)
    assert _validate(context, verifier=wrong) == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
    assert context[0].entries(shell.ARTIFACT_LOG) == ()


@pytest.mark.parametrize("field", ["candidate_ref", "candidate_family_ref", "experiment_ref", "execution_manifest_ref", "selection_declaration_ref", "manifest_cutoff_ref"])
def test_boolean_versions_and_cutoffs_are_not_canonical(field, tmp_path):
    context = shell._setup(tmp_path)
    value = verify_candidate_provenance(context[3], context[0], context[2])
    fields = {name: dict(getattr(value, name)) for name in value.__dataclass_fields__}
    fields[field]["log_sequence" if field == "manifest_cutoff_ref" else "schema_version"] = True
    with pytest.raises(ValueError):
        VerifiedCandidateProvenance(**fields)


def test_incomplete_experiment_is_not_runtime_provenance(tmp_path):
    context = list(shell._setup(tmp_path))
    context[3] = shell._incomplete_candidate(context[0], context[1], context[2])
    assert _validate(context) == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
    assert context[0].entries(shell.ARTIFACT_LOG) == ()


@pytest.mark.parametrize("mutation", ["empty", "duplicate", "reordered", "foreign_case", "threshold", "boolean_threshold", "result", "limitations", "fake_assessment", "case_outcome", "case_reason", "case_limitations", "case_evidence", "case_missing_evidence", "case_extra_fields", "boolean_assessment"])
def test_existing_report_is_recomputed_not_trusted(mutation, tmp_path, monkeypatch):
    context = shell._setup(tmp_path)
    publish = runtime._publish
    def corrupt(foundation, kind, payload):
        payload = deepcopy(payload)
        if kind == "validation_case_result" and payload["evidence"] is not None and "publication_ref" in payload["evidence"]:
            if mutation == "case_outcome": payload["outcome"] = "PASS"
            if mutation == "case_reason": payload["reason_codes"] = ["HOLDOUT_CONTAMINATED"]
            if mutation == "case_limitations": payload["limitations"] = ["invented"]
            if mutation == "case_evidence": payload["evidence"]["metric_value"] = "100"
            if mutation == "case_missing_evidence": payload.pop("evidence")
            if mutation == "case_extra_fields": payload["extra"] = "unrecognized"
        if kind == "validation_report":
            if mutation == "empty": payload["case_result_refs"] = []
            if mutation == "duplicate": payload["case_result_refs"][1] = payload["case_result_refs"][0]
            if mutation == "reordered": payload["case_result_refs"].reverse()
            if mutation == "foreign_case":
                foreign_case = publish(foundation, "validation_case", {"validation_plan_ref": shell._wire(context[3]), "case_type": "out_of_sample"})
                payload["case_result_refs"][1] = shell._wire(publish(foundation, "validation_case_result", {"case_ref": shell._wire(foreign_case), "outcome": "PASS", "reason_codes": [], "limitations": [], "evidence": None}))
            if mutation == "threshold": payload["threshold_evaluations"][0]["threshold"] = "-100"
            if mutation == "boolean_threshold": payload["threshold_evaluations"][0]["passed"] = 0
            if mutation == "result": payload["result"] = "supported"
            if mutation == "limitations": payload["limitations"] = ["invented"]
            if mutation in {"fake_assessment", "boolean_assessment"}:
                assessment_ref = runtime._ref_from_wire(payload["sample_integrity_ref"])
                assessment = shell._payload(foundation, assessment_ref)
                assessment["untouched"] = False if mutation == "fake_assessment" else 1
                payload["sample_integrity_ref"] = shell._wire(publish(foundation, "sample_integrity_assessment", assessment))
        return publish(foundation, kind, payload)
    monkeypatch.setattr(runtime, "_publish", corrupt)
    assert type(_validate(context)) is PublishedValidationReport
    _read_only(context, monkeypatch)
    assert _validate(context) == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")


def test_report_without_original_oos_reservation_is_not_repaired(tmp_path, monkeypatch):
    context = shell._setup(tmp_path)
    monkeypatch.setattr(context[1], "reserve", lambda *args: None)
    assert type(_validate(context)) is PublishedValidationReport
    _read_only(context, monkeypatch)
    assert _validate(context) == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")


def test_replay_uses_original_snapshot_despite_later_sample_usage(tmp_path, monkeypatch):
    context = shell._setup(tmp_path)
    first = _validate(context)
    late_case = ArtifactRef("validation_case", 1, "sha256:" + "f" * 64)
    context[1].reserve(shell._record(late_case, "validation"), late_case)
    _read_only(context, monkeypatch)
    assert _validate(context) == first


def test_noncallable_callback_is_a_public_type_error(tmp_path):
    with pytest.raises(TypeError, match="candidate_provenance_verifier must be callable"):
        _validate(shell._setup(tmp_path), verifier=1)


def test_target_missing_or_noncallable_verifier_has_no_io(tmp_path, monkeypatch):
    from crypto_quant_validation import validate_target_candidate
    foundation, ledger, port, _, policy = shell._setup(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("target missing verifier performed I/O")
    for name in ("read", "entries", "put", "append", "checkpoint", "_clock"):
        monkeypatch.setattr(foundation, name, forbidden)
    candidate = ArtifactRef("strategy_candidate", 3, "sha256:" + "a" * 64)
    args = (candidate, policy, shell.RESERVED_AT, foundation, ledger, object(), object())
    assert validate_target_candidate(*args) == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
    with pytest.raises(TypeError, match="candidate_provenance_verifier must be callable"):
        validate_target_candidate(*args, candidate_provenance_verifier=1)


@pytest.mark.parametrize("case_id", ["terminal_blocked", "terminal_cancelled"])
def test_terminal_report_replay_loads_original_terminal_without_execution(case_id, tmp_path, monkeypatch):
    foundation, ledger, port, candidate, policy = context = shell._setup(tmp_path)
    first = validate_candidate(candidate, policy, {"fixture_case": case_id}, shell.RESERVED_AT, foundation, ledger, port, candidate_provenance_verifier=verify_candidate_provenance)
    assert type(first) is PublishedValidationReport
    _read_only(context, monkeypatch)
    assert _validate(context) == first


@pytest.mark.parametrize("mutation", ["other_trial", "late_start", "late_selection"])
def test_runtime_rejects_reopened_or_misselected_research_candidate(mutation, tmp_path, monkeypatch):
    from crypto_quant_research import execute_experiment
    path = Path(__file__).resolve().parents[2] / "research-platform/tests/test_research_shell.py"
    spec = importlib.util.spec_from_file_location("runtime_provenance_research_shell", path)
    research = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(research)
    foundation, ledger, port = research._runtime(tmp_path)
    result = execute_experiment(research._inputs(), foundation, ledger, port)
    candidate_ref = result.strategy_candidate_ref
    candidate = shell._payload(foundation, candidate_ref)
    if mutation == "other_trial":
        trial = next(shell._wire(ArtifactRef(item["artifact_type"], item["schema_version"], item["content_hash"]))
                     for entry in foundation.entries(shell.RESEARCH_ARTIFACT_LOG)
                     if (item := json.loads(entry.payload))["artifact_type"] == "trial_declaration"
                     and shell._wire(ArtifactRef(item["artifact_type"], item["schema_version"], item["content_hash"])) != candidate["selected_trial_declaration_ref"])
        candidate["selected_trial_declaration_ref"] = trial
        candidate_ref = shell._publish(foundation, shell.RESEARCH_ARTIFACT_LOG, "strategy_candidate", candidate)
    if mutation == "late_start":
        shell._publish(foundation, shell.RESEARCH_EXECUTION_LOG, "task_attempt_started", {
            "task_ref": {"kind": "TRIAL", "task_artifact_ref": candidate["selected_trial_declaration_ref"]},
            "ordinal": 2, "parent_closed_attempt_ref": None, "selection_declaration_refs": [], "dispatch_ref": None,
        })
    if mutation == "late_selection":
        entries = foundation.entries
        def late(log, checkpoint=None):
            return tuple(replace(entry, ledger_sequence=100000) if json.loads(entry.payload)["artifact_type"] == "selection_declaration" else entry for entry in entries(log, checkpoint))
        monkeypatch.setattr(foundation, "entries", late)
    profile = port.case("completed-10-1")["derive"]["metric_profile_ref"]
    context = (foundation, ledger, port, candidate_ref, shell._policy(profile))
    _read_only(context, monkeypatch)
    assert _validate(context) == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")


@pytest.mark.parametrize("code", ["PORT_RETENTION_UNAVAILABLE", "PORT_MANIFEST_INVALID"])
def test_real_verifier_provider_failure_keeps_port_code(code, tmp_path, monkeypatch):
    from tests.support.backtest_consumer_port import PortFailure
    context = shell._setup(tmp_path)
    def fail(*args, **kwargs):
        raise PortFailure(code)
    monkeypatch.setattr(context[2], "load_completed", fail)
    assert _validate(context) == NoReport(None, code)
    assert context[0].entries(shell.ARTIFACT_LOG) == ()


@pytest.mark.parametrize("code", ["ARTIFACT_NOT_FOUND", "WRITE_LOCK_UNAVAILABLE"])
def test_real_verifier_foundation_failure_preserves_infrastructure(code, tmp_path, monkeypatch):
    from crypto_quant_foundation import FoundationFailure
    context = shell._setup(tmp_path)
    def fail(*args, **kwargs):
        raise FoundationFailure(code)
    monkeypatch.setattr(context[0], "read", fail)
    if code == "ARTIFACT_NOT_FOUND":
        assert _validate(context) == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
    else:
        with pytest.raises(FoundationFailure) as caught:
            _validate(context)
        assert caught.value.code == code


def test_wrong_callback_result_and_semantic_error_fail_closed(tmp_path):
    context = shell._setup(tmp_path)
    assert _validate(context, verifier=lambda *args: True) == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
    def malformed(*args):
        raise ValueError("invalid Research closure")
    assert _validate(context, verifier=malformed) == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")


def test_original_oos_reservation_must_precede_original_result_publication(tmp_path, monkeypatch):
    context = shell._setup(tmp_path)
    assert type(_validate(context)) is PublishedValidationReport
    foundation = context[0]
    entries = foundation.entries
    def late(log, checkpoint=None):
        return tuple(
            replace(entry, ledger_sequence=100000)
            if log == shell.SAMPLE_LOG and json.loads(entry.payload)["payload"]["producer_ref"]["artifact_type"] == "validation_case"
            else entry for entry in entries(log, checkpoint)
        )
    monkeypatch.setattr(foundation, "entries", late)
    _read_only(context, monkeypatch)
    assert _validate(context) == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
