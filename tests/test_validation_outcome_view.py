from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from crypto_quant_domain import ArtifactEnvelope, ArtifactRef, canonical_bytes
from crypto_quant_foundation import FoundationFailure
from crypto_quant_research import verify_candidate_provenance
from crypto_quant_validation import (
    NoReport, PublishedValidationReport, describe_validation_return,
    read_validation_outcome_view, validate_candidate, validate_target_candidate,
)
from crypto_quant_validation import runtime, target_runtime


def _fixture(name):
    spec = importlib.util.spec_from_file_location("outcome_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shell = _fixture("test_validation_shell")
target = _fixture("test_target_validation")


def _validate(context, case_id="adverse_completed"):
    f, ledger, port, candidate, policy = context
    return validate_candidate(candidate, policy, {"fixture_case": case_id}, shell.RESERVED_AT,
        f, ledger, port, candidate_provenance_verifier=verify_candidate_provenance)


def _view(foundation, backtest, plan_ref):
    return json.loads(read_validation_outcome_view(plan_ref, foundation=foundation, backtest=backtest,
        candidate_provenance_verifier=verify_candidate_provenance))


def _frozen(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _readonly(monkeypatch, foundation, ledger, port, materializer=None):
    def forbidden(*args, **kwargs):
        pytest.fail("reporting attempted execution, publication, reservation, or clock access")
    for name in ("put", "append", "checkpoint", "_clock"):
        monkeypatch.setattr(foundation, name, forbidden)
    for name in ("reserve", "freeze_snapshot", "assess_holdout"):
        monkeypatch.setattr(ledger, name, forbidden)
    for name in ("run", "derive", "prepare_target", "publish_target"):
        if hasattr(port, name):
            monkeypatch.setattr(port, name, forbidden)
    if materializer is not None:
        monkeypatch.setattr(materializer, "materialize_target", forbidden)


def test_supported_negative_return_is_not_reported_as_profitable(tmp_path, monkeypatch):
    f, ledger, port, candidate, policy = shell._setup(tmp_path)
    policy = replace(policy, oos_rule=replace(policy.oos_rule, threshold="-0.2"))
    result = _validate((f, ledger, port, candidate, policy))
    before = _frozen(tmp_path)
    _readonly(monkeypatch, f, ledger, port)
    view = _view(f, port, result.validation_plan_ref)
    assert view["original_result"]["value"] == "supported"
    threshold = view["threshold_evaluations"][0]
    assert threshold["metric"]["fact"]["value"] == "-0.1"
    assert threshold["rule"]["threshold"] == "-0.2"
    assert threshold["passed"]["value"] is True
    assert "profitable" not in json.dumps(view)
    assert view["report_ref"] == shell._wire(result.validation_report_ref)
    assert _frozen(tmp_path) == before


def test_corrupt_referenced_snapshot_is_not_unknown(tmp_path, monkeypatch):
    f, ledger, port, _, _ = context = shell._setup(tmp_path)
    result = _validate(context)
    payload = shell._payload(f, result.validation_plan_ref)
    snapshot_ref = runtime._ref_from_wire(payload["sample_consumption_snapshot_ref"])
    read = f.read
    def corrupt(*, ref):
        if ref == snapshot_ref:
            raise FoundationFailure("ARTIFACT_INTEGRITY")
        return read(ref=ref)
    monkeypatch.setattr(f, "read", corrupt)
    _readonly(monkeypatch, f, ledger, port)
    with pytest.raises(FoundationFailure) as caught:
        _view(f, port, result.validation_plan_ref)
    assert caught.value.code == "ARTIFACT_INTEGRITY"


@pytest.mark.parametrize(("case_id", "result", "outcome"), [
    ("adverse_completed", "rejected", "FAIL"),
    ("terminal_blocked", "inconclusive", "BLOCKED"),
    ("terminal_cancelled", "inconclusive", "INCONCLUSIVE"),
    ("terminal_failed", None, "FAILED"),
    ("provider_failure", None, "FAILED"),
])
def test_existing_results_and_durable_no_report_facts(tmp_path, monkeypatch, case_id, result, outcome):
    f, ledger, port, _, _ = context = shell._setup(tmp_path)
    returned = _validate(context, case_id)
    before = _frozen(tmp_path)
    _readonly(monkeypatch, f, ledger, port)
    view = _view(f, port, returned.validation_plan_ref)
    assert (view["original_result"]["value"] if view["original_result"] else None) == result
    assert view["case_rows"][1]["outcome"] == outcome
    assert view["threshold_evaluations"] == [] or case_id == "adverse_completed"
    if result is None:
        assert view["report_ref"] is None
        assert any("not recoverable" in note for note in view["limitations"])
    assert _frozen(tmp_path) == before


@pytest.mark.parametrize("missing", [True, False], ids=["missing_metric", "insufficient_trades"])
def test_inconclusive_metrics_remain_missing_not_zero(tmp_path, monkeypatch, missing):
    mutation = (lambda r: r.pop("simple_period_return")) if missing else (lambda r: r.__setitem__("trade_count", 0))
    f, ledger, port, _, _ = context = shell._setup(tmp_path, shell.OosAnalysisPort, mutation)
    result = _validate(context)
    _readonly(monkeypatch, f, ledger, port)
    view = _view(f, port, result.validation_plan_ref)
    assert view["original_result"]["value"] == "inconclusive"
    assert view["threshold_evaluations"] == []
    assert view["case_rows"][1]["evidence"]["metric_value"] is None if missing else view["case_rows"][1]["evidence"]["trade_count"] == 0


def _target_validate(context):
    f, ledger, materializer, port, candidate = context
    return validate_target_candidate(candidate.strategy_candidate_ref, target._policy(), target._RESEARCH.RESERVED_AT,
        f, ledger, materializer, port, candidate_provenance_verifier=verify_candidate_provenance)


def _prefix(context, monkeypatch, stage, target_mode=False):
    module = target_runtime if target_mode else runtime
    publish = module._publish
    def interrupt(*args):
        ref = publish(*args)
        kind = args[1]
        payload = args[-1]
        selected = {
            "plan": kind == "validation_plan",
            "admission_case": kind == "validation_case" and payload.get("case_type") == "evidence_integrity",
            "admission": kind == "validation_case_result" and "snapshot_ref" in (payload.get("evidence") or {}),
            "oos_case": kind == "validation_case" and payload.get("case_type") == "out_of_sample",
            "target": kind == "validation_target_materialization_evidence",
            "oos_result": kind == "validation_case_result" and "snapshot_ref" not in (payload.get("evidence") or {}),
        }[stage]
        if selected:
            raise KeyboardInterrupt("synthetic publication boundary")
        return ref
    with monkeypatch.context() as scoped:
        scoped.setattr(module, "_publish", interrupt)
        with pytest.raises(KeyboardInterrupt):
            (_target_validate if target_mode else _validate)(context)
    return next(ArtifactRef.from_envelope(ArtifactEnvelope(
        value["artifact_type"], value["schema_version"], value["payload"], value["content_hash"]))
        for entry in context[0].entries(shell.ARTIFACT_LOG)
        if (value := json.loads(entry.payload))["artifact_type"] == "validation_plan")


@pytest.mark.parametrize("target_mode", [False, True], ids=["ordinary", "target"])
@pytest.mark.parametrize("stage", ["plan", "admission_case", "admission", "oos_case", "oos_result"])
def test_legitimate_publication_prefix_is_not_resumed(tmp_path, monkeypatch, target_mode, stage):
    context = target._runtime(tmp_path) if target_mode else shell._setup(tmp_path)
    f, ledger = context[:2]
    port = context[3] if target_mode else context[2]
    plan_ref = _prefix(context, monkeypatch, stage, target_mode)
    before = _frozen(tmp_path)
    _readonly(monkeypatch, f, ledger, port, context[2] if target_mode else None)
    view = _view(f, port, plan_ref)
    assert view["report_ref"] is None and view["original_result"] is None
    assert view["threshold_evaluations"] == []
    assert (view["sample_integrity_refs"]["assessment_ref"] is None) == (stage == "plan")
    assert (view["case_rows"][0]["case_ref"] is None) == (stage == "plan")
    assert (view["case_rows"][0]["case_result_ref"] is not None) == (stage in ["admission", "oos_case", "oos_result"])
    assert (view["case_rows"][1]["case_ref"] is not None) == (stage in ["oos_case", "oos_result"])
    assert (view["case_rows"][1]["case_result_ref"] is not None) == (stage == "oos_result")
    anchor = view["observation_anchor"]
    assert anchor["kind"] == ("oos_reservation" if stage == "oos_result" else "plan_snapshot")
    if stage != "oos_result":
        assert anchor["source_ref"] == view["sample_integrity_refs"]["snapshot_ref"]
        assert anchor["source_fields"][0]["pointer"] == "/checkpoint/as_of"
    assert _frozen(tmp_path) == before


def test_committed_target_without_oos_result_is_verified_readonly(tmp_path, monkeypatch):
    context = target._runtime(tmp_path)
    f, ledger, materializer, port, _ = context
    plan_ref = _prefix(context, monkeypatch, "target", True)
    before = _frozen(tmp_path)
    _readonly(monkeypatch, f, ledger, port, materializer)
    view = _view(f, port, plan_ref)
    evidence_ref = next(ref for ref, _ in runtime._artifact_log_payloads(f)
        if ref.artifact_type == "validation_target_materialization_evidence")
    assert view["target_evidence_ref"] == shell._wire(evidence_ref)
    assert view["case_rows"][1]["case_result_ref"] is None
    assert view["case_rows"][1]["evidence"] is None
    assert view["observation_anchor"]["kind"] == "oos_reservation"
    assert _frozen(tmp_path) == before


@pytest.mark.parametrize("mutation", ["unsupported", "ambiguous"])
def test_committed_target_prefix_rejects_unsupported_or_ambiguous_evidence(tmp_path, monkeypatch, mutation):
    context = target._runtime(tmp_path)
    f, ledger, materializer, port, _ = context
    publish = target_runtime._publish
    def forge(foundation, kind, version, payload):
        if kind == "validation_target_materialization_evidence":
            if mutation == "ambiguous":
                publish(foundation, kind, version, payload)
            version = 2
        return publish(foundation, kind, version, payload)
    with monkeypatch.context() as scoped:
        scoped.setattr(target_runtime, "_publish", forge)
        plan_ref = _prefix(context, scoped, "target", True)
    publications = runtime._artifact_log_payloads(f)
    evidence_refs = [ref for ref, _ in publications
        if ref.artifact_type == "validation_target_materialization_evidence"]
    assert [ref.schema_version for ref in evidence_refs] == ([1, 2] if mutation == "ambiguous" else [2])
    assert all(ref.artifact_type != "validation_report" for ref, _ in publications)
    before = _frozen(tmp_path)
    _readonly(monkeypatch, f, ledger, port, materializer)
    with pytest.raises(ValueError) as caught:
        _view(f, port, plan_ref)
    assert caught.value.code == "VALIDATION_OUTCOME_VIEW_INVALID"
    assert _frozen(tmp_path) == before


@pytest.mark.parametrize("target_mode", [False, True], ids=["ordinary", "target"])
def test_original_reservation_time_and_coordinate_not_snapshot_time(tmp_path, monkeypatch, target_mode):
    context = target._runtime(tmp_path) if target_mode else shell._setup(tmp_path)
    f, ledger = context[:2]
    port = context[3] if target_mode else context[2]
    result = (_target_validate if target_mode else _validate)(context)
    snapshot_ref = runtime._ref_from_wire(shell._payload(f, result.validation_plan_ref)["sample_consumption_snapshot_ref"])
    snapshot_time = shell._payload(f, snapshot_ref)["checkpoint"]["as_of"]
    reserve = f.entries(shell.SAMPLE_LOG)[-1]
    record = json.loads(reserve.payload)["payload"]["record"]
    assert record["consumed_at"] != snapshot_time
    _readonly(monkeypatch, f, ledger, port, context[2] if target_mode else None)
    view = _view(f, port, result.validation_plan_ref)
    anchor = view["observation_anchor"]
    assert anchor == {"kind": "oos_reservation", "value": record["consumed_at"],
        "source_ref": runtime._entry_ref_wire(reserve.entry_ref),
        "source_fields": [{"source_ref": runtime._entry_ref_wire(reserve.entry_ref),
            "document_kind": "artifact_payload", "pointer": "/record/consumed_at"}]}


@pytest.mark.parametrize("target_mode", [False, True], ids=["ordinary", "target"])
@pytest.mark.parametrize("mutation", ["missing", "scope", "purpose", "consumer", "ambiguous", "late", "future"])
def test_invalid_original_reservation_cannot_fall_back_to_snapshot(tmp_path, monkeypatch, target_mode, mutation):
    context = target._runtime(tmp_path) if target_mode else shell._setup(tmp_path)
    f = context[0]
    port = context[3] if target_mode else context[2]
    result = (_target_validate if target_mode else _validate)(context)
    entries = f.entries
    reservation = entries(shell.SAMPLE_LOG)[-1]
    def invalid_entries(log, through=None):
        values = entries(log, through)
        if log != shell.SAMPLE_LOG or through is not None:
            return values
        if mutation == "missing":
            return values[:-1]
        if mutation == "ambiguous":
            return values + (reservation,)
        if mutation == "late":
            return values[:-1] + (replace(reservation, ledger_sequence=100000),)
        wire = json.loads(reservation.payload)
        record = wire["payload"]["record"]
        record[{"scope": "dataset_revision", "purpose": "purpose", "consumer": "consumer_id", "future": "consumed_at"}[mutation]] = (
            "2099-01-01T00:00:00.000000Z" if mutation == "future" else "forged")
        envelope = ArtifactEnvelope.create("sample_consumption_append", 1, wire["payload"])
        return values[:-1] + (replace(reservation, payload=canonical_bytes(envelope)),)
    monkeypatch.setattr(f, "entries", invalid_entries)
    with pytest.raises((ValueError, RuntimeError)):
        _view(f, port, result.validation_plan_ref)


@pytest.mark.parametrize("mutation", ["outcome", "metric", "analysis", "case", "extra", "report"])
def test_forged_or_foreign_graph_is_not_reinterpreted(tmp_path, monkeypatch, mutation):
    context = shell._setup(tmp_path)
    publish = runtime._publish
    def forge(f, kind, payload):
        payload = deepcopy(payload)
        if kind == "validation_case_result" and "publication_ref" in (payload.get("evidence") or {}):
            if mutation == "outcome": payload["outcome"] = "PASS"
            if mutation == "metric": payload["evidence"]["metric_value"] = "100"
            if mutation == "analysis": payload["evidence"]["analysis_ref"] = context[2].case("research:adverse_completed")["analysis"]["analysis_ref"]
            if mutation == "case": payload["case_ref"] = shell._wire(ArtifactRef("validation_case", 1, "sha256:" + "f" * 64))
            if mutation == "extra": payload["unrecognized"] = True
        if kind == "validation_report" and mutation == "report": payload["result"] = "supported"
        return publish(f, kind, payload)
    with monkeypatch.context() as scoped:
        scoped.setattr(runtime, "_publish", forge)
        result = _validate(context)
    with pytest.raises((ValueError, RuntimeError)):
        _view(context[0], context[2], result.validation_plan_ref)


@pytest.mark.parametrize("mutation", ["missing", "digest", "producer", "count", "discovery"])
def test_validation_target_cannot_be_missing_tampered_or_discovery_target(tmp_path, monkeypatch, mutation):
    context = target._runtime(tmp_path)
    f, _, _, port, candidate = context
    result = _target_validate(context)
    load = port.load_target
    discovery = shell._payload(f, runtime._ref_from_wire(shell._payload(f, candidate.strategy_candidate_ref)["selected_target_materialization_evidence_ref"]))
    def corrupt(ref):
        record = load(ref)
        if record["producer_context_ref"]["artifact_type"] == "validation_case":
            if mutation == "missing": raise FoundationFailure("ARTIFACT_NOT_FOUND")
            if mutation == "digest": record["digest"] = "sha256:" + "f" * 64
            if mutation == "producer": record["producer_context_ref"] = discovery["trial_declaration_ref"]
            if mutation == "count": record["target_stream"]["events"] = []
            if mutation == "discovery": return load(discovery["target_stream_ref"])
        return record
    monkeypatch.setattr(port, "load_target", corrupt)
    with pytest.raises((ValueError, RuntimeError, FoundationFailure)):
        _view(f, port, result.validation_plan_ref)


@pytest.mark.parametrize("outcome", [NoReport(None, "CANDIDATE_PROVENANCE_INVALID"),
    NoReport(ArtifactRef("validation_plan", 1, "sha256:" + "a" * 64), "PORT_RETENTION_UNAVAILABLE"),
    PublishedValidationReport(ArtifactRef("validation_plan", 2, "sha256:" + "b" * 64), ArtifactRef("validation_report", 2, "sha256:" + "c" * 64))])
def test_return_descriptor_is_explicitly_not_persisted_evidence(outcome):
    value = json.loads(describe_validation_return(outcome))
    assert set(value) == {"outcome_type", "plan_ref", "report_ref", "reason_code", "basis"}
    assert value["basis"] == "in_process_observation"
    assert value["outcome_type"] == type(outcome).__name__
    assert value["reason_code"] == (outcome.reason_code if type(outcome) is NoReport else None)
    assert value["plan_ref"] == (shell._wire(outcome.validation_plan_ref) if outcome.validation_plan_ref else None)
    with pytest.raises(TypeError):
        describe_validation_return(value)


@pytest.mark.parametrize("target_mode", [False, True], ids=["ordinary", "target"])
def test_admission_blocked_has_snapshot_anchor_not_oos_consumption(tmp_path, monkeypatch, target_mode):
    from crypto_quant_domain import canonical_sha256
    from crypto_quant_validation import SampleConsumptionRecord
    context = target._runtime(tmp_path) if target_mode else shell._setup(tmp_path)
    f, ledger = context[:2]
    port = context[3] if target_mode else context[2]
    policy = target._policy() if target_mode else context[4]
    previous_case = ArtifactRef("validation_case", 1, "sha256:" + "f" * 64)
    ledger.reserve(SampleConsumptionRecord(policy.holdout.dataset_revision, policy.holdout.interval_start,
        policy.holdout.interval_end, "validation", canonical_sha256(("sample-consumer-v1", previous_case)),
        target._RESEARCH.RESERVED_AT if target_mode else shell.RESERVED_AT), previous_case)
    result = (_target_validate if target_mode else _validate)(context)
    before = _frozen(tmp_path)
    _readonly(monkeypatch, f, ledger, port, context[2] if target_mode else None)
    view = _view(f, port, result.validation_plan_ref)
    assert view["observation_anchor"]["kind"] == "plan_snapshot"
    assert view["case_rows"][0]["outcome"] == view["case_rows"][1]["outcome"] == "BLOCKED"
    assert view["case_rows"][1]["evidence"] is None
    assert view["target_evidence_ref"] is None
    assert view["original_result"]["value"] == "inconclusive"
    assert _frozen(tmp_path) == before


def test_model_candidate_v2_with_real_backtest_sources_and_explicit_scope(tmp_path, monkeypatch):
    integrated = _fixture("test_integrated_validation")
    f, ledger, port, candidate, _, policy = integrated._model_runtime(tmp_path)
    result = validate_candidate(candidate, policy, {"binding_key": "validation:oos:model"},
        integrated._RESERVED_AT, f, ledger, port, candidate_provenance_verifier=verify_candidate_provenance)
    assert type(result) is PublishedValidationReport
    before = _frozen(tmp_path)
    counts = (port.run_calls, port.source_read_calls, port.oos_prepare_calls)
    _readonly(monkeypatch, f, ledger, port)
    view = _view(f, port, result.validation_plan_ref)
    assert view["plan_schema_version"] == view["report_schema_version"] == 1
    assert view["candidate_ref"]["schema_version"] == 2
    assert view["original_result"]["value"] == "rejected"
    row = view["scope"]["rows"][0]
    assert row["model_artifact_ref"] is not None and row["target_recipe_ref"] is None
    assert len(view["scope"]["training_slices"]) == len(view["scope"]["evaluation_slices"]) == 1
    assert row["data_slice"]["interval_start"] == policy.holdout.interval_start
    assert counts == (port.run_calls, port.source_read_calls, port.oos_prepare_calls)
    assert _frozen(tmp_path) == before


@pytest.mark.parametrize("target_mode", [False, True], ids=["ordinary", "target"])
def test_exact_nested_protocol_and_resolvable_source_pointers(tmp_path, target_mode):
    context = target._runtime(tmp_path) if target_mode else shell._setup(tmp_path)
    f = context[0]
    port = context[3] if target_mode else context[2]
    result = (_target_validate if target_mode else _validate)(context)
    view = _view(f, port, result.validation_plan_ref)
    assert set(view) == {"view_format_version", "view_kind", "plan_ref", "plan_schema_version", "candidate_ref",
        "report_ref", "report_schema_version", "original_result", "case_rows", "threshold_evaluations",
        "sample_integrity_refs", "target_evidence_ref", "scope", "observation_anchor", "source_log_heads", "publication_facts", "limitations"}
    assert set(view["scope"]) == {"rows", "training_slices", "evaluation_slices"}
    assert set(view["scope"]["rows"][0]) == {"trial_ref", "analysis_task_ref", "case_ref", "strategy_ref", "parameters",
        "data_slice", "scenario_ref", "metric_profile_ref", "model_artifact_ref", "target_recipe_ref"}
    assert type(view["scope"]["rows"][0]["parameters"]) is dict
    assert [r["case_type"] for r in view["case_rows"]] == ["evidence_integrity", "out_of_sample"]
    for row in view["case_rows"]:
        assert set(row) == {"case_type", "case_ref", "case_result_ref", "outcome", "reason_codes", "limitations", "evidence", "target_evidence_ref", "source_fields"}
        original = shell._payload(f, runtime._ref_from_wire(row["case_result_ref"]))
        assert row["outcome"] == original["outcome"] and row["evidence"] == original["evidence"]
        assert [s["pointer"] for s in row["source_fields"]] == ["/evidence", "/outcome"]
    threshold = view["threshold_evaluations"][0]
    assert set(threshold) == {"case_ref", "metric_profile_ref", "metric", "rule", "passed", "trade_count"}
    assert set(threshold["metric"]) == {"metric_key", "unit", "metric_profile_ref", "fact"}
    assert set(threshold["rule"]) == {"operator", "threshold", "minimum_trade_count"}
    assert threshold["case_ref"] == view["case_rows"][1]["case_ref"]
    for fact in [view["original_result"], threshold["metric"]["fact"], threshold["passed"], threshold["trade_count"]["fact"]]:
        assert set(fact) == {"value", "source_refs", "source_fields", "basis"}
        assert fact["basis"] == "owner_evidence"
        assert len(fact["source_refs"]) == len(fact["source_fields"]) == 1
        source = fact["source_fields"][0]
        assert set(source) == {"source_ref", "document_kind", "pointer"}
        assert source["source_ref"] == fact["source_refs"][0] == view["report_ref"]
        value = shell._payload(f, runtime._ref_from_wire(source["source_ref"]))
        for step in source["pointer"].split("/")[1:]:
            value = value[int(step)] if type(value) is list else value[step]
        assert type(fact["value"]) is type(value) and fact["value"] == value
    assert set(view["source_log_heads"]) == {shell.ARTIFACT_LOG, shell.RESEARCH_ARTIFACT_LOG, shell.RESEARCH_EXECUTION_LOG, shell.SAMPLE_LOG}
    for log, head in view["source_log_heads"].items():
        assert head == runtime._entry_ref_wire(f.entries(log)[-1].entry_ref)
    facts = view["publication_facts"]
    assert facts == sorted(facts, key=lambda p: (p["entry_ref"]["log_name"], p["entry_ref"]["log_sequence"]))
    for fact in facts:
        ref = fact["entry_ref"]
        entry = f.entries(ref["log_name"])[ref["log_sequence"] - 1]
        assert fact["accepted_at"] == entry.accepted_at
        assert fact["subject_ref"] == shell._wire(ArtifactRef.from_envelope(f.read(ref=runtime._ref_from_wire(fact["subject_ref"])).envelope))
    assert read_validation_outcome_view(result.validation_plan_ref, foundation=f, backtest=port,
        candidate_provenance_verifier=verify_candidate_provenance) == canonical_bytes(view)


@pytest.mark.parametrize("target_mode", [False, True], ids=["ordinary", "target"])
def test_old_execution_replay_and_new_view_agree_without_republication(tmp_path, target_mode):
    context = target._runtime(tmp_path) if target_mode else shell._setup(tmp_path)
    validate = _target_validate if target_mode else _validate
    first = validate(context)
    before = _frozen(tmp_path)
    view = _view(context[0], context[3] if target_mode else context[2], first.validation_plan_ref)
    assert validate(context) == first
    assert view["original_result"]["value"] == shell._payload(context[0], first.validation_report_ref)["result"]
    assert _frozen(tmp_path) == before


def test_unknown_plan_version_is_rejected_before_source_io(tmp_path, monkeypatch):
    f, ledger, port, _, _ = shell._setup(tmp_path)
    def forbidden(*args, **kwargs): pytest.fail("invalid version attempted source I/O")
    monkeypatch.setattr(f, "entries", forbidden)
    with pytest.raises(ValueError):
        _view(f, port, ArtifactRef("validation_plan", 3, "sha256:" + "a" * 64))


def test_occupied_foundation_lock_is_preserved(tmp_path):
    f, _, port, _, _ = context = shell._setup(tmp_path)
    result = _validate(context)
    lock = tmp_path / ".foundation.write.lock"
    lock.write_bytes(b"owned by another writer")
    with pytest.raises(FoundationFailure) as caught:
        _view(f, port, result.validation_plan_ref)
    assert caught.value.code == "WRITE_LOCK_UNAVAILABLE"
    assert lock.read_bytes() == b"owned by another writer"


def test_source_change_during_read_has_no_view_or_retry(tmp_path):
    f, _, port, _, _ = context = shell._setup(tmp_path)
    result = _validate(context)
    calls = 0
    def changing(candidate, foundation, backtest):
        nonlocal calls
        calls += 1
        value = verify_candidate_provenance(candidate, foundation, backtest)
        runtime._publish(foundation, "validation_case", {"validation_plan_ref": shell._wire(ArtifactRef("validation_plan", 1, "sha256:" + "f" * 64)), "case_type": "out_of_sample"})
        return value
    with pytest.raises(ValueError) as caught:
        read_validation_outcome_view(result.validation_plan_ref, foundation=f, backtest=port, candidate_provenance_verifier=changing)
    assert caught.value.code == "REPORT_SOURCE_CHANGED" and calls == 1


@pytest.mark.parametrize("target_source", [False, True], ids=["oos_terminal", "oos_target"])
def test_provider_and_foundation_load_errors_are_not_scientific_results(tmp_path, monkeypatch, target_source):
    context = target._runtime(tmp_path) if target_source else shell._setup(tmp_path)
    f = context[0]
    port = context[3] if target_source else context[2]
    result = _target_validate(context) if target_source else _validate(context, "terminal_blocked")
    failure = FoundationFailure("ARTIFACT_NOT_FOUND")
    original = port.load_target if target_source else port.load_terminal
    def fail(ref):
        if not target_source or original(ref)["producer_context_ref"]["artifact_type"] == "validation_case":
            raise failure
        return original(ref)
    monkeypatch.setattr(port, "load_target" if target_source else "load_terminal", fail)
    with pytest.raises(FoundationFailure) as caught:
        _view(f, port, result.validation_plan_ref)
    assert caught.value is failure


@pytest.mark.parametrize("status", ["BLOCKED", "CANCELLED", "FAILED", "provider_failure"])
def test_target_terminal_and_provider_failure_results_are_not_profit_or_exceptions(tmp_path, monkeypatch, status):
    context = target._runtime(tmp_path)
    f, ledger, materializer, port, _ = context
    if status == "provider_failure":
        def fail(*args): raise RuntimeError("synthetic provider failure")
        with monkeypatch.context() as scoped:
            scoped.setattr(port.delegate, "run", fail)
            result = _target_validate(context)
    else:
        port.delegate.terminal_status = status
        result = _target_validate(context)
    _readonly(monkeypatch, f, ledger, port, materializer)
    view = _view(f, port, result.validation_plan_ref)
    if status in ("FAILED", "provider_failure"):
        assert view["report_ref"] is None
        assert view["case_rows"][1]["outcome"] == "FAILED"
    else:
        assert view["original_result"]["value"] == "inconclusive"
    assert view["threshold_evaluations"] == []
    assert view["observation_anchor"]["kind"] == "oos_reservation"


@pytest.mark.parametrize("mutation", ["request", "recipe", "candidate", "event_count", "discovery"])
def test_original_validation_target_evidence_links_must_bind_exact_oos_request(tmp_path, monkeypatch, mutation):
    context = target._runtime(tmp_path)
    f, _, _, port, candidate = context
    publish = target_runtime._publish
    def forge(foundation, kind, version, payload):
        payload = deepcopy(payload)
        if kind == "validation_target_materialization_evidence":
            if mutation == "event_count": payload["event_count"] += 1
            elif mutation == "discovery":
                c = shell._payload(f, candidate.strategy_candidate_ref)
                evidence = shell._payload(f, runtime._ref_from_wire(c["selected_target_materialization_evidence_ref"]))
                payload["target_stream_ref"] = evidence["target_stream_ref"]
            else:
                key = {"request": "materialization_request_hash", "recipe": "target_recipe_ref", "candidate": "candidate_ref"}[mutation]
                if mutation == "request": payload[key] = "sha256:" + "f" * 64
                else: payload[key]["content_hash"] = "sha256:" + "f" * 64
            publish(foundation, kind, version, payload)
            raise KeyboardInterrupt("after forged committed evidence")
        return publish(foundation, kind, version, payload)
    with monkeypatch.context() as scoped:
        scoped.setattr(target_runtime, "_publish", forge)
        with pytest.raises(KeyboardInterrupt): _target_validate(context)
    plan_ref = next(ref for ref, _ in runtime._artifact_log_payloads(f) if ref.artifact_type == "validation_plan")
    with pytest.raises((ValueError, RuntimeError)):
        _view(f, port, plan_ref)


@pytest.mark.parametrize("kind", ["validation_case_result", "validation_report"])
def test_target_v2_never_falls_back_to_v1_evidence(tmp_path, monkeypatch, kind):
    context = target._runtime(tmp_path)
    publish = target_runtime._publish
    def fallback(foundation, artifact_type, version, payload):
        if artifact_type == kind:
            publish(foundation, artifact_type, 1, payload)
            raise KeyboardInterrupt("wrong version publication")
        return publish(foundation, artifact_type, version, payload)
    with monkeypatch.context() as scoped:
        scoped.setattr(target_runtime, "_publish", fallback)
        with pytest.raises(KeyboardInterrupt): _target_validate(context)
    plan_ref = next(ref for ref, _ in runtime._artifact_log_payloads(context[0]) if ref.artifact_type == "validation_plan")
    with pytest.raises((ValueError, RuntimeError)):
        _view(context[0], context[3], plan_ref)


def test_other_holdout_assessment_for_same_snapshot_does_not_impersonate_this_plan(tmp_path, monkeypatch):
    f, ledger, port, _, policy = context = shell._setup(tmp_path)
    plan_ref = _prefix(context, monkeypatch, "admission")
    snapshot_ref = runtime._ref_from_wire(shell._payload(f, plan_ref)["sample_consumption_snapshot_ref"])
    other = replace(policy.holdout, dataset_revision="different-dataset")
    foreign = ledger.assess_holdout(snapshot_ref, other)
    view = _view(f, port, plan_ref)
    assert view["sample_integrity_refs"]["assessment_ref"] != shell._wire(foreign)
    assert view["case_rows"][0]["outcome"] == "PASS"
