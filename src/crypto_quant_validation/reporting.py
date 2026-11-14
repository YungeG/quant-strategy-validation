"""Readonly projections of published Validation evidence; never an execution entry point."""
from __future__ import annotations

from dataclasses import fields
from typing import Any

from crypto_quant_domain import ArtifactRef, canonical_bytes
from crypto_quant_foundation import FoundationFailure, LocalFoundation

from . import runtime as v1
from . import target_runtime as v2
from .integration import (
    DecisionRule, Holdout, OosRule, ResolvedArtifact, SampleAdmissionEvidence,
    ValidationPolicy, assess_admission, build_validation_plan,
)
from .ledger import _append_record, _snapshot_checkpoint
from .sample_consumption import SampleConsumptionRecord, assess_untouched_holdout, build_snapshot

_LOGS = (v1._RESEARCH_ARTIFACT_LOG, v1._RESEARCH_EXECUTION_LOG, v1._ARTIFACT_LOG, v1._SAMPLE_LOG)
_CASE_TYPES = ("evidence_integrity", "out_of_sample")
_INVALID = "VALIDATION_OUTCOME_VIEW_INVALID"


def _invalid() -> None:
    raise v1._GraphFailure(_INVALID)


def _heads(foundation: LocalFoundation) -> dict[str, object]:
    return {
        log: v1._entry_ref_wire(entries[-1].entry_ref) if entries else None
        for log in _LOGS for entries in (foundation.entries(log),)
    }


def _source(ref: object, pointer: str) -> dict[str, object]:
    return {"source_ref": ref, "document_kind": "artifact_payload", "pointer": pointer}


def _fact(value: object, ref: object, pointer: str) -> dict[str, object]:
    return {"value": value, "source_refs": [ref], "source_fields": [_source(ref, pointer)], "basis": "owner_evidence"}


def _snapshot(foundation: LocalFoundation, snapshot_ref: ArtifactRef, policy: ValidationPolicy):
    payload = v1._published(foundation, snapshot_ref, "sample_consumption_ledger_snapshot", v1._ARTIFACT_LOG)
    checkpoint = _snapshot_checkpoint(payload)
    entries = foundation.entries(v1._SAMPLE_LOG, checkpoint)
    records = tuple(_append_record(e.payload, e.event_id, e.accepted_at) for e in entries)
    snapshot = build_snapshot(records, as_of=checkpoint.as_of)
    integrity = assess_untouched_holdout(
        snapshot, dataset_revision=policy.holdout.dataset_revision,
        interval_start=policy.holdout.interval_start, interval_end=policy.holdout.interval_end,
    )
    return SampleAdmissionEvidence(v1._wire(snapshot_ref), v1._checkpoint_wire(checkpoint), snapshot, integrity)


def _read_observation_anchor(plan_ref: ArtifactRef, plan_payload: dict[str, Any], foundation: LocalFoundation) -> tuple[str, str, object]:
    """Select an original reservation, or the original issued snapshot (not consumption)."""
    snapshot_ref = v1._ref_from_wire(plan_payload["sample_consumption_snapshot_ref"])
    snapshot = v1._published(foundation, snapshot_ref, "sample_consumption_ledger_snapshot", v1._ARTIFACT_LOG)
    checkpoint = _snapshot_checkpoint(snapshot)
    # An issued checkpoint is required even if a reservation is also present.
    for entry in foundation.entries(v1._SAMPLE_LOG, checkpoint):
        _append_record(entry.payload, entry.event_id, entry.accepted_at)
    oos_case = v1._case_ref(plan_ref, "out_of_sample")
    matches = []
    for entry in foundation.entries(v1._SAMPLE_LOG):
        record = _append_record(entry.payload, entry.event_id, entry.accepted_at)
        _, producer = v1._sample_entry(entry.payload)
        if producer == oos_case:
            matches.append((entry, record))
    if not matches:
        return "plan_snapshot", checkpoint.as_of, snapshot_ref
    if len(matches) != 1:
        _invalid()
    entry, record = matches[0]
    holdout = plan_payload["holdout"]
    expected = SampleConsumptionRecord(
        holdout["dataset_revision"], holdout["interval_start"], holdout["interval_end"],
        "validation", v1._consumer_id(oos_case), record.consumed_at,
    )
    if record != expected:
        _invalid()
    # The raw sample append, not a synthesized expected record, is the citation.
    return "oos_reservation", record.consumed_at, entry.entry_ref


def _policy(payload: dict[str, Any]) -> ValidationPolicy:
    decision = dict(payload["decision_rule"])
    decision["required_case_types"] = tuple(decision["required_case_types"])
    return ValidationPolicy(
        tuple(payload["accepted_backtest_grades"]), tuple(payload["accepted_metric_profile_refs"]),
        Holdout(**payload["holdout"]), OosRule(**payload["oos_rule"]), DecisionRule(**decision),
    )


def _observed_cases(plan_ref, foundation, publications, version):
    rows = []
    for case_type in _CASE_TYPES:
        expected_ref = v1._case_ref(plan_ref, case_type)
        declarations = [(ref, payload) for _, ref, payload in publications if ref == expected_ref]
        results = [(ref, payload) for _, ref, payload in publications
                   if ref.artifact_type == "validation_case_result" and payload.get("case_ref") == v1._wire(expected_ref)]
        if len(declarations) > 1 or len(results) > 1 or (results and not declarations):
            _invalid()
        case_ref = expected_ref if declarations else None
        if declarations:
            if v1._published(foundation, expected_ref, "validation_case", v1._ARTIFACT_LOG) != {
                "validation_plan_ref": v1._wire(plan_ref), "case_type": case_type,
            }:
                _invalid()
        result = results[0] if results else None
        if result:
            ref, payload = result
            if ref.schema_version != version or v1._published(
                foundation, ref, "validation_case_result", v1._ARTIFACT_LOG, (version,),
            ) != payload:
                _invalid()
        rows.append((case_type, case_ref, result))
    return rows


def _assessment(foundation, publications, snapshot_ref, policy, report):
    matches = [(ref, payload) for _, ref, payload in publications
               if ref.artifact_type == "sample_integrity_assessment"
               and payload.get("snapshot_ref") == v1._wire(snapshot_ref)
               and all(payload.get(key) == getattr(policy.holdout, key)
                       for key in ("dataset_revision", "interval_start", "interval_end"))]
    if len(matches) > 1:
        _invalid()
    ref = matches[0][0] if matches else None
    if report:
        linked = v1._ref_from_wire(report[1]["sample_integrity_ref"])
        # Exact-read referenced artifacts even when not among the discovered entries.
        v1._published(foundation, linked, "sample_integrity_assessment", v1._ARTIFACT_LOG)
        if linked != ref:
            _invalid()
    if ref is not None:
        evidence = v1._snapshot_evidence(foundation, snapshot_ref, ref, policy)
        if evidence.ledger_conflict:
            _invalid()
        return ref, evidence
    return None, _snapshot(foundation, snapshot_ref, policy)


def _verify_cases(plan_ref, plan, foundation, backtest, graph, evidence, rows, version, target_ref):
    admission = assess_admission(plan, graph, evidence)
    for case_type, case_ref, result in rows:
        if result is None:
            continue
        ref, payload = result
        if case_type == "evidence_integrity":
            recovered = admission
        else:
            if rows[0][2] is None:
                _invalid()
            if admission.outcome != "PASS":
                if target_ref is not None:
                    _invalid()
                recovered = v1._blocked_oos(plan, admission)
            elif version == 2:
                if target_ref is None:
                    _invalid()
                v2._existing_case_result(foundation, case_ref, target_ref, plan_ref, case_type)
                v2._verify_completed_oos_payload(plan, case_ref, target_ref, payload, backtest)
                continue
            else:
                recovered = v1._replay_oos_result(plan, case_ref, payload, backtest)
        expected = (v1._case_result_payload(case_ref, recovered) if version == 1
                    else v2._case_result_payload(case_ref, recovered, None))
        if canonical_bytes(expected) != canonical_bytes(payload):
            _invalid()
    return admission


def _read_v1_view(plan_ref, payload, foundation, backtest, verifier, anchor):
    policy = _policy(payload)
    candidate_ref = v1._ref_from_wire(payload["candidate_ref"])
    if candidate_ref.schema_version not in (1, 2):
        _invalid()
    snapshot_ref = v1._ref_from_wire(payload["sample_consumption_snapshot_ref"])
    plan = build_validation_plan(v1._wire(candidate_ref), v1._wire(snapshot_ref), policy)
    if v1._plan_payload(plan) != payload:
        _invalid()
    graph, _ = v1._candidate_graph(candidate_ref, foundation, backtest, anchor[1], candidate_provenance_verifier=verifier)
    return _reconstruct(plan_ref, payload, policy, plan, graph, None, foundation, backtest, anchor)


def _readonly_verified_target(backtest, target_ref, producer_context_ref, **expected):
    # Keep loader/infrastructure exceptions outside the pure identity validator.
    target_ref = v2._canonical_target_ref(target_ref)
    loaded = backtest.load_target(v1._plain(target_ref))
    return v2._verify_loaded_target(loaded, target_ref, producer_context_ref, **expected)


def _read_target_v2_view(plan_ref, payload, foundation, backtest, verifier, anchor):
    policy = _policy(payload)
    candidate_ref = v1._ref_from_wire(payload["candidate_ref"])
    if candidate_ref.schema_version != 3:
        _invalid()
    graph, _, context = v2._target_context_with_loader(
        candidate_ref, foundation, backtest, anchor[1], verifier, _readonly_verified_target,
    )
    snapshot_ref = v1._ref_from_wire(payload["sample_consumption_snapshot_ref"])
    plan, expected = v2._target_plan_payload(candidate_ref, snapshot_ref, policy, context)
    if expected != payload:
        _invalid()
    return _reconstruct(plan_ref, payload, policy, plan, graph, context, foundation, backtest, anchor)


def _reconstruct(plan_ref, payload, policy, plan, graph, context, foundation, backtest, anchor):
    version = plan_ref.schema_version
    publications = v2._entries(foundation, v1._ARTIFACT_LOG)
    rows = _observed_cases(plan_ref, foundation, publications, version)
    oos_case = v1._case_ref(plan_ref, "out_of_sample")
    target_matches = [(ref, value) for _, ref, value in publications
                      if ref.artifact_type == "validation_target_materialization_evidence"
                      and value.get("validation_case_ref") == v1._wire(oos_case)]
    if len(target_matches) > 1 or any(ref.schema_version != 1 for ref, _ in target_matches):
        _invalid()
    result_wires = [v1._wire(row[2][0]) for row in rows if row[2]]
    reports = [(ref, value) for _, ref, value in publications if ref.artifact_type == "validation_report" and (
        value.get("validation_plan_ref") == v1._wire(plan_ref)
        or any(w in value.get("case_result_refs", []) for w in result_wires)
        or any(value.get("validation_target_materialization_evidence_ref") == v1._wire(r) for r, _ in target_matches)
    )]
    if len(reports) > 1 or (reports and reports[0][0].schema_version != version):
        _invalid()
    report = reports[0] if reports else None
    snapshot_ref = v1._ref_from_wire(payload["sample_consumption_snapshot_ref"])
    assessment_ref, evidence = _assessment(foundation, publications, snapshot_ref, policy, report)
    admission = assess_admission(plan, graph, evidence)
    oos_result = rows[1][2]
    needs_reservation = bool(target_matches) or (oos_result is not None and admission.outcome == "PASS")
    if needs_reservation and anchor[0] != "oos_reservation":
        _invalid()
    if anchor[0] == "oos_reservation":
        record = SampleConsumptionRecord(plan.holdout.dataset_revision, plan.holdout.interval_start,
                                        plan.holdout.interval_end, "validation", v1._consumer_id(oos_case), anchor[1])
        bounds = [ref for ref, _ in target_matches] + ([oos_result[0]] if oos_result else [])
        ledger = v1.SampleConsumptionLedger(foundation)
        ledger.require_reservation(record, oos_case)
        for ref in bounds:
            entries = [entry for entry, seen, _ in publications if seen == ref]
            if len(entries) != 1:
                _invalid()
            ledger.require_reservation(record, oos_case, before_entry_ref=entries[0].entry_ref)
    target_ref = None
    if context is not None:
        recovered = v2._existing_evidence_metadata(
            foundation, v1._ref_from_wire(payload["candidate_ref"]), oos_case, policy, context, anchor[1],
        )
        target_ref = recovered[0] if recovered else None
        if [ref for ref, _ in target_matches] != ([target_ref] if target_ref is not None else []):
            _invalid()
        if recovered:
            target = recovered[1]
            _readonly_verified_target(
                backtest, target.target_stream_ref, oos_case,
                expected_digest=target.target_stream_digest, expected_event_count=target.event_count,
            )
        if target_ref is not None and (admission.outcome != "PASS" or rows[0][2] is None or rows[1][1] is None):
            _invalid()
    elif target_matches:
        _invalid()
    _verify_cases(plan_ref, plan, foundation, backtest, graph, evidence, rows, version, target_ref)
    if report:
        if context is None:
            verified = v1._existing_report(foundation, plan_ref, plan, graph, snapshot_ref, policy, backtest)
            if verified is None or verified.validation_report_ref != report[0]:
                _invalid()
        else:
            if rows[0][2] is None or rows[1][2] is None or assessment_ref is None:
                _invalid()
            admission_ref, admission_payload = rows[0][2]
            oos_ref, oos_payload = rows[1][2]
            if target_ref is not None:
                expected = v2._report_payload_from_results(
                    plan_ref, admission_ref, admission_payload, oos_ref, oos_payload, assessment_ref, target_ref,
                )
            elif admission.outcome != "PASS" and admission.outcome != "FAILED":
                expected = {
                    "validation_plan_ref": v1._wire(plan_ref), "result": "inconclusive",
                    "case_result_refs": [v1._wire(admission_ref), v1._wire(oos_ref)],
                    "threshold_evaluations": [], "sample_integrity_ref": v1._wire(assessment_ref),
                    "limitations": sorted(set(admission_payload["limitations"]) | set(oos_payload["limitations"])),
                    "validation_target_materialization_evidence_ref": None,
                }
            else:
                _invalid()
            published = v1._published(foundation, report[0], "validation_report", v1._ARTIFACT_LOG, (2,))
            if canonical_bytes(published) != canonical_bytes(expected):
                _invalid()
    return _project(plan_ref, payload, graph, rows, snapshot_ref, assessment_ref, target_ref, report, anchor, foundation)


def _project(plan_ref, payload, graph, rows, snapshot_ref, assessment_ref, target_ref, report, anchor, foundation):
    case_rows = []
    for case_type, case_ref, result in rows:
        ref, value = result if result else (None, {})
        wire = v1._wire(ref) if ref else None
        case_rows.append({
            "case_type": case_type, "case_ref": v1._wire(case_ref) if case_ref else None,
            "case_result_ref": wire, "outcome": value.get("outcome"),
            "reason_codes": value.get("reason_codes", []), "limitations": value.get("limitations", []),
            "evidence": value.get("evidence"),
            "target_evidence_ref": value.get("validation_target_materialization_evidence_ref"),
            "source_fields": [_source(wire, "/evidence"), _source(wire, "/outcome")] if ref else [],
        })
    thresholds = []
    rule = payload["oos_rule"]
    report_wire = v1._wire(report[0]) if report else None
    for i, value in enumerate(report[1]["threshold_evaluations"] if report else []):
        prefix = f"/threshold_evaluations/{i}"
        def measurement(key, unit, field):
            return {"metric_key": key, "unit": unit, "metric_profile_ref": rule["metric_profile_ref"],
                    "fact": _fact(value[field], report_wire, prefix + "/" + field)}
        thresholds.append({
            "case_ref": case_rows[1]["case_ref"], "metric_profile_ref": rule["metric_profile_ref"],
            "metric": measurement(rule["metric_key"], rule["unit"], "observed"),
            "rule": {key: rule[key] for key in ("operator", "threshold", "minimum_trade_count")},
            "passed": _fact(value["passed"], report_wire, prefix + "/passed"),
            "trade_count": measurement("trade_count", "count", "trade_count"),
        })
    experiment_ref = v1._ref_from_wire(graph.candidate_family.payload["experiment_ref"])
    experiment = v1._published(foundation, experiment_ref, "experiment_spec", v1._RESEARCH_ARTIFACT_LOG, (1, 2, 3))
    trial = graph.selected_trial_declaration.payload
    parameters = trial["parameter_values"]
    if type(parameters) is list:
        parameters = dict(parameters)
    scope = {"rows": [{
        "trial_ref": graph.selected_trial_declaration.ref, "analysis_task_ref": None,
        "case_ref": case_rows[1]["case_ref"], "strategy_ref": experiment["strategy_definition_ref"],
        "parameters": parameters, "data_slice": {key: payload["holdout"][key] for key in (
            "market_bundle_ref", "dataset_revision", "interval_start", "interval_end")},
        "scenario_ref": None, "metric_profile_ref": rule["metric_profile_ref"],
        "model_artifact_ref": graph.model_build.model_build_evidence.payload["model_artifact"] if graph.model_build else None,
        "target_recipe_ref": payload.get("target_recipe_ref"),
    }], "training_slices": [graph.model_build.model_build_plan.payload["training_slice"]] if graph.model_build else [],
        "evaluation_slices": experiment.get("evaluation_slices", [])}
    kind, instant, coordinate = anchor
    anchor_wire = v1._wire(coordinate) if kind == "plan_snapshot" else v1._entry_ref_wire(coordinate)
    refs = {plan_ref, snapshot_ref, experiment_ref}
    for ref in (assessment_ref, target_ref, report[0] if report else None):
        if ref is not None:
            refs.add(ref)
    for _, case_ref, result in rows:
        if case_ref:
            refs.add(case_ref)
        if result:
            refs.add(result[0])
    if plan_ref.schema_version == 2:
        recipe_ref = v1._ref_from_wire(payload["target_recipe_ref"])
        candidate = v1._published(
            foundation, v1._ref_from_wire(payload["candidate_ref"]), "strategy_candidate", v1._RESEARCH_ARTIFACT_LOG, (3,),
        )
        discovery_ref = v1._ref_from_wire(candidate["selected_target_materialization_evidence_ref"])
        discovery = v1._published(foundation, discovery_ref, "target_materialization_evidence", v1._RESEARCH_ARTIFACT_LOG)
        refs.update((recipe_ref, discovery_ref, v1._ref_from_wire(discovery["target_build_task_ref"])))
    for node in (graph, graph.model_build):
        if node is not None:
            for field in fields(node):
                value = getattr(node, field.name)
                if type(value) is ResolvedArtifact:
                    refs.add(v1._ref_from_wire(v1._plain(value.ref)))
    publications = []
    for log in _LOGS[:-1]:
        for entry, ref, _ in v2._entries(foundation, log):
            if ref in refs:
                # Do not claim a log-only orphan as a verified artifact.
                if foundation.read(ref=ref).source_bytes != entry.payload:
                    _invalid()
                publications.append({"subject_ref": v1._wire(ref), "entry_ref": v1._entry_ref_wire(entry.entry_ref), "accepted_at": entry.accepted_at})
    limitations = list(report[1]["limitations"]) if report else []
    additions = [
        "Internal publication is not independently verified external publication.",
        "Scenario/cost assumptions are not resolved by the Validation plan.",
        "Only declared threshold metrics are presented; other economics and significance are not computed.",
    ]
    if report is None:
        additions.append("No published report is witnessed; the original in-process NoReport reason is not recoverable.")
    if kind == "plan_snapshot":
        additions.append("Snapshot observation time is not OOS consumed_at and does not prove absence of uninstrumented reads.")
    limitations.extend(sorted(set(additions) - set(limitations)))
    return {
        "view_format_version": 1, "view_kind": "validation_outcome", "plan_ref": v1._wire(plan_ref),
        "plan_schema_version": plan_ref.schema_version, "candidate_ref": payload["candidate_ref"],
        "report_ref": report_wire, "report_schema_version": report[0].schema_version if report else None,
        "original_result": _fact(report[1]["result"], report_wire, "/result") if report else None,
        "case_rows": case_rows, "threshold_evaluations": thresholds,
        "sample_integrity_refs": {"snapshot_ref": v1._wire(snapshot_ref), "assessment_ref": v1._wire(assessment_ref) if assessment_ref else None},
        "target_evidence_ref": v1._wire(target_ref) if target_ref else None, "scope": scope,
        "observation_anchor": {"kind": kind, "value": instant, "source_ref": anchor_wire,
            "source_fields": [_source(anchor_wire, "/checkpoint/as_of" if kind == "plan_snapshot" else "/record/consumed_at")]},
        "source_log_heads": {}, "publication_facts": sorted(publications, key=lambda p: (p["entry_ref"]["log_name"], p["entry_ref"]["log_sequence"])),
        "limitations": limitations,
    }


def read_validation_outcome_view(plan_ref: ArtifactRef, *, foundation: LocalFoundation, backtest: object,
                                 candidate_provenance_verifier: v1.CandidateProvenanceVerifier) -> bytes:
    """Verify a frozen v1/v2 graph and return canonical presentation bytes without publication."""
    plan_ref = v1._artifact_ref(plan_ref, "plan_ref")
    if plan_ref.artifact_type != "validation_plan" or plan_ref.schema_version not in (1, 2):
        raise ValueError("plan_ref must reference exact ValidationPlan@1 or @2")
    if type(foundation) is not LocalFoundation or not callable(candidate_provenance_verifier):
        raise TypeError("foundation and candidate_provenance_verifier are required")
    before = _heads(foundation)
    try:
        payload = v1._published(foundation, plan_ref, "validation_plan", v1._ARTIFACT_LOG, (plan_ref.schema_version,))
        anchor = _read_observation_anchor(plan_ref, payload, foundation)
        reader = _read_v1_view if plan_ref.schema_version == 1 else _read_target_v2_view
        view = reader(plan_ref, payload, foundation, backtest, candidate_provenance_verifier, anchor)
    except FoundationFailure:
        raise
    except v1._GraphFailure as error:
        if error.code == "CANDIDATE_PROVENANCE_INVALID":
            raise v1._GraphFailure(_INVALID) from error
        raise
    if before != _heads(foundation):
        raise v1._GraphFailure("REPORT_SOURCE_CHANGED")
    view["source_log_heads"] = before
    return canonical_bytes(view)


def describe_validation_return(outcome: v1.PublishedValidationReport | v1.NoReport) -> bytes:
    """Describe only an actual process return; this does not authenticate its references."""
    if type(outcome) not in (v1.PublishedValidationReport, v1.NoReport):
        raise TypeError("outcome must be PublishedValidationReport or NoReport")
    plan = outcome.validation_plan_ref
    report = outcome.validation_report_ref if type(outcome) is v1.PublishedValidationReport else None
    return canonical_bytes({
        "outcome_type": type(outcome).__name__, "plan_ref": v1._wire(plan) if plan else None,
        "report_ref": v1._wire(report) if report else None,
        "reason_code": outcome.reason_code if type(outcome) is v1.NoReport else None,
        "basis": "in_process_observation",
    })


__all__ = ["read_validation_outcome_view", "describe_validation_return"]
