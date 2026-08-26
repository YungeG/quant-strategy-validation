from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from crypto_quant_domain import (
    ArtifactEnvelope,
    ArtifactRef,
    canonical_bytes,
    canonical_sha256,
)
from crypto_quant_research import PublishedStrategyCandidate, execute_target_experiment

from crypto_quant_validation import (
    Holdout,
    NoReport,
    OosRule,
    PublishedValidationReport,
    ValidationPolicy,
    ValidationTargetMaterializationEvidence,
    target_runtime,
    validate_candidate,
    validate_target_candidate,
)

_ROOT = Path(__file__).resolve().parents[2]
_RESEARCH_TEST = next(
    path
    for path in (
        _ROOT / "research-platform/tests/test_target_stream_research.py",
        _ROOT / "quant-research-tsr-rp/tests/test_target_stream_research.py",
    )
    if path.is_file()
)
_SPEC = importlib.util.spec_from_file_location(
    "accepted_target_research_fixture", _RESEARCH_TEST
)
assert _SPEC is not None and _SPEC.loader is not None
_RESEARCH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RESEARCH
_SPEC.loader.exec_module(_RESEARCH)

ARTIFACT_LOG = "validation.artifacts.v1"
SAMPLE_LOG = "validation.sample-consumption.v1"
HOLDOUT_START = "2026-03-01T00:00:00.000000Z"
HOLDOUT_END = "2026-04-01T00:00:00.000000Z"


def _payload(foundation, ref: ArtifactRef) -> dict[str, object]:
    return json.loads(foundation.read(ref=ref).source_bytes)["payload"]


def _ref(value: dict[str, object]) -> ArtifactRef:
    return ArtifactRef(
        value["artifact_type"], value["schema_version"], value["content_hash"]
    )


def _policy() -> ValidationPolicy:
    profile = _RESEARCH._artifact_ref("backtest_metric_profile", "6")
    return ValidationPolicy(
        ("development",),
        (profile,),
        Holdout(
            _RESEARCH._tagged_ref(
                "backtest_market_bundle_ref", "backtest_market_bundle", "9"
            ),
            "holdout-v1",
            HOLDOUT_START,
            HOLDOUT_END,
            "HOLDOUT",
            False,
        ),
        OosRule(profile, "simple_period_return", "fraction", "gte", "0", 1),
    )


class _ValidationBacktest:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.preparation_calls = 0
        self.fail_prepare = False
        self.interrupt_after_prepare = False
        self._prepare_interrupted = False

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)

    def prepare_target(self, validation_case_ref: dict, target_ref: dict) -> dict:
        self.preparation_calls += 1
        if self.fail_prepare:
            raise RuntimeError("preparation unavailable")
        request = {
            "validation_case_ref": deepcopy(validation_case_ref),
            "target_ref": deepcopy(target_ref),
        }
        if self.interrupt_after_prepare and not self._prepare_interrupted:
            self._prepare_interrupted = True
            raise KeyboardInterrupt("after preparation")
        return request


def _runtime(tmp_path: Path):
    foundation, ledger, materializer, backtest = _RESEARCH._runtime(tmp_path)
    candidate = execute_target_experiment(
        _RESEARCH._inputs(), foundation, ledger, materializer, backtest
    )
    assert type(candidate) is PublishedStrategyCandidate
    return foundation, ledger, materializer, _ValidationBacktest(backtest), candidate


def _artifact_payloads(foundation, artifact_type: str) -> list[tuple[dict, dict]]:
    return [
        (envelope, envelope["payload"])
        for entry in foundation.entries(ARTIFACT_LOG)
        if (envelope := json.loads(entry.payload))["artifact_type"] == artifact_type
    ]


def test_target_validation_golden_links_full_provenance(
    tmp_path: Path,
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    discovery_candidate = _payload(foundation, candidate.strategy_candidate_ref)
    discovery_evidence = _payload(
        foundation,
        _ref(discovery_candidate["selected_target_materialization_evidence_ref"]),
    )

    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert type(result) is PublishedValidationReport
    assert result.validation_plan_ref.schema_version == 2
    assert result.validation_report_ref.schema_version == 2
    plan = _payload(foundation, result.validation_plan_ref)
    assert set(plan) == {
        "candidate_ref",
        "sample_consumption_snapshot_ref",
        "accepted_backtest_grades",
        "accepted_metric_profile_refs",
        "holdout",
        "oos_rule",
        "decision_rule",
        "target_recipe_ref",
        "strategy_artifact",
    }
    assert plan["target_recipe_ref"] == discovery_evidence["target_recipe_ref"]
    evidence_envelope, evidence = _artifact_payloads(
        foundation, "validation_target_materialization_evidence"
    )[0]
    assert evidence_envelope["schema_version"] == 1
    assert set(evidence) == {
        "validation_case_ref",
        "candidate_ref",
        "target_recipe_ref",
        "materialization_request_hash",
        "input_data_hash",
        "target_stream_ref",
        "target_stream_digest",
        "event_count",
    }
    assert (
        evidence["candidate_ref"]
        == candidate.strategy_candidate_ref.to_canonical_dict()
    )
    assert evidence["target_stream_ref"] != discovery_evidence["target_stream_ref"]
    discovery_loaded = backtest.load_target(discovery_evidence["target_stream_ref"])
    oos_loaded = backtest.load_target(evidence["target_stream_ref"])
    assert canonical_bytes(discovery_loaded["target_stream"]) == canonical_bytes(
        oos_loaded["target_stream"]
    )
    report = _payload(foundation, result.validation_report_ref)
    assert report["validation_target_materialization_evidence_ref"] == {
        "type": "artifact_ref",
        "artifact_type": evidence_envelope["artifact_type"],
        "schema_version": evidence_envelope["schema_version"],
        "content_hash": evidence_envelope["content_hash"],
    }
    assert report["result"] == "supported"
    case_results = _artifact_payloads(foundation, "validation_case_result")
    assert [item[0]["schema_version"] for item in case_results] == [2, 2]
    assert case_results[0][1]["validation_target_materialization_evidence_ref"] is None
    assert (
        case_results[1][1]["validation_target_materialization_evidence_ref"]
        == report["validation_target_materialization_evidence_ref"]
    )
    assert len(foundation.entries(SAMPLE_LOG)) == 3

    old = validate_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        {},
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        backtest,
    )
    assert old == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")


def test_legacy_candidate_v3_rejection_has_no_validation_publication(
    tmp_path: Path,
) -> None:
    foundation, ledger, _, backtest, candidate = _runtime(tmp_path)
    before = (
        foundation.entries(ARTIFACT_LOG),
        foundation.entries(SAMPLE_LOG),
        backtest.run_calls,
    )

    result = validate_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        {},
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        backtest,
    )

    assert result == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
    assert foundation.entries(ARTIFACT_LOG) == before[0] == ()
    assert foundation.entries(SAMPLE_LOG) == before[1]
    assert backtest.run_calls == before[2]


def test_closed_replay_only_revalidates_target_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    governance_calls = {"assess": 0, "reserve": 0, "freeze": 0}
    assess_holdout = ledger.assess_holdout
    reserve = ledger.reserve
    freeze_snapshot = ledger.freeze_snapshot

    def counted_assess(*args, **kwargs):
        governance_calls["assess"] += 1
        return assess_holdout(*args, **kwargs)

    def counted_reserve(*args, **kwargs):
        governance_calls["reserve"] += 1
        return reserve(*args, **kwargs)

    def counted_freeze(*args, **kwargs):
        governance_calls["freeze"] += 1
        return freeze_snapshot(*args, **kwargs)

    monkeypatch.setattr(ledger, "assess_holdout", counted_assess)
    monkeypatch.setattr(ledger, "reserve", counted_reserve)
    monkeypatch.setattr(ledger, "freeze_snapshot", counted_freeze)
    first = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )
    assert governance_calls == {"assess": 1, "reserve": 1, "freeze": 1}
    before = (
        materializer.calls,
        backtest.store_calls,
        backtest.preparation_calls,
        backtest.run_calls,
        backtest.economic_run_calls,
        backtest.cache_calls,
        len(foundation.entries(ARTIFACT_LOG)),
        len(foundation.entries(SAMPLE_LOG)),
        backtest.load_target_calls,
        dict(governance_calls),
    )

    second = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert second == first
    after = (
        materializer.calls,
        backtest.store_calls,
        backtest.preparation_calls,
        backtest.run_calls,
        backtest.economic_run_calls,
        backtest.cache_calls,
        len(foundation.entries(ARTIFACT_LOG)),
        len(foundation.entries(SAMPLE_LOG)),
        backtest.load_target_calls,
        dict(governance_calls),
    )
    assert after[:8] == before[:8]
    assert after[8] == before[8] + 2
    assert after[9] == before[9]


def test_target_cas_orphan_rematerializes_but_runs_economics_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    publish = backtest.delegate.publish_target
    interrupted = False

    def orphan(context: dict, stream: dict):
        nonlocal interrupted
        ref = publish(context, stream)
        if context["artifact_type"] == "validation_case" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("after target CAS")
        return ref

    monkeypatch.setattr(backtest.delegate, "publish_target", orphan)
    with pytest.raises(KeyboardInterrupt):
        validate_target_candidate(
            candidate.strategy_candidate_ref,
            _policy(),
            _RESEARCH.RESERVED_AT,
            foundation,
            ledger,
            materializer,
            backtest,
        )
    monkeypatch.setattr(backtest.delegate, "publish_target", publish)
    validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert materializer.calls == 3
    assert backtest.store_calls == 3
    assert backtest.preparation_calls == 1
    assert backtest.run_calls == backtest.economic_run_calls == 2
    assert backtest.cache_calls == 0


def test_evidence_commit_and_preparation_interruptions_do_not_rematerialize_or_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    publish = target_runtime._publish
    interrupted = False

    def after_evidence(*args, **kwargs):
        nonlocal interrupted
        ref = publish(*args, **kwargs)
        if args[1] == "validation_target_materialization_evidence" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("after evidence commit")
        return ref

    monkeypatch.setattr(target_runtime, "_publish", after_evidence)
    with pytest.raises(KeyboardInterrupt):
        validate_target_candidate(
            candidate.strategy_candidate_ref,
            _policy(),
            _RESEARCH.RESERVED_AT,
            foundation,
            ledger,
            materializer,
            backtest,
        )
    monkeypatch.setattr(target_runtime, "_publish", publish)
    backtest.interrupt_after_prepare = True
    with pytest.raises(KeyboardInterrupt):
        validate_target_candidate(
            candidate.strategy_candidate_ref,
            _policy(),
            _RESEARCH.RESERVED_AT,
            foundation,
            ledger,
            materializer,
            backtest,
        )
    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert type(result) is PublishedValidationReport
    assert materializer.calls == backtest.store_calls == 2
    assert backtest.preparation_calls == 2
    assert backtest.run_calls == backtest.economic_run_calls == 2


def test_committed_evidence_recovery_exact_load_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    publish = target_runtime._publish
    interrupted = False

    def after_evidence(*args, **kwargs):
        nonlocal interrupted
        ref = publish(*args, **kwargs)
        if args[1] == "validation_target_materialization_evidence" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("after evidence commit")
        return ref

    monkeypatch.setattr(target_runtime, "_publish", after_evidence)
    with pytest.raises(KeyboardInterrupt):
        validate_target_candidate(
            candidate.strategy_candidate_ref,
            _policy(),
            _RESEARCH.RESERVED_AT,
            foundation,
            ledger,
            materializer,
            backtest,
        )
    monkeypatch.setattr(target_runtime, "_publish", publish)
    load = backtest.load_target

    def tamper(ref: dict[str, object]):
        loaded = load(ref)
        if loaded["producer_context_ref"]["artifact_type"] == "validation_case":
            loaded["digest"] = _RESEARCH._hash("f")
        return loaded

    monkeypatch.setattr(backtest, "load_target", tamper, raising=False)
    before = (materializer.calls, backtest.store_calls)
    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert type(result) is NoReport
    assert result.reason_code == "TARGET_STORE_INVALID"
    assert before == (materializer.calls, backtest.store_calls)
    assert backtest.preparation_calls == 0


def test_committed_evidence_event_count_recovery_rejects_valid_bound_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    publish = target_runtime._publish

    def forge_count(*args, **kwargs):
        if args[1] == "validation_target_materialization_evidence":
            payload = deepcopy(args[3])
            payload["event_count"] += 1
            publish(args[0], args[1], args[2], payload)
            raise KeyboardInterrupt("after forged evidence commit")
        return publish(*args, **kwargs)

    monkeypatch.setattr(target_runtime, "_publish", forge_count)
    with pytest.raises(KeyboardInterrupt):
        validate_target_candidate(
            candidate.strategy_candidate_ref,
            _policy(),
            _RESEARCH.RESERVED_AT,
            foundation,
            ledger,
            materializer,
            backtest,
        )
    monkeypatch.setattr(target_runtime, "_publish", publish)
    before = (materializer.calls, backtest.store_calls, backtest.preparation_calls)

    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert result == NoReport(result.validation_plan_ref, "TARGET_STORE_INVALID")
    assert (materializer.calls, backtest.store_calls, backtest.preparation_calls) == before


@pytest.mark.parametrize(
    "mutation",
    (
        "case_ref",
        "evidence_ref",
        "empty_evidence",
        "wrong_evidence",
        "malformed_threshold",
        "outcome",
        "reason_codes",
        "limitation",
    ),
)
def test_committed_oos_case_mutation_rejects_without_rerunning_economics(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    publish = target_runtime._publish

    def forge_case(*args, **kwargs):
        if (
            args[1] == "validation_case_result"
            and args[3]["validation_target_materialization_evidence_ref"] is not None
        ):
            payload = deepcopy(args[3])
            if mutation == "case_ref":
                payload["case_ref"] = _RESEARCH._artifact_ref(
                    "validation_case", "f"
                )
            elif mutation == "evidence_ref":
                payload["validation_target_materialization_evidence_ref"] = (
                    _RESEARCH._artifact_ref(
                        "validation_target_materialization_evidence", "f"
                    )
                )
            elif mutation == "empty_evidence":
                payload["evidence"] = {}
            elif mutation == "wrong_evidence":
                payload["evidence"] = {"code": "BACKTEST_OPERATION_FAILED"}
            elif mutation == "malformed_threshold":
                payload["threshold_evaluation"] = {"metric_key": "forged"}
            elif mutation == "outcome":
                payload["outcome"] = "FAIL"
            elif mutation == "reason_codes":
                payload["reason_codes"] = ["OOS_THRESHOLD_NOT_MET"]
            else:
                payload["limitations"] = ["forged"]
            publish(args[0], args[1], args[2], payload)
            raise KeyboardInterrupt("after forged case result commit")
        return publish(*args, **kwargs)

    monkeypatch.setattr(target_runtime, "_publish", forge_case)
    with pytest.raises(KeyboardInterrupt):
        validate_target_candidate(
            candidate.strategy_candidate_ref,
            _policy(),
            _RESEARCH.RESERVED_AT,
            foundation,
            ledger,
            materializer,
            backtest,
        )
    monkeypatch.setattr(target_runtime, "_publish", publish)
    before = (
        materializer.calls,
        backtest.store_calls,
        backtest.preparation_calls,
        backtest.run_calls,
        backtest.economic_run_calls,
    )

    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert result == NoReport(result.validation_plan_ref, "CASE_COVER_INVALID")
    assert (
        materializer.calls,
        backtest.store_calls,
        backtest.preparation_calls,
        backtest.run_calls,
        backtest.economic_run_calls,
    ) == before


def test_committed_terminal_case_evidence_mutation_rejects_without_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    backtest.delegate.terminal_status = "BLOCKED"
    publish = target_runtime._publish

    def forge_terminal(*args, **kwargs):
        if (
            args[1] == "validation_case_result"
            and args[3]["validation_target_materialization_evidence_ref"] is not None
        ):
            payload = deepcopy(args[3])
            payload["evidence"]["status"] = "CANCELLED"
            publish(args[0], args[1], args[2], payload)
            raise KeyboardInterrupt("after forged terminal case result commit")
        return publish(*args, **kwargs)

    monkeypatch.setattr(target_runtime, "_publish", forge_terminal)
    with pytest.raises(KeyboardInterrupt):
        validate_target_candidate(
            candidate.strategy_candidate_ref,
            _policy(),
            _RESEARCH.RESERVED_AT,
            foundation,
            ledger,
            materializer,
            backtest,
        )
    monkeypatch.setattr(target_runtime, "_publish", publish)
    before = (
        materializer.calls,
        backtest.store_calls,
        backtest.preparation_calls,
        backtest.run_calls,
        backtest.economic_run_calls,
    )

    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert result == NoReport(result.validation_plan_ref, "CASE_COVER_INVALID")
    assert (
        materializer.calls,
        backtest.store_calls,
        backtest.preparation_calls,
        backtest.run_calls,
        backtest.economic_run_calls,
    ) == before


@pytest.mark.parametrize(
    "mutation",
    ("plan", "link", "target_evidence", "threshold", "limitation", "assessment"),
)
def test_report_replay_rejects_forged_links_and_derived_fields(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    publish = target_runtime._publish
    assess_holdout = ledger.assess_holdout
    assessment_calls = 0

    def counted_assess(*args, **kwargs):
        nonlocal assessment_calls
        assessment_calls += 1
        return assess_holdout(*args, **kwargs)

    monkeypatch.setattr(ledger, "assess_holdout", counted_assess)

    def forge_report(*args, **kwargs):
        if args[1] == "validation_report":
            payload = deepcopy(args[3])
            if mutation == "plan":
                payload["validation_plan_ref"] = _RESEARCH._artifact_ref(
                    "validation_plan", "f"
                )
            elif mutation == "link":
                payload["case_result_refs"] = list(reversed(payload["case_result_refs"]))
            elif mutation == "target_evidence":
                payload["validation_target_materialization_evidence_ref"] = (
                    _RESEARCH._artifact_ref(
                        "validation_target_materialization_evidence", "f"
                    )
                )
            elif mutation == "threshold":
                payload["threshold_evaluations"] = []
            elif mutation == "limitation":
                payload["limitations"] = ["forged"]
            else:
                payload["sample_integrity_ref"] = _RESEARCH._artifact_ref(
                    "sample_integrity_assessment", "f"
                )
            publish(args[0], args[1], args[2], payload)
            raise KeyboardInterrupt("after forged report commit")
        return publish(*args, **kwargs)

    monkeypatch.setattr(target_runtime, "_publish", forge_report)
    with pytest.raises(KeyboardInterrupt):
        validate_target_candidate(
            candidate.strategy_candidate_ref,
            _policy(),
            _RESEARCH.RESERVED_AT,
            foundation,
            ledger,
            materializer,
            backtest,
        )
    monkeypatch.setattr(target_runtime, "_publish", publish)
    assert assessment_calls == 1
    before = (
        materializer.calls,
        backtest.store_calls,
        backtest.preparation_calls,
        backtest.run_calls,
        backtest.economic_run_calls,
    )

    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert type(result) is NoReport
    assert assessment_calls == 1
    assert (
        materializer.calls,
        backtest.store_calls,
        backtest.preparation_calls,
        backtest.run_calls,
        backtest.economic_run_calls,
    ) == before


def test_run_return_case_result_and_report_interruptions_use_cache_then_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    run = backtest.delegate.run
    interrupted = False

    def after_run(request: dict):
        nonlocal interrupted
        result = run(request)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("after run return")
        return result

    monkeypatch.setattr(backtest.delegate, "run", after_run)
    with pytest.raises(KeyboardInterrupt):
        validate_target_candidate(
            candidate.strategy_candidate_ref,
            _policy(),
            _RESEARCH.RESERVED_AT,
            foundation,
            ledger,
            materializer,
            backtest,
        )
    monkeypatch.setattr(backtest.delegate, "run", run)
    publish = target_runtime._publish
    stages = iter(("validation_case_result", "validation_report"))
    current = next(stages)

    def after_publication(*args, **kwargs):
        nonlocal current
        ref = publish(*args, **kwargs)
        if args[1] == current and (
            current == "validation_report"
            or args[3].get("validation_target_materialization_evidence_ref") is not None
        ):
            try:
                current = next(stages)
            except StopIteration:
                current = "done"
            raise KeyboardInterrupt("after Validation publication")
        return ref

    monkeypatch.setattr(target_runtime, "_publish", after_publication)
    with pytest.raises(KeyboardInterrupt):
        validate_target_candidate(
            candidate.strategy_candidate_ref,
            _policy(),
            _RESEARCH.RESERVED_AT,
            foundation,
            ledger,
            materializer,
            backtest,
        )
    with pytest.raises(KeyboardInterrupt):
        validate_target_candidate(
            candidate.strategy_candidate_ref,
            _policy(),
            _RESEARCH.RESERVED_AT,
            foundation,
            ledger,
            materializer,
            backtest,
        )
    monkeypatch.setattr(target_runtime, "_publish", publish)
    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert type(result) is PublishedValidationReport
    assert materializer.calls == backtest.store_calls == 2
    assert backtest.run_calls == 3
    assert backtest.economic_run_calls == 2
    assert backtest.cache_calls == 1


@pytest.mark.parametrize(
    ("boundary", "reason"),
    (
        ("materializer", "TARGET_MATERIALIZATION_INVALID"),
        ("store", "TARGET_STORE_INVALID"),
        ("prepare", "TARGET_PREPARATION_FAILED"),
    ),
)
def test_target_boundary_failures_are_validation_local(
    tmp_path: Path, boundary: str, reason: str
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    if boundary == "materializer":
        materializer.mutate = "failure"
    elif boundary == "store":
        backtest.delegate.store_mutation = "failure"
    else:
        backtest.fail_prepare = True

    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert result == NoReport(result.validation_plan_ref, reason)


@pytest.mark.parametrize("mutation", ("malformed", "mismatch"))
def test_materializer_strategy_identity_fails_before_materialization_read(
    tmp_path: Path, mutation: str
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    materializer.strategy_artifact = (
        {"type": "not-a-build-artifact"}
        if mutation == "malformed"
        else _RESEARCH._strategy("f")
    )
    before = (materializer.calls, backtest.store_calls)

    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert result == NoReport(result.validation_plan_ref, "TARGET_MATERIALIZATION_INVALID")
    assert (materializer.calls, backtest.store_calls) == before
    assert backtest.preparation_calls == 0


@pytest.mark.parametrize("mutation", ("extra", "artifact", "request"))
def test_materializer_artifact_request_and_result_shape_fail_before_target_store(
    tmp_path: Path, mutation: str
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    materializer.mutate = mutation

    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert type(result) is NoReport
    assert result.reason_code == "TARGET_MATERIALIZATION_INVALID"
    assert backtest.store_calls == 1
    assert backtest.preparation_calls == 0


@pytest.mark.parametrize(
    "mutation", ("ref", "producer", "stream", "digest", "failure")
)
def test_target_ref_context_digest_count_load_and_retention_fail_before_evidence(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    if mutation == "ref":
        backtest.delegate.store_mutation = "ref"
    else:
        load = backtest.load_target

        def mutate(ref: dict[str, object]):
            loaded = load(ref)
            producer = loaded["producer_context_ref"]
            if producer["artifact_type"] != "validation_case":
                return loaded
            if mutation == "failure":
                raise RuntimeError("retention unavailable")
            if mutation == "producer":
                loaded["producer_context_ref"] = _RESEARCH._artifact_ref(
                    "validation_case", "f"
                )
            elif mutation == "stream":
                loaded["target_stream"] = _RESEARCH._target_stream("e")
            elif mutation == "digest":
                loaded["digest"] = _RESEARCH._hash("f")
            return loaded

        monkeypatch.setattr(backtest, "load_target", mutate, raising=False)

    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert type(result) is NoReport
    assert result.reason_code == "TARGET_STORE_INVALID"
    assert not _artifact_payloads(
        foundation, "validation_target_materialization_evidence"
    )
    assert backtest.preparation_calls == 0


def test_validation_evidence_publication_failure_is_the_commit_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    publish = target_runtime._publish

    def fail(*args, **kwargs):
        if args[1] == "validation_target_materialization_evidence":
            raise RuntimeError("Validation evidence unavailable")
        return publish(*args, **kwargs)

    monkeypatch.setattr(target_runtime, "_publish", fail)
    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert type(result) is NoReport
    assert result.reason_code == "TARGET_EVIDENCE_PUBLICATION_FAILED"
    assert backtest.preparation_calls == 0


@pytest.mark.parametrize(
    ("status", "result_type", "report_result", "reason"),
    (
        ("BLOCKED", PublishedValidationReport, "inconclusive", None),
        ("CANCELLED", PublishedValidationReport, "inconclusive", None),
        ("FAILED", NoReport, None, "BACKTEST_TERMINAL_FAILED"),
    ),
)
def test_backtest_terminal_outcomes_are_normalized_without_analysis(
    tmp_path: Path,
    status: str,
    result_type: type,
    report_result: str | None,
    reason: str | None,
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    backtest.delegate.terminal_status = status
    derives = backtest.derive_calls

    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert type(result) is result_type
    assert backtest.derive_calls == derives
    if type(result) is PublishedValidationReport:
        assert (
            _payload(foundation, result.validation_report_ref)["result"]
            == report_result
        )
    else:
        assert result.reason_code == reason


def test_holdout_reservation_failure_precedes_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    reserve = ledger.reserve

    def fail(record, producer_ref):
        if producer_ref.artifact_type == "validation_case":
            raise RuntimeError("reservation unavailable")
        return reserve(record, producer_ref)

    monkeypatch.setattr(ledger, "reserve", fail)
    calls = materializer.calls
    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert result.reason_code == "SAMPLE_LEDGER_CONFLICT"
    assert materializer.calls == calls
    assert backtest.preparation_calls == 0


@pytest.mark.parametrize(
    "mutation", ("wrong_event_id", "early_acceptance", "duplicate", "after_evidence")
)
def test_oos_reservation_recovery_requires_one_exact_preceding_append(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    first = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )
    assert type(first) is PublishedValidationReport
    original_entries = foundation.entries
    sample_entries = list(original_entries(SAMPLE_LOG))
    reservation_index = next(
        index
        for index, entry in enumerate(sample_entries)
        if json.loads(entry.payload)["payload"]["producer_ref"]["artifact_type"]
        == "validation_case"
    )
    reservation = sample_entries[reservation_index]
    evidence_sequence = next(
        entry.ledger_sequence
        for entry in original_entries(ARTIFACT_LOG)
        if json.loads(entry.payload)["artifact_type"]
        == "validation_target_materialization_evidence"
    )
    if mutation == "wrong_event_id":
        sample_entries[reservation_index] = replace(
            reservation, event_id=_RESEARCH._hash("f")
        )
    elif mutation == "early_acceptance":
        sample_entries[reservation_index] = replace(
            reservation, accepted_at="2026-08-25T23:59:59.000000Z"
        )
    elif mutation == "duplicate":
        sample_entries.append(
            replace(
                reservation,
                event_id=_RESEARCH._hash("e"),
                ledger_sequence=reservation.ledger_sequence + 100,
                log_sequence=reservation.log_sequence + 100,
            )
        )
    else:
        sample_entries[reservation_index] = replace(
            reservation, ledger_sequence=evidence_sequence
        )

    def entries(log_name, checkpoint=None):
        if log_name == SAMPLE_LOG and checkpoint is None:
            return tuple(sample_entries)
        return original_entries(log_name, checkpoint)

    monkeypatch.setattr(foundation, "entries", entries)
    monkeypatch.setattr(ledger, "reserve", lambda record, producer_ref: None)
    before = (
        materializer.calls,
        backtest.store_calls,
        backtest.preparation_calls,
        backtest.run_calls,
    )

    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert result == NoReport(result.validation_plan_ref, "SAMPLE_LEDGER_CONFLICT")
    assert (
        materializer.calls,
        backtest.store_calls,
        backtest.preparation_calls,
        backtest.run_calls,
    ) == before


def test_discovery_target_ref_substitution_is_rejected_even_for_equal_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    selected = _payload(foundation, candidate.strategy_candidate_ref)
    discovery = _payload(
        foundation, _ref(selected["selected_target_materialization_evidence_ref"])
    )["target_stream_ref"]
    publish = backtest.delegate.publish_target

    def substitute(context: dict, stream: dict):
        if context["artifact_type"] == "validation_case":
            return deepcopy(discovery)
        return publish(context, stream)

    monkeypatch.setattr(backtest.delegate, "publish_target", substitute)
    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert type(result) is NoReport
    assert result.reason_code == "TARGET_STORE_INVALID"
    assert backtest.preparation_calls == 0


def test_candidate_v3_substituted_target_evidence_fails_before_validation_publication(
    tmp_path: Path,
) -> None:
    foundation, ledger, materializer, backtest, candidate = _runtime(tmp_path)
    forged = _payload(foundation, candidate.strategy_candidate_ref)
    forged["selected_target_materialization_evidence_ref"] = _RESEARCH._artifact_ref(
        "target_materialization_evidence", "f"
    )
    envelope = ArtifactEnvelope.create("strategy_candidate", 3, forged)
    candidate_ref = foundation.put(envelope=envelope)
    foundation.append(
        "research.artifacts.v1",
        canonical_sha256(
            ("artifact-publication-v1", "research.artifacts.v1", candidate_ref)
        ),
        canonical_bytes(envelope),
    )
    calls = materializer.calls

    result = validate_target_candidate(
        candidate_ref,
        _policy(),
        _RESEARCH.RESERVED_AT,
        foundation,
        ledger,
        materializer,
        backtest,
    )

    assert result == NoReport(None, "CANDIDATE_PROVENANCE_INVALID")
    assert foundation.entries(ARTIFACT_LOG) == ()
    assert materializer.calls == calls


@pytest.mark.parametrize(
    ("field", "artifact_type", "schema_version"),
    (
        ("validation_case_ref", "wrong_case", 1),
        ("validation_case_ref", "validation_case", 2),
        ("candidate_ref", "wrong_candidate", 3),
        ("candidate_ref", "strategy_candidate", 2),
        ("target_recipe_ref", "wrong_recipe", 1),
        ("target_recipe_ref", "target_recipe", 2),
    ),
)
def test_validation_target_evidence_refs_require_exact_nominal_identity(
    field: str, artifact_type: str, schema_version: int
) -> None:
    values = {
        "validation_case_ref": _RESEARCH._artifact_ref("validation_case", "1"),
        "candidate_ref": {
            **_RESEARCH._artifact_ref("strategy_candidate", "2"),
            "schema_version": 3,
        },
        "target_recipe_ref": _RESEARCH._artifact_ref("target_recipe", "3"),
        "materialization_request_hash": _RESEARCH._hash("4"),
        "input_data_hash": _RESEARCH._hash("5"),
        "target_stream_ref": _RESEARCH._tagged_ref(
            "backtest_target_stream_ref", "backtest_target_stream", "6"
        ),
        "target_stream_digest": _RESEARCH._hash("7"),
        "event_count": 1,
    }
    values[field] = {
        **values[field],
        "artifact_type": artifact_type,
        "schema_version": schema_version,
    }

    with pytest.raises(ValueError, match="must reference"):
        ValidationTargetMaterializationEvidence(**values)


def test_target_runtime_has_no_research_or_backtest_production_import() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/crypto_quant_validation/target_runtime.py"
    ).read_text(encoding="utf-8")
    assert "crypto_quant_research" not in source
    assert "crypto_quant_backtest" not in source
    assert "Protocol" not in source
    assert "generic loader" not in source.lower()
