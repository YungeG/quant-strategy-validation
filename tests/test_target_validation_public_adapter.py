from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import crypto_quant_backtest as backtest
from crypto_quant_domain import (
    ArtifactRef,
    UtcInstant,
    canonical_bytes,
    canonical_sha256,
)
from crypto_quant_foundation import LocalFoundation
from crypto_quant_research import (
    DataSlice,
    ExperimentParameterCombination,
    ExperimentSelectionPolicy,
    ExperimentSpec,
    FrozenTargetExperimentInputs,
    HardFilter,
    OrderingCriterion,
    PublishedStrategyCandidate,
    TargetRecipe,
    TrialExecution,
    execute_target_experiment,
)

from crypto_quant_validation import (
    Holdout,
    OosRule,
    PublishedValidationReport,
    SampleConsumptionLedger,
    ValidationPolicy,
    validate_target_candidate,
)

_ROOT = Path(__file__).resolve().parents[2]
_RESEARCH_ROOT = next(
    path
    for path in (_ROOT / "research-platform", _ROOT / "quant-research-tsr-rp")
    if path.is_dir()
)
_PUBLIC_ADAPTER_PATH = (
    _RESEARCH_ROOT / "tests/test_target_stream_backtest_public_adapter.py"
)
_RESERVED_AT = "2026-08-26T00:00:00.000000Z"
_RECEIVED_AT = "2026-08-26T00:00:01.000000Z"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PUBLIC = _load("accepted_research_backtest_public_adapter", _PUBLIC_ADAPTER_PATH)


def _plain(value: object):
    return json.loads(canonical_bytes(value))


def _artifact(value: dict[str, object]) -> ArtifactRef:
    return ArtifactRef(
        value["artifact_type"], value["schema_version"], value["content_hash"]
    )


def _hash(marker: str) -> str:
    return "sha256:" + marker * 64


def _artifact_wire(artifact_type: str, marker: str) -> dict[str, object]:
    return ArtifactRef(artifact_type, 1, _hash(marker)).to_canonical_dict()


def _tagged_ref(tag: str, artifact_type: str, marker: str) -> dict[str, object]:
    return {"type": tag, "artifact_ref": _artifact_wire(artifact_type, marker)}


class _Materializer:
    def __init__(self, strategy_artifact: object, stream: object) -> None:
        self.strategy_artifact = strategy_artifact
        self.stream = stream
        self.calls = 0

    def materialize_target(self, request: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        return {
            "type": "target_materialization_result",
            "schema_version": 1,
            "request_hash": canonical_sha256(request),
            "strategy_artifact": self.strategy_artifact,
            "input_data_hash": _hash("d"),
            "target_stream": self.stream,
        }


class _DiscoveryBacktest:
    def __init__(self, public: object, metric_profile_ref: object) -> None:
        self.public = public
        self.metric_profile_ref = metric_profile_ref
        self.publication_ref = _tagged_ref(
            "backtest_canonical_publication_ref",
            "canonical_publication_manifest",
            "7",
        )
        self.analysis_ref = _tagged_ref(
            "analysis_artifact_ref", "backtest_analysis", "8"
        )

    def publish_target(self, context: dict[str, object], stream: dict[str, object]):
        return self.public.publish_target(context, stream)

    def load_target(self, ref: dict[str, object]) -> dict[str, object]:
        return self.public.load_target(ref)

    def prepare_trials(self, trials: tuple[object, ...], target_ref: dict[str, object]):
        prepared = self.public.prepare_trials(trials, target_ref)
        return (
            TrialExecution(
                prepared[0].trial_declaration_ref,
                prepared[0].request_spec,
                {"type": "backtest_request_ref", "id": "discovery"},
            ),
        )

    def run(self, request: dict[str, object]) -> dict[str, object]:
        return self.publication_ref

    def load_completed(self, ref: object) -> dict[str, object]:
        return {
            "publication_ref": self.publication_ref,
            "semantic_run_id": "public-discovery",
            "execution_result_hash": _hash("a"),
            "result_grade": "development",
        }

    def load_terminal(self, ref: object) -> dict[str, object]:
        raise AssertionError(f"unexpected terminal: {ref}")

    def derive(self, publication_ref: object, metric_profile_ref: object) -> object:
        return self.analysis_ref

    def load_analysis(self, ref: object) -> dict[str, object]:
        return {
            "analysis_ref": self.analysis_ref,
            "metric_profile_ref": self.metric_profile_ref,
            "source_publication_ref": self.publication_ref,
            "source_execution_result_hash": _hash("a"),
            "simple_period_return": "0.1",
            "trade_count": 1,
            "result_grade": "development",
        }


class _AcceptedPublicValidationAdapter:
    def __init__(
        self,
        public: object,
        discovery: _DiscoveryBacktest,
        metric_profile_ref: ArtifactRef,
    ) -> None:
        self.public = public
        self.discovery = discovery
        self.metric_profile_ref = metric_profile_ref
        self.prepared: dict[str, object] = {}
        self.run_calls = 0
        self.prepare_calls = 0
        self.repository = backtest.BacktestEvidenceRepository(self.public.cas)
        self.analysis_runtime = backtest.BacktestAnalysisRuntime(self.public.cas)

    def publish_target(self, context: dict[str, object], stream: dict[str, object]):
        return self.public.publish_target(context, stream)

    def load_target(self, ref: dict[str, object]) -> dict[str, object]:
        return self.public.load_target(ref)

    def prepare_target(
        self, validation_case_ref: dict[str, object], target_ref: dict[str, object]
    ):
        self.prepare_calls += 1
        artifact = target_ref["artifact_ref"]
        prepared = backtest.prepare_cash_target_stream_backtest(
            request_intent=backtest.CashDevelopmentRequestIntent(
                1,
                canonical_bytes(validation_case_ref).decode(),
                backtest.TimelineWindow(UtcInstant(0), UtcInstant(90), UtcInstant(300)),
                "account:primary",
                _PUBLIC._USD,
                7,
            ),
            provider_inputs=backtest.CashDevelopmentProviderInputs(
                1,
                _PUBLIC._manifest(),
                _PUBLIC._catalog(),
                "trend-v1",
                _PUBLIC.StrategySleeveId("trend.primary"),
                _PUBLIC.Money(100_000, _PUBLIC.Scale(2), "USD"),
                _PUBLIC._lattice(),
                _PUBLIC._mark(10_000, 100, "decision"),
                _PUBLIC._mark(8_000, 299, "final"),
                _PUBLIC._capabilities(),
            ),
            target_stream_ref=backtest.BacktestTargetStreamRef(
                ArtifactRef(
                    artifact["artifact_type"],
                    artifact["schema_version"],
                    artifact["content_hash"],
                )
            ),
            artifact_reader=self.public.cas,
            artifact_publisher=self.public.cas,
            market_reader=self.public.market_reader,
            publication_root=self.public.root,
        )
        key = canonical_bytes(validation_case_ref).decode()
        self.prepared[key] = prepared
        return key

    def run(self, request: str):
        self.run_calls += 1
        prepared = self.prepared[request]
        return _plain(prepared.runtime.run(prepared.execution_request))

    def load_completed(self, ref: dict[str, object]) -> dict[str, object]:
        if canonical_bytes(ref) == canonical_bytes(self.discovery.publication_ref):
            return self.discovery.load_completed(ref)
        nominal = backtest.BacktestCanonicalPublicationRef(
            _artifact(ref["artifact_ref"])
        )
        completed = self.repository.load_completed(nominal)
        return {
            "publication_ref": _plain(completed.source_publication_ref),
            "semantic_run_id": completed.semantic_run_id,
            "execution_result_hash": completed.source_execution_result_hash,
            "result_grade": completed.result_grade.value,
        }

    def load_terminal(self, ref: object) -> dict[str, object]:
        raise AssertionError(f"unexpected terminal: {ref}")

    def derive(self, ref: dict[str, object], metric_profile_ref: dict[str, object]):
        assert _artifact(metric_profile_ref) == self.metric_profile_ref
        completed = self.repository.load_completed(
            backtest.BacktestCanonicalPublicationRef(_artifact(ref["artifact_ref"]))
        )
        return _plain(self.analysis_runtime.derive(completed, self.metric_profile_ref))

    def load_analysis(self, ref: dict[str, object]) -> dict[str, object]:
        if canonical_bytes(ref) == canonical_bytes(self.discovery.analysis_ref):
            return self.discovery.load_analysis(ref)
        loaded = self.repository.load_analysis(
            backtest.AnalysisArtifactRef(_artifact(ref["artifact_ref"]))
        )
        return _plain(loaded)


def _inputs(
    metric_profile_ref: ArtifactRef,
    strategy_artifact: object,
    market_bundle_ref: object,
):
    recipe = TargetRecipe(
        "fixed-targets",
        strategy_artifact,
        _hash("c"),
        ("bars.open",),
    )
    metric_ref = metric_profile_ref.to_canonical_dict()
    spec = ExperimentSpec(
        _artifact_wire("hypothesis", "1"),
        _artifact_wire("strategy_definition", "2"),
        (
            DataSlice(
                market_bundle_ref,
                "public-discovery-v1",
                "2026-01-01T00:00:00.000000Z",
                "2026-02-01T00:00:00.000000Z",
            ),
        ),
        (ExperimentParameterCombination((("lookback", "10"),)),),
        (1,),
        (_artifact_wire("scenario", "4"),),
        _artifact_wire("backtest_template", "5"),
        None,
        (metric_ref,),
        {"max_trials": 1},
        recipe.ref,
    )
    policy = ExperimentSelectionPolicy(
        metric_ref,
        ("COMPLETED",),
        ("development",),
        (HardFilter("trade_count", "gte", 1),),
        (OrderingCriterion("simple_period_return", "descending"),),
        1,
        "trial_declaration_ref_ascending",
    )
    return FrozenTargetExperimentInputs(
        spec,
        recipe,
        policy,
        {"type": "actor_ref", "actor_id": "research"},
        _RESERVED_AT,
    )


def test_validation_adapter_has_no_research_internal_imports() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    imports = ast.walk(ast.parse(source))
    assert all(
        node.module == "crypto_quant_research"
        for node in imports
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("crypto_quant_research")
    )
    assert all(
        alias.name == "crypto_quant_research"
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith("crypto_quant_research")
    )


def test_validation_adapter_composes_only_accepted_research_and_backtest_public_roots(
    tmp_path: Path,
) -> None:
    foundation = LocalFoundation(
        tmp_path / "foundation", clock=lambda: _RECEIVED_AT
    )
    ledger = SampleConsumptionLedger(foundation)
    cas = _PUBLIC._Cas()
    metric_profile_ref = backtest.BacktestAnalysisRuntime(cas).publish_metric_profile()
    strategy_artifact = _plain(
        next(
            item
            for item in _PUBLIC._manifest().artifacts
            if item.role is backtest.BuildArtifactRole.DECISION_SOURCE
        )
    )
    public_targets = _PUBLIC._PublicTargetAdapter(tmp_path / "public-backtest")
    discovery_materializer = _Materializer(
        strategy_artifact, _plain(public_targets.stream)
    )
    discovery_backtest = _DiscoveryBacktest(
        public_targets, metric_profile_ref.to_canonical_dict()
    )
    candidate = execute_target_experiment(
        _inputs(
            metric_profile_ref,
            strategy_artifact,
            public_targets.market_reader.bundle_ref.to_canonical_dict(),
        ),
        foundation,
        ledger,
        discovery_materializer,
        discovery_backtest,
    )
    assert type(candidate) is PublishedStrategyCandidate
    candidate_payload = _plain(
        foundation.read(ref=candidate.strategy_candidate_ref).envelope.payload
    )
    discovery_evidence_ref = _artifact(
        candidate_payload["selected_target_materialization_evidence_ref"]
    )
    discovery_evidence = _plain(
        foundation.read(ref=discovery_evidence_ref).envelope.payload
    )
    discovery_target = public_targets.load_target(
        discovery_evidence["target_stream_ref"]
    )
    assert type(public_targets.repository) is backtest.BacktestTargetStreamRepository
    assert discovery_target["producer_context_ref"]["artifact_type"] == (
        "trial_declaration"
    )

    public_adapter = _AcceptedPublicValidationAdapter(
        public_targets, discovery_backtest, metric_profile_ref
    )
    assert (
        public_adapter.analysis_runtime.publish_metric_profile() == metric_profile_ref
    )
    validation_materializer = _Materializer(
        strategy_artifact, _plain(public_adapter.public.stream)
    )
    policy = ValidationPolicy(
        ("development",),
        (metric_profile_ref.to_canonical_dict(),),
        Holdout(
            public_adapter.public.market_reader.bundle_ref.to_canonical_dict(),
            "public-oos-v1",
            "2026-03-01T00:00:00.000000Z",
            "2026-04-01T00:00:00.000000Z",
            "HOLDOUT",
            False,
        ),
        OosRule(
            metric_profile_ref.to_canonical_dict(),
            "simple_period_return",
            "fraction",
            "gte",
            "-1",
            1,
        ),
    )

    result = validate_target_candidate(
        candidate.strategy_candidate_ref,
        policy,
        _RESERVED_AT,
        foundation,
        ledger,
        validation_materializer,
        public_adapter,
    )

    assert type(result) is PublishedValidationReport
    report = _plain(
        foundation.read(ref=result.validation_report_ref).envelope.payload
    )
    assert report["result"] == "supported"
    oos_evidence_ref = _artifact(
        report["validation_target_materialization_evidence_ref"]
    )
    oos_evidence = _plain(foundation.read(ref=oos_evidence_ref).envelope.payload)
    oos_target = public_targets.load_target(oos_evidence["target_stream_ref"])
    assert oos_target["producer_context_ref"]["artifact_type"] == "validation_case"
    assert oos_evidence["target_stream_ref"] != discovery_evidence["target_stream_ref"]
    assert (
        validation_materializer.calls
        == public_adapter.prepare_calls
        == public_adapter.run_calls
        == 1
    )
    assert "prepare_cash_target_stream_backtest" in backtest.__all__
    assert "execute_target_experiment" in __import__("crypto_quant_research").__all__
