# Strategy Validation 模块设计

- **实现状态：** 以 [Roadmap status registry](../implementation/roadmap.md#2-status-registry) 为唯一权威；本文不维护节点状态
- **版本：** 0.3
- **集成权威：** [Integration v1 §5、§7、§9](../overall/integration-v1.md#5-validation-integration-sv-thin-01)
- **实现计划：** [Strategy Validation](../implementation/plans/validation.md)
- **上游：** StrategyCandidate、CandidateFamily、Backtest public evidence
- **下游：** Promotion Gate

Validation owns sample-consumption semantics, candidate admission, the v1 evidence-integrity/OOS cases, and the immutable report. It does not select candidates, mutate Backtest evidence, or authorize Shadow/Live operation.

## 1. Boundary

The integrated thin path freezes a checkpoint-backed `SampleConsumptionLedgerSnapshot` and candidate-specific ValidationPlan before OOS evidence is read. It publishes evidence-integrity and out-of-sample cases, runs the OOS request through the public Backtest seam, and emits `supported | rejected | inconclusive` only when required execution completes.

The accepted design is not an integrated runtime claim. Pure admission/OOS/report behavior is tested against frozen Research and Backtest consumer fixtures. Production-shell prerequisites and current node state are maintained only in the roadmap registry.

## 2. Frozen `SV-00A-core`

Validation is the sole semantic owner of `SampleConsumptionRecord`, `SampleConsumptionSnapshot`, and supplied-snapshot projection semantics. The current pure interface is:

```python
def build_snapshot(
    records: tuple[SampleConsumptionRecord, ...], *, as_of: UtcInstant
) -> SampleConsumptionSnapshot: ...


def assess_untouched_holdout(
    snapshot: SampleConsumptionSnapshot,
    *,
    dataset_revision: str,
    interval_start: UtcInstant,
    interval_end: UtcInstant,
) -> SampleIntegrityResult: ...
```

The Frozen record has the six documented fields, canonical UTC microsecond instants, half-open intervals, and the closed purpose vocabulary. `build_snapshot()` filters supplied records by `consumed_at`, canonical-sorts them, and `assess_untouched_holdout()` fails closed on an exact-revision overlap. `untouched=True` says only that the supplied pure snapshot contains no conflicting record; it does not prove ledger completeness, authority, selection lineage, or absence of an uninstrumented read.

The Frozen in-memory `SampleConsumptionSnapshot` and `SampleIntegrityResult` are pure values. They are not the later checkpoint-backed `SampleConsumptionLedgerSnapshot@1` and `SampleIntegrityAssessment@1` artifacts. ArtifactRef, append idempotency, authoritative checkpoint reconstruction, and cross-module producer publication are outside this Frozen slice.

## 3. `SV-THIN-01` accepted contract

The normative wrapper schemas, reservation time semantics, canonical append-entry refs, Plan/Case/Report links, flow, and outcome mappings are in [Integration v1 §5](../overall/integration-v1.md#5-validation-integration-sv-thin-01).

`consumed_at` is the logical pre-read reservation instant for the integration protocol, not physical I/O completion. `SampleConsumptionAppend@1` embeds the canonical six-field record together with `producer_ref`; the record is not a separately published/ref-addressed artifact, so one owner-log append atomically publishes the reservation and producer. TrialDeclaration and SelectionDeclaration are the Research producers; CandidateFamily and StrategyCandidate are provenance only. Only the `out_of_sample` ValidationCase emits the holdout `validation` reservation. An append conflict or failure blocks the corresponding read.

The Plan carries the candidate once. Cases and Reports resolve candidate/family provenance through the Plan rather than duplicating those refs. Forged or cross-plan links fail closed.

Validation receives `BacktestFacade` and `CanonicalEvidenceRepository` directly from the Backtest public root. Backtest alone verifies canonical bytes, manifests, retention, and hash chains; Validation only checks its own semantic/link eligibility. Consumers branch before analysis; only `CompletedPublication.publication_ref` reaches `derive()`.

## 4. Acceptance and deferral

`SV-THIN-01` acceptance requires a precommitted authoritative checkpoint and Plan, required reservation coverage, exact terminal/contamination mappings, an adverse OOS fixture with `simple_period_return = -0.1` and `trade_count = 1` yielding `ValidationReport(result="rejected")`, and forged/cross-case cache-link rejection.

Additional validation methods, Feature/model/trainer ABI, ModelBuild, and non-null model plans are outside v1.
