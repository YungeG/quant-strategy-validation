"""Public, versioned paired DEVELOPMENT comparison; never a ValidationReport.

Backtest's injected owner port alone verifies NAV and context. Validation checks
semantic pairing and the predeclared thresholds. Week counts are declarations,
not sample-ledger evidence. Formal OOS/sample/source/fee gates stay independent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from crypto_quant_domain import canonical_bytes

from .integration import _artifact_ref_matches
from .paired_portfolio_oos_draft_v0 import (
    DraftPairedPortfolioRuleV0,
    evaluate_draft_paired_portfolio_thresholds_v0,
)

PAIRED_PORTFOLIO_COMPARISON_FAILURE_PRECEDENCE = (
    "COMPARISON_RULE_INVALID", "ANALYSIS_REF_INVALID", "DECLARED_SAMPLE_COUNTS_INVALID",
    "BACKTEST_EVIDENCE_ERROR", "PAIR_ANALYSIS_LINK_INVALID", "PAIR_ARMS_NOT_DISTINCT",
    "PAIR_SOURCE_CONTEXT_MISMATCH", "METRIC_MISSING_OR_INVALID", "INSUFFICIENT_DECLARED_WEEKS",
)
_CONTEXT_FIELDS = {"type", "schema_version", "analysis_ref", "analysis", "timeline_window",
                   "target_stream_ref", "build_artifact_manifest_hash", "validation_eligible"}


def _wire(value: object) -> Any:
    try:
        return json.loads(canonical_bytes(value))
    except (TypeError, ValueError) as error:
        raise ValueError("comparison requires canonical wire values") from error


def _exact_ref(value: object, name: str, version: int) -> bool:
    return (type(value) is dict and type(value.get("schema_version")) is int
            and _artifact_ref_matches(value, name, version))


def _nav_ref(value: object) -> dict[str, Any]:
    raw = _wire(value)
    if (type(raw) is not dict or set(raw) != {"type", "schema_version", "artifact_ref"}
            or raw["type"] != "cn_a_share_portfolio_daily_nav_analysis_ref"
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1
            or not _exact_ref(raw["artifact_ref"], "cn_a_share_portfolio_daily_nav_analysis", 1)):
        raise ValueError("ANALYSIS_REF_INVALID: exact public NAV@1 ref required")
    return raw


@dataclass(frozen=True, slots=True)
class PairedPortfolioComparisonRuleV1:
    metric_profile_ref: object
    minimum_evaluable_weeks: int
    minimum_bull_weeks: int

    def __post_init__(self) -> None:
        if not _exact_ref(_wire(self.metric_profile_ref), "cn_a_share_portfolio_daily_nav_metric_profile", 1):
            raise ValueError("COMPARISON_RULE_INVALID: exact NAV metric-profile@1 ref required")
        DraftPairedPortfolioRuleV0(self.minimum_evaluable_weeks, self.minimum_bull_weeks)

    def to_canonical_dict(self) -> dict[str, object]:
        return {"type": "paired_portfolio_comparison_rule", "schema_version": 1,
                "metric_profile_ref": self.metric_profile_ref, "unit": "fraction",
                "net_return_difference_operator": "gte", "net_return_difference_threshold": "0",
                "maximum_drawdown_operator": "B_lt_A",
                "minimum_evaluable_weeks": self.minimum_evaluable_weeks,
                "minimum_bull_weeks": self.minimum_bull_weeks,
                "scope": "development_diagnostic_only", "validation_eligible": False}


@dataclass(frozen=True, slots=True)
class PairedPortfolioComparisonV1:
    rule: PairedPortfolioComparisonRuleV1
    a_analysis_ref: object
    b_analysis_ref: object
    threshold_outcome: str
    reason_codes: tuple[str, ...] = ()
    context_differences: tuple[str, ...] = ()
    net_return_difference: str | None = None
    a_maximum_drawdown: str | None = None
    b_maximum_drawdown: str | None = None
    provider_failure_code: str | None = None
    formal_result: str = field(default="inconclusive", init=False)
    validation_eligible: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if type(self.rule) is not PairedPortfolioComparisonRuleV1 or self.threshold_outcome not in {
                "not_evaluated", "missing_metrics", "insufficient_sample", "would_pass", "would_fail"}:
            raise ValueError("invalid development comparison result")
        if any(code not in PAIRED_PORTFOLIO_COMPARISON_FAILURE_PRECEDENCE for code in self.reason_codes):
            raise ValueError("unknown comparison reason code")

    def to_canonical_dict(self) -> dict[str, object]:
        return {"type": "paired_portfolio_comparison", "schema_version": 1,
                "rule": self.rule, "a_analysis_ref": self.a_analysis_ref, "b_analysis_ref": self.b_analysis_ref,
                "threshold_outcome": self.threshold_outcome, "reason_codes": self.reason_codes,
                "context_differences": self.context_differences,
                "net_return_difference": self.net_return_difference,
                "a_maximum_drawdown": self.a_maximum_drawdown, "b_maximum_drawdown": self.b_maximum_drawdown,
                "provider_failure_code": self.provider_failure_code, "formal_result": "inconclusive",
                "validation_eligible": False, "sample_coverage_verified": False,
                "holdout_reserved": False, "validation_report": None, "deployment_authorized": False}


def compare_paired_portfolio_development_v1(
    *, rule: PairedPortfolioComparisonRuleV1, a_analysis_ref: object, b_analysis_ref: object,
    backtest: object, evaluable_weeks: int, bull_weeks: int,
) -> PairedPortfolioComparisonV1:
    """Read two verified owner views, never caller-supplied return/DD metrics."""
    if type(rule) is not PairedPortfolioComparisonRuleV1:
        raise ValueError("COMPARISON_RULE_INVALID: exact paired comparison rule required")
    # Recheck declarations before the first I/O; do not trust a mutated value.
    PairedPortfolioComparisonRuleV1(rule.metric_profile_ref, rule.minimum_evaluable_weeks, rule.minimum_bull_weeks)
    refs = (_nav_ref(a_analysis_ref), _nav_ref(b_analysis_ref))
    if (type(evaluable_weeks) is not int or evaluable_weeks < 0
            or type(bull_weeks) is not int or not 0 <= bull_weeks <= evaluable_weeks):
        raise ValueError("DECLARED_SAMPLE_COUNTS_INVALID: bounded integer counts required")
    def fail(code: str, *, differences: tuple[str, ...] = (), provider: str | None = None):
        return PairedPortfolioComparisonV1(rule, a_analysis_ref, b_analysis_ref, "not_evaluated",
                                           (code,), differences, provider_failure_code=provider)
    loader = getattr(backtest, "load_cn_a_share_portfolio_daily_nav_comparison", None)
    if not callable(loader):
        return fail("BACKTEST_EVIDENCE_ERROR", provider="PORT_REF_TYPE_MISMATCH")
    try:
        views = tuple(_wire(loader(ref)) for ref in (a_analysis_ref, b_analysis_ref))
    except Exception as error:
        code = getattr(error, "code", None)
        code = code if type(code) is str else getattr(code, "value", "BACKTEST_OPERATION_FAILED")
        return fail("BACKTEST_EVIDENCE_ERROR", provider=str(code))
    for view, ref in zip(views, refs, strict=True):
        if (type(view) is not dict or set(view) != _CONTEXT_FIELDS
                or view["type"] != "cn_a_share_portfolio_daily_nav_comparison_context"
                or type(view["schema_version"]) is not int or view["schema_version"] != 1
                or view["analysis_ref"] != ref or type(view["validation_eligible"]) is not bool
                or view["validation_eligible"]
                or type(view["analysis"]) is not dict):
            return fail("PAIR_ANALYSIS_LINK_INVALID")
        analysis = view["analysis"]
        if (analysis.get("type") != "cn_a_share_portfolio_daily_nav_analysis"
                or type(analysis.get("schema_version")) is not int or analysis["schema_version"] != 1
                or analysis.get("metric_profile_ref") != _wire(rule.metric_profile_ref)
                or analysis.get("result_grade") != "development"
                or type(analysis.get("validation_eligible")) is not bool or analysis["validation_eligible"]):
            return fail("PAIR_ANALYSIS_LINK_INVALID")
    a, b = (v["analysis"] for v in views)
    if (refs[0] == refs[1] or a.get("source_publication_ref") == b.get("source_publication_ref")
            or views[0]["target_stream_ref"] == views[1]["target_stream_ref"]):
        return fail("PAIR_ARMS_NOT_DISTINCT")
    try:
        pairs = {key: (a[key], b[key]) for key in (
            "definition_ref", "starting_snapshot_hash", "initial_equity")}
        pairs.update({key: (views[0][key], views[1][key]) for key in (
            "timeline_window", "build_artifact_manifest_hash")})
        pairs["close_clock_sequence"] = (
            [(p["trade_date"], p["occurred_at"]) for p in a["points"]],
            [(p["trade_date"], p["occurred_at"]) for p in b["points"]],
        )
        differences = tuple(key for key, (left, right) in pairs.items() if left != right)
    except (KeyError, TypeError):
        return fail("PAIR_ANALYSIS_LINK_INVALID")
    if differences:
        return fail("PAIR_SOURCE_CONTEXT_MISMATCH", differences=differences)
    try:
        value = evaluate_draft_paired_portfolio_thresholds_v0(
            rule=DraftPairedPortfolioRuleV0(rule.minimum_evaluable_weeks, rule.minimum_bull_weeks),
            a_net_return=a.get("simple_period_return"), b_net_return=b.get("simple_period_return"),
            a_maximum_drawdown=a.get("maximum_drawdown"), b_maximum_drawdown=b.get("maximum_drawdown"),
            evaluable_weeks=evaluable_weeks, bull_weeks=bull_weeks)
    except (TypeError, ValueError):
        return fail("METRIC_MISSING_OR_INVALID")
    reason = (("METRIC_MISSING_OR_INVALID",) if value.threshold_outcome == "missing_metrics" else
              ("INSUFFICIENT_DECLARED_WEEKS",) if value.threshold_outcome == "insufficient_sample" else ())
    return PairedPortfolioComparisonV1(rule, a_analysis_ref, b_analysis_ref, value.threshold_outcome,
        reason, net_return_difference=value.net_return_difference,
        a_maximum_drawdown=value.a_maximum_drawdown, b_maximum_drawdown=value.b_maximum_drawdown)
