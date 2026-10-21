"""Synthetic owner-port contract tests. No samples, engine, ledger, or report."""
from __future__ import annotations

from copy import deepcopy
from decimal import localcontext

import pytest

from crypto_quant_domain import canonical_bytes
from crypto_quant_validation import (
    PAIRED_PORTFOLIO_COMPARISON_FAILURE_PRECEDENCE,
    PairedPortfolioComparisonRuleV1,
    ValidationReport,
    compare_paired_portfolio_development_v1,
)


def _ref(name, char):
    return {"type": "artifact_ref", "artifact_type": name, "schema_version": 1,
            "content_hash": "sha256:" + char * 64}


def _nav_ref(char):
    return {"type": "cn_a_share_portfolio_daily_nav_analysis_ref", "schema_version": 1,
            "artifact_ref": _ref("cn_a_share_portfolio_daily_nav_analysis", char)}


_PROFILE = _ref("cn_a_share_portfolio_daily_nav_metric_profile", "1")
_RULE = PairedPortfolioComparisonRuleV1(_PROFILE, 4, 1)


def _context(char):
    return {"type": "cn_a_share_portfolio_daily_nav_comparison_context", "schema_version": 1,
        "analysis_ref": _nav_ref(char), "validation_eligible": False,
        "timeline_window": {"frozen_test_interval": "same"},
        "target_stream_ref": {"type": "backtest_target_stream_ref", "artifact_ref": _ref("test_target", char)},
        "build_artifact_manifest_hash": "sha256:" + "2" * 64,
        "analysis": {"type": "cn_a_share_portfolio_daily_nav_analysis", "schema_version": 1,
            "metric_profile_ref": _PROFILE, "result_grade": "development", "validation_eligible": False,
            "source_publication_ref": {"type": "backtest_canonical_publication_ref",
                "artifact_ref": _ref("canonical_publication_manifest", char)},
            "definition_ref": _ref("backtest_profile_portfolio_definition", "3"),
            "starting_snapshot_hash": "sha256:" + "4" * 64,
            "initial_equity": {"type": "money", "currency": "CNY", "scale": 2, "units": 10000000},
            "points": [{"trade_date": "2024-10-14", "occurred_at": {"test_full_clock": 1}}],
            "simple_period_return": "0.1", "maximum_drawdown": "0.2" if char == "a" else "0.19"}}


class _Port:
    def __init__(self):
        self.views = {"a": _context("a"), "b": _context("b")}
        self.reads = []

    def load_cn_a_share_portfolio_daily_nav_comparison(self, ref):
        char = ref["artifact_ref"]["content_hash"][-1]
        self.reads.append(char)
        return deepcopy(self.views[char])


def _compare(port=None, **changes):
    values = {"rule": _RULE, "a_analysis_ref": _nav_ref("a"), "b_analysis_ref": _nav_ref("b"),
              "backtest": port or _Port(), "evaluable_weeks": 4, "bull_weeks": 1}
    return compare_paired_portfolio_development_v1(**{**values, **changes})


def test_public_equal_return_and_strictly_lower_drawdown_is_diagnostic_only():
    port = _Port()
    result = _compare(port)
    assert port.reads == ["a", "b"]
    assert result.threshold_outcome == "would_pass" and result.net_return_difference == "0"
    assert result.formal_result == "inconclusive" and not result.validation_eligible
    assert not isinstance(result, ValidationReport)
    wire = result.to_canonical_dict()
    assert wire["validation_report"] is None and not wire["holdout_reserved"]
    assert not wire["sample_coverage_verified"] and not wire["deployment_authorized"]
    assert canonical_bytes(result) == canonical_bytes(_compare())
    assert _RULE.to_canonical_dict()["maximum_drawdown_operator"] == "B_lt_A"


@pytest.mark.parametrize("change,outcome", [
    ({"maximum_drawdown": "0.2"}, "would_fail"),
    ({"simple_period_return": "0.09"}, "would_fail"),
    ({"simple_period_return": None}, "missing_metrics"),
    ({"maximum_drawdown": None}, "missing_metrics"),
])
def test_metric_thresholds_never_zero_fill_or_accept_equal_drawdown(change, outcome):
    port = _Port()
    port.views["b"]["analysis"].update(change)
    result = _compare(port)
    assert result.threshold_outcome == outcome
    assert result.formal_result == "inconclusive"
    if outcome == "missing_metrics":
        assert result.net_return_difference is None
        assert result.reason_codes == ("METRIC_MISSING_OR_INVALID",)


@pytest.mark.parametrize("key,value", [
    ("definition_ref", _ref("backtest_profile_portfolio_definition", "9")),
    ("starting_snapshot_hash", "sha256:" + "9" * 64),
    ("initial_equity", {"type": "money", "currency": "CNY", "scale": 2, "units": 9999999}),
    ("timeline_window", {"frozen_test_interval": "other"}),
    ("build_artifact_manifest_hash", "sha256:" + "9" * 64),
    ("close_clock_sequence", [{"trade_date": "2024-10-15", "occurred_at": {"test_full_clock": 2}}]),
])
def test_same_source_context_is_checked_before_insufficient_sample_and_bad_metrics(key, value):
    port = _Port()
    target = port.views["b"] if key in ("timeline_window", "build_artifact_manifest_hash") else port.views["b"]["analysis"]
    target["points" if key == "close_clock_sequence" else key] = value
    port.views["b"]["analysis"]["simple_period_return"] = "nan"
    result = _compare(port, evaluable_weeks=0, bull_weeks=0)
    assert result.reason_codes == ("PAIR_SOURCE_CONTEXT_MISMATCH",)
    assert result.context_differences == (key,) and result.net_return_difference is None


@pytest.mark.parametrize("part", ["same_publication", "same_target", "wrong_ref", "wrong_profile",
    "wrong_grade", "true_eligible", "integer_eligible", "bool_version", "unknown_context_field"])
def test_link_and_distinct_arm_failures_do_not_evaluate_thresholds(part):
    port = _Port()
    a, b = port.views["a"], port.views["b"]
    if part == "same_publication":
        b["analysis"]["source_publication_ref"] = a["analysis"]["source_publication_ref"]
    elif part == "same_target":
        b["target_stream_ref"] = a["target_stream_ref"]
    elif part == "wrong_ref":
        b["analysis_ref"] = a["analysis_ref"]
    elif part == "wrong_profile":
        b["analysis"]["metric_profile_ref"] = _ref("cn_a_share_portfolio_daily_nav_metric_profile", "9")
    elif part == "wrong_grade":
        b["analysis"]["result_grade"] = "decision_grade"
    elif part == "true_eligible":
        b["validation_eligible"] = True
    elif part == "integer_eligible":
        b["analysis"]["validation_eligible"] = 0
    elif part == "bool_version":
        b["schema_version"] = True
    else:
        b["unknown"] = 1
    result = _compare(port)
    expected = "PAIR_ARMS_NOT_DISTINCT" if part in ("same_publication", "same_target") else "PAIR_ANALYSIS_LINK_INVALID"
    assert result.reason_codes == (expected,) and result.threshold_outcome == "not_evaluated"


@pytest.mark.parametrize("change", [
    {"rule": object()}, {"a_analysis_ref": _ref("backtest_analysis", "a")},
    {"evaluable_weeks": True}, {"evaluable_weeks": -1}, {"bull_weeks": 5},
])
def test_invalid_declarations_fail_before_any_owner_read(change):
    port = _Port()
    with pytest.raises(ValueError):
        _compare(port, **change)
    assert port.reads == []


def test_provider_error_retains_exact_code_without_terminal_or_metric_fabrication():
    class Failure(Exception):
        code = "PORT_MANIFEST_INVALID"
    class Port:
        def load_cn_a_share_portfolio_daily_nav_comparison(self, ref):
            raise Failure("retained source missing")
    result = _compare(Port())
    assert result.reason_codes == ("BACKTEST_EVIDENCE_ERROR",)
    assert result.provider_failure_code == "PORT_MANIFEST_INVALID"
    assert result.net_return_difference is None
    assert _compare(backtest=object()).provider_failure_code == "PORT_REF_TYPE_MISMATCH"


@pytest.mark.parametrize("counts", [(3, 1), (4, 0), (0, 0)])
def test_declared_sample_insufficiency_precedes_an_adverse_threshold(counts):
    port = _Port()
    port.views["b"]["analysis"]["simple_period_return"] = "-0.2"
    result = _compare(port, evaluable_weeks=counts[0], bull_weeks=counts[1])
    assert result.threshold_outcome == "insufficient_sample"
    assert result.reason_codes == ("INSUFFICIENT_DECLARED_WEEKS",)


@pytest.mark.parametrize("a,b,delta", [("0", "100", "100"),
    ("123456789012345678901234567890.1", "123456789012345678901234567891.1", "1")])
def test_delta_preserves_integer_zeroes_and_ignores_global_decimal_precision(a, b, delta):
    port = _Port()
    port.views["a"]["analysis"]["simple_period_return"] = a
    port.views["b"]["analysis"]["simple_period_return"] = b
    with localcontext() as context:
        context.prec = 3
        result = _compare(port)
    assert result.net_return_difference == delta and result.threshold_outcome == "would_pass"


def test_failure_precedence_is_frozen():
    codes = PAIRED_PORTFOLIO_COMPARISON_FAILURE_PRECEDENCE
    assert codes.index("BACKTEST_EVIDENCE_ERROR") < codes.index("PAIR_ANALYSIS_LINK_INVALID")
    assert codes.index("PAIR_SOURCE_CONTEXT_MISMATCH") < codes.index("METRIC_MISSING_OR_INVALID")
    assert codes.index("METRIC_MISSING_OR_INVALID") < codes.index("INSUFFICIENT_DECLARED_WEEKS")
