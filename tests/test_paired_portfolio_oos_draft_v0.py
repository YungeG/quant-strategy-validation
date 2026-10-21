"""Proposed paired threshold boundaries: no accepted ValidationReport or holdout."""
from __future__ import annotations

from importlib import import_module

import pytest

from crypto_quant_validation import OosRule, ValidationReport

_draft = import_module("crypto_quant_validation.paired_portfolio_oos_draft_v0")
DraftPairedPortfolioRuleV0 = _draft.DraftPairedPortfolioRuleV0
evaluate_draft_paired_portfolio_thresholds_v0 = _draft.evaluate_draft_paired_portfolio_thresholds_v0

_RULE = DraftPairedPortfolioRuleV0(4, 1)


def _compare(**changes):
    values = {
        "rule": _RULE, "a_net_return": "0.1", "b_net_return": "0.1",
        "a_maximum_drawdown": "0.2", "b_maximum_drawdown": "0.19",
        "evaluable_weeks": 4, "bull_weeks": 1,
    }
    return evaluate_draft_paired_portfolio_thresholds_v0(**{**values, **changes})


def test_equal_net_return_passes_only_draft_math_when_b_drawdown_is_strictly_lower():
    value = _compare()
    assert value.net_return_difference == "0"
    assert value.threshold_outcome == "would_pass"
    assert value.formal_result == "inconclusive" and not value.validation_eligible
    assert value.to_canonical_dict()["schema_version"] == 0
    assert value.rule.to_canonical_dict()["accepted_rule"] is False
    assert not isinstance(value, ValidationReport)


@pytest.mark.parametrize(("change", "outcome"), [
    ({"b_maximum_drawdown": "0.2"}, "would_fail"),
    ({"b_net_return": "0.09"}, "would_fail"),
    ({"evaluable_weeks": 3}, "insufficient_sample"),
    ({"bull_weeks": 0}, "insufficient_sample"),
    ({"a_net_return": None}, "missing_metrics"),
    ({"b_maximum_drawdown": None}, "missing_metrics"),
])
def test_strict_drawdown_return_and_missing_week_metric_precedence(change, outcome):
    value = _compare(**change)
    assert value.threshold_outcome == outcome
    assert value.formal_result == "inconclusive" and not value.validation_eligible
    if outcome == "missing_metrics":
        assert value.net_return_difference is None


@pytest.mark.parametrize("change", [
    {"a_net_return": "nan"}, {"a_net_return": "-1"},
    {"a_net_return": "-0"}, {"a_maximum_drawdown": "-0.01"},
    {"a_maximum_drawdown": "1"}, {"b_maximum_drawdown": "0.1234567890123456789"},
    {"source_kind": "verified_completed"}, {"evaluable_weeks": True},
    {"bull_weeks": 5},
])
def test_invalid_or_unbound_authority_does_not_gain_threshold_evaluation(change):
    with pytest.raises(ValueError):
        _compare(**change)


def test_accepted_v1_oos_rule_still_rejects_paired_drawdown_metric():
    metric = {"type": "artifact_ref", "artifact_type": "backtest_metric_profile",
              "schema_version": 1, "content_hash": "sha256:" + "1" * 64}
    with pytest.raises(ValueError, match="metric_key is not supported"):
        OosRule(metric, "maximum_drawdown", "fraction", "gte", "0", 1)
    assert _compare().formal_result != "supported"
