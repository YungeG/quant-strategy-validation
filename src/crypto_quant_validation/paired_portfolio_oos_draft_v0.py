"""DRAFT A/B metric-threshold semantics, not an accepted ValidationPlan/Report.

Numeric inputs have NO verified Backtest publication or reserved holdout source here.
Threshold math may be unit-tested; formal result stays inconclusive by design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext

from .integration import _canonical_decimal


@dataclass(frozen=True, slots=True)
class DraftPairedPortfolioRuleV0:
    minimum_evaluable_weeks: int
    minimum_bull_weeks: int

    def __post_init__(self) -> None:
        if (type(self.minimum_evaluable_weeks) is not int or self.minimum_evaluable_weeks <= 0
                or type(self.minimum_bull_weeks) is not int or self.minimum_bull_weeks <= 0
                or self.minimum_bull_weeks > self.minimum_evaluable_weeks):
            raise ValueError("draft pair requires precommitted positive independent week minima")

    def to_canonical_dict(self) -> dict[str, object]:
        return {"type": "draft_paired_portfolio_oos_rule", "schema_version": 0,
                "minimum_evaluable_weeks": self.minimum_evaluable_weeks,
                "minimum_bull_weeks": self.minimum_bull_weeks,
                "accepted_rule": False}


@dataclass(frozen=True, slots=True)
class DraftPairedPortfolioThresholdsV0:
    rule: DraftPairedPortfolioRuleV0
    threshold_outcome: str
    net_return_difference: str | None
    a_maximum_drawdown: str | None
    b_maximum_drawdown: str | None
    formal_result: str = field(default="inconclusive", init=False)
    validation_eligible: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if (type(self.rule) is not DraftPairedPortfolioRuleV0
                or self.threshold_outcome not in {"would_pass", "would_fail", "insufficient_sample", "missing_metrics"}
                or (self.threshold_outcome == "missing_metrics") != (self.net_return_difference is None)):
            raise ValueError("draft paired thresholds lack an exact unadmitted outcome")

    def to_canonical_dict(self) -> dict[str, object]:
        return {"type": "draft_paired_portfolio_thresholds", "schema_version": 0,
                "rule": self.rule, "threshold_outcome": self.threshold_outcome,
                "net_return_difference": self.net_return_difference,
                "a_maximum_drawdown": self.a_maximum_drawdown,
                "b_maximum_drawdown": self.b_maximum_drawdown,
                "formal_result": "inconclusive", "validation_eligible": False}


def _fraction(value: str, name: str, *, drawdown: bool) -> Decimal:
    raw = _canonical_decimal(value, name)
    if "." in raw and len(raw.rsplit(".", 1)[1]) > 18:
        raise ValueError(f"{name} exceeds Backtest's 18-digit fraction policy")
    amount = Decimal(raw)
    if (drawdown and not Decimal(0) <= amount < Decimal(1)
            or not drawdown and amount <= Decimal(-1)):
        raise ValueError(f"{name} is outside admissible positive-equity fraction bounds")
    return amount


def evaluate_draft_paired_portfolio_thresholds_v0(
    *, rule: DraftPairedPortfolioRuleV0,
    a_net_return: str | None, b_net_return: str | None,
    a_maximum_drawdown: str | None, b_maximum_drawdown: str | None,
    evaluable_weeks: int, bull_weeks: int,
    source_kind: str = "synthetic_development",
) -> DraftPairedPortfolioThresholdsV0:
    """A/B boundary math only. No caller can convert this to ValidationReport."""
    if type(rule) is not DraftPairedPortfolioRuleV0:
        raise TypeError("draft paired rule type mismatch")
    if source_kind not in {"synthetic_development", "unverified"}:
        raise ValueError("draft pair cannot accept unbound verified publication/holdout claims")
    if (type(evaluable_weeks) is not int or evaluable_weeks < 0
            or type(bull_weeks) is not int or bull_weeks < 0
            or bull_weeks > evaluable_weeks):
        raise ValueError("draft pair week counts must be bounded nonnegative integers")
    if (a_net_return is None or b_net_return is None
            or a_maximum_drawdown is None or b_maximum_drawdown is None):
        return DraftPairedPortfolioThresholdsV0(rule, "missing_metrics", None, None, None)
    if not all(type(value) is str for value in (
            a_net_return, b_net_return, a_maximum_drawdown, b_maximum_drawdown)):
        raise TypeError("draft pair metrics must be exact canonical strings")
    a_return = _fraction(a_net_return, "A net return", drawdown=False)
    b_return = _fraction(b_net_return, "B net return", drawdown=False)
    a_dd = _fraction(a_maximum_drawdown, "A maximum drawdown", drawdown=True)
    b_dd = _fraction(b_maximum_drawdown, "B maximum drawdown", drawdown=True)
    with localcontext() as context:
        context.prec = max(len(a_net_return), len(b_net_return)) + 40
        delta = b_return - a_return
    text = format(delta, "f") if delta else "0"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    outcome = ("insufficient_sample" if evaluable_weeks < rule.minimum_evaluable_weeks
               or bull_weeks < rule.minimum_bull_weeks else
               "would_pass" if delta >= 0 and b_dd < a_dd else "would_fail")
    return DraftPairedPortfolioThresholdsV0(
        rule, outcome, text, a_maximum_drawdown, b_maximum_drawdown)
