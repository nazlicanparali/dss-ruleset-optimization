"""Turns the test results of a rule into a recommended action.

Plain if/else on a few thresholds (DecisionThresholds), so each decision can
be explained.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecisionThresholds:
    strong_lift: float = 2.0
    weak_lift: float = 1.2         # below this: no real signal
    min_reliable_n: int = 30
    significance_alpha: float = 0.05


def trend_status(cochran_armitage_result: dict, alpha: float = 0.05) -> str:
    """Label for the Cochran-Armitage result: increasing, non-increasing,
    no significant trend, or undetermined (binary rules)."""
    p = cochran_armitage_result.get("p_value", float("nan"))
    direction = cochran_armitage_result.get("direction", "undefined")

    if p != p:  # NaN
        return "undetermined"
    if p >= alpha:
        return "no_significant_trend"
    return "monotonic_increasing" if direction == "increasing" else "monotonic_non_increasing"


def decide_action(
    n_total: int,
    lift: float,
    lift_ci_lower: float,
    p_value: float,
    trend: str,
    thresholds: DecisionThresholds = DecisionThresholds(),
) -> dict:
    """Returns one of:

    automation_candidate   strong lift (CI lower bound too) and increasing trend
    manual_review          strong lift but CI or trend not clean
    recalibrate_thresholds right direction but weak, cut points should move
    do_not_use             too few rows, not significant, or lift ~ 1
    """
    reliable = n_total >= thresholds.min_reliable_n
    significant = p_value == p_value and p_value < thresholds.significance_alpha

    if not reliable:
        return _result("do_not_use", "Sample size below the minimum reliability threshold.", locals())

    if not significant:
        return _result("do_not_use", "No statistically significant association with the outcome.", locals())

    if lift >= thresholds.strong_lift and lift_ci_lower >= thresholds.strong_lift and trend == "monotonic_increasing":
        return _result(
            "automation_candidate",
            "Strong and monotonic signal.",
            locals(),
        )

    if lift >= thresholds.strong_lift:
        return _result(
            "manual_review",
            "Strong lift, but the CI or the trend is not clean enough.",
            locals(),
        )

    if lift >= thresholds.weak_lift:
        return _result(
            "recalibrate_thresholds",
            "Right direction but weak; try moving the cut points.",
            locals(),
        )

    return _result("do_not_use", "Lift is too close to 1 to be actionable.", locals())


def _result(action: str, rationale: str, ctx: dict) -> dict:
    return {
        "action": action,
        "rationale": rationale,
        "n_total": ctx["n_total"],
        "lift": ctx["lift"],
        "lift_ci_lower": ctx["lift_ci_lower"],
        "p_value": ctx["p_value"],
        "trend": ctx["trend"],
    }


def build_decision_table(rule_results: list[dict]) -> "pandas.DataFrame":  # noqa: F821
    """List of analyze_rule() results -> DataFrame."""
    import pandas as pd

    return pd.DataFrame(rule_results)
