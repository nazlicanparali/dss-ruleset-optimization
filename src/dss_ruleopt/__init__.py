"""Statistical checks for traffic-light (green/yellow/red) decision rules.

Takes a binary outcome and a set of rule flags and reports, per rule, whether
the rule is related to the outcome, how strong the effect is and whether it
still holds within segments.
"""

from .stats_utils import (
    wilson_ci,
    cramers_v,
    chi2_or_monte_carlo,
    cochran_armitage_trend,
    fit_logistic_irls,
    fisher_odds_ratio,
    mantel_haenszel,
    lift_metrics,
)
from .rules import Rule, RuleSet
from .decision import DecisionThresholds, decide_action, build_decision_table

__all__ = [
    "wilson_ci",
    "cramers_v",
    "chi2_or_monte_carlo",
    "cochran_armitage_trend",
    "fit_logistic_irls",
    "fisher_odds_ratio",
    "mantel_haenszel",
    "lift_metrics",
    "Rule",
    "RuleSet",
    "DecisionThresholds",
    "decide_action",
    "build_decision_table",
]

__version__ = "0.1.0"
