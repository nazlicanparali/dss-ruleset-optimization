"""Runs all the tests for each rule and builds the decision table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .decision import DecisionThresholds, decide_action, trend_status
from .stats_utils import (
    chi2_or_monte_carlo,
    cochran_armitage_trend,
    cramers_v,
    fit_logistic_irls,
    lift_metrics,
    mantel_haenszel,
    wilson_ci,
)


def analyze_rule(
    flag_series: pd.Series,
    outcome: pd.Series,
    rule_name: str,
    description: str = "",
    segment: pd.Series | None = None,
    thresholds: DecisionThresholds = DecisionThresholds(),
) -> dict:
    """All tests for one rule. flag_series is 0/1/2 (or 0/2 for a
    combination), outcome is 1 = bad. Returns one row of the decision table.

    The trend test needs at least 3 levels, so for combination rules the
    trend is "undetermined".
    """
    df = pd.DataFrame({"flag": flag_series, "outcome": outcome}).dropna()
    levels = sorted(df["flag"].unique())
    n_total = len(df)

    # per-level counts & Wilson CIs
    level_stats = {}
    for lvl in levels:
        sub = df[df["flag"] == lvl]
        bad = int(sub["outcome"].sum())
        n = len(sub)
        p_hat, lo, hi = wilson_ci(bad, n)
        level_stats[lvl] = {"n": n, "bad": bad, "rate": p_hat, "ci_lower": lo, "ci_upper": hi}

    # association: chi-square / Monte Carlo + Cramer's V
    table = pd.crosstab(df["flag"], df["outcome"]).reindex(columns=[0, 1], fill_value=0).values
    assoc = chi2_or_monte_carlo(table)
    v = cramers_v(table)

    # trend test (only meaningful with >= 3 ordered levels)
    if len(levels) >= 3:
        counts_bad = [level_stats[lvl]["bad"] for lvl in levels]
        counts_total = [level_stats[lvl]["n"] for lvl in levels]
        trend_result = cochran_armitage_trend(counts_bad, counts_total, scores=levels)
    else:
        trend_result = {"statistic": float("nan"), "p_value": float("nan"), "direction": "n/a (binary rule)"}
    trend = trend_status(trend_result, alpha=thresholds.significance_alpha)

    # lift of the worst level vs. the best level
    worst, best = max(levels), min(levels)
    lift = lift_metrics(
        level_stats[worst]["bad"], level_stats[worst]["n"], level_stats[best]["bad"], level_stats[best]["n"]
    )

    # logistic regression odds ratio (worst level vs. rest)
    x = (df["flag"] == worst).astype(float).values.reshape(-1, 1)
    y = df["outcome"].astype(float).values
    logit = fit_logistic_irls(x, y)
    or_point = float(logit["odds_ratios"][1])
    or_lo, or_hi = float(logit["or_ci_lower"][1]), float(logit["or_ci_upper"][1])

    # optional Mantel-Haenszel segment-adjusted OR
    mh = None
    if segment is not None:
        seg = segment.reindex(df.index)
        strata_tables = []
        for seg_value in seg.dropna().unique():
            mask = seg == seg_value
            sub = df[mask.reindex(df.index, fill_value=False)]
            a = int(((sub["flag"] == worst) & (sub["outcome"] == 1)).sum())
            b = int(((sub["flag"] == worst) & (sub["outcome"] == 0)).sum())
            c = int(((sub["flag"] != worst) & (sub["outcome"] == 1)).sum())
            d = int(((sub["flag"] != worst) & (sub["outcome"] == 0)).sum())
            if min(a, b, c, d) >= 0 and (a + b + c + d) > 0:
                strata_tables.append(np.array([[a, b], [c, d]]))
        if strata_tables:
            mh = mantel_haenszel(strata_tables)

    decision = decide_action(
        n_total=n_total,
        lift=lift["lift"],
        lift_ci_lower=lift["ci_lower"],
        p_value=assoc["p_value"],
        trend=trend,
        thresholds=thresholds,
    )

    return {
        "rule": rule_name,
        "description": description,
        "n_total": n_total,
        "levels": levels,
        "level_stats": level_stats,
        "association_method": assoc["method"],
        "association_p_value": assoc["p_value"],
        "cramers_v": v,
        "trend_p_value": trend_result["p_value"],
        "trend_status": trend,
        "lift": lift["lift"],
        "lift_ci_lower": lift["ci_lower"],
        "lift_ci_upper": lift["ci_upper"],
        "odds_ratio": or_point,
        "or_ci_lower": or_lo,
        "or_ci_upper": or_hi,
        "or_mantel_haenszel": mh["or_mantel_haenszel"] if mh else None,
        "mh_vs_raw_pct_change": (
            100 * abs(mh["or_mantel_haenszel"] - or_point) / or_point if mh and or_point else None
        ),
        "recommended_action": decision["action"],
        "rationale": decision["rationale"],
    }


def run_full_validation(
    rule_flags: pd.DataFrame,
    outcome: pd.Series,
    descriptions: dict[str, str] | None = None,
    segment: pd.Series | None = None,
    thresholds: DecisionThresholds = DecisionThresholds(),
) -> pd.DataFrame:
    """analyze_rule for every column, sorted by Cramer's V (strongest first)."""
    descriptions = descriptions or {}
    results = [
        analyze_rule(
            rule_flags[col],
            outcome,
            rule_name=col,
            description=descriptions.get(col, ""),
            segment=segment,
            thresholds=thresholds,
        )
        for col in rule_flags.columns
    ]
    table = pd.DataFrame(results)
    return table.sort_values("cramers_v", ascending=False).reset_index(drop=True)
