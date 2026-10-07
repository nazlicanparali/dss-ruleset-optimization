"""Statistical tests used by pipeline.py. Nothing domain-specific here."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a proportion. Returns (p_hat, lower, upper).

    Better than the normal approximation when n is small or p is close to 0/1,
    which happens a lot for rules that rarely fire.
    """
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    p_hat = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)

    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (p_hat, max(0.0, lower), min(1.0, upper))


def cramers_v(table: np.ndarray) -> float:
    """Cramer's V for an r x c table (0 = no association, 1 = perfect).

    Used to compare rules that fired on very different numbers of rows,
    where p-values alone are not comparable.
    """
    table = np.asarray(table, dtype=float)
    n = table.sum()
    if n == 0:
        return float("nan")
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    r, c = table.shape
    return float(np.sqrt((chi2 / n) / (min(r - 1, c - 1) or 1)))


def chi2_or_monte_carlo(
    table: np.ndarray,
    min_expected_count: float = 5.0,
    n_simulations: int = 100_000,
    random_state: int | None = 42,
) -> dict:
    """Pearson chi-square test. If an expected count is below
    `min_expected_count`, the p-value is simulated instead (random tables
    with the same margins)."""
    table = np.asarray(table, dtype=float)
    chi2, p_asymptotic, dof, expected = stats.chi2_contingency(table, correction=False)

    if expected.min() >= min_expected_count:
        return {
            "method": "chi2",
            "statistic": float(chi2),
            "p_value": float(p_asymptotic),
            "dof": int(dof),
            "min_expected_count": float(expected.min()),
        }

    rng = np.random.default_rng(random_state)
    row_totals = table.sum(axis=1).astype(int)
    col_totals = table.sum(axis=0).astype(int)
    n = int(table.sum())

    # tables with fixed margins under independence
    extreme_count = 0
    for _ in range(n_simulations):
        sim = stats.random_table(row_totals, col_totals, seed=rng).rvs()
        sim_chi2, _, _, _ = stats.chi2_contingency(sim, correction=False)
        if sim_chi2 >= chi2 - 1e-9:
            extreme_count += 1

    p_mc = (extreme_count + 1) / (n_simulations + 1)
    return {
        "method": "monte_carlo",
        "statistic": float(chi2),
        "p_value": float(p_mc),
        "dof": int(dof),
        "min_expected_count": float(expected.min()),
        "n_simulations": n_simulations,
    }


def cochran_armitage_trend(
    counts_bad: Sequence[int], counts_total: Sequence[int], scores: Sequence[float] | None = None
) -> dict:
    """Cochran-Armitage test for a trend in the bad rate across ordered
    categories (green -> yellow -> red).

    A rule can be significant in the chi-square test and still be badly
    ordered, e.g. yellow riskier than red. This test catches that.
    """
    counts_bad = np.asarray(counts_bad, dtype=float)
    counts_total = np.asarray(counts_total, dtype=float)
    if scores is None:
        scores = np.arange(len(counts_bad), dtype=float)
    else:
        scores = np.asarray(scores, dtype=float)

    n = counts_total.sum()
    p_bar = counts_bad.sum() / n
    score_bar = (scores * counts_total).sum() / n

    numerator = (counts_bad - counts_total * p_bar) @ (scores - score_bar)
    denominator = p_bar * (1 - p_bar) * ((counts_total * (scores - score_bar) ** 2).sum())

    if denominator <= 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "direction": "undefined"}

    z = numerator / np.sqrt(denominator)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    direction = "increasing" if z > 0 else ("decreasing" if z < 0 else "flat")
    return {"statistic": float(z), "p_value": float(p_value), "direction": direction}


def fit_logistic_irls(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None, max_iter: int = 50, tol: float = 1e-8
) -> dict:
    """Logistic regression with intercept, fitted by IRLS.

    Works on row-level data, or on grouped data if `weights` are the counts
    and y is the rate. Returns coefficients, SEs, Wald p-values and odds
    ratios with 95% CI. Written by hand to avoid a statsmodels dependency.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    beta = np.zeros(Xd.shape[1])
    for _ in range(max_iter):
        eta = Xd @ beta
        mu = 1 / (1 + np.exp(-eta))
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        W = w * mu * (1 - mu)
        z = eta + (y - mu) / (mu * (1 - mu))

        WX = Xd * W[:, None]
        XtWX = Xd.T @ WX
        XtWz = Xd.T @ (W * z)
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    eta = Xd @ beta
    mu = 1 / (1 + np.exp(-eta))
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    W = w * mu * (1 - mu)
    XtWX = Xd.T @ (Xd * W[:, None])
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(XtWX)
    se = np.sqrt(np.diag(cov))

    z_stats = beta / se
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
    odds_ratios = np.exp(beta)
    ci_lower = np.exp(beta - 1.96 * se)
    ci_upper = np.exp(beta + 1.96 * se)

    return {
        "coefficients": beta,
        "std_errors": se,
        "z_stats": z_stats,
        "p_values": p_values,
        "odds_ratios": odds_ratios,
        "or_ci_lower": ci_lower,
        "or_ci_upper": ci_upper,
    }


def fisher_odds_ratio(a: int, b: int, c: int, d: int) -> dict:
    """Fisher exact test for a 2x2 table [[a, b], [c, d]]
    (rows = groups, columns = bad / good). For small or sparse tables."""
    table = np.array([[a, b], [c, d]])
    odds_ratio, p_value = stats.fisher_exact(table)
    return {"odds_ratio": float(odds_ratio), "p_value": float(p_value)}


def mantel_haenszel(strata_tables: Sequence[np.ndarray]) -> dict:
    """Mantel-Haenszel pooled odds ratio over strata, and the CMH test.

    `strata_tables` is a list of 2x2 arrays [[a, b], [c, d]], one per segment.
    If the pooled OR is far (~10%+) from the raw OR, the segment is a
    confounder.
    """
    num_or = 0.0
    den_or = 0.0
    num_chi = 0.0
    var_chi = 0.0

    for table in strata_tables:
        a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
        n = a + b + c + d
        if n == 0:
            continue
        num_or += (a * d) / n
        den_or += (b * c) / n

        row1, row2 = a + b, c + d
        col1, col2 = a + c, b + d
        num_chi += a - (row1 * col1) / n
        var_chi += (row1 * row2 * col1 * col2) / (n**2 * (n - 1)) if n > 1 else 0.0

    or_mh = num_or / den_or if den_or > 0 else float("nan")
    chi2_mh = (abs(num_chi) - 0.5) ** 2 / var_chi if var_chi > 0 else float("nan")
    p_value = 1 - stats.chi2.cdf(chi2_mh, df=1) if not np.isnan(chi2_mh) else float("nan")

    return {"or_mantel_haenszel": float(or_mh), "chi2": float(chi2_mh), "p_value": float(p_value)}


def lift_metrics(bad_in_group: int, n_in_group: int, bad_in_baseline: int, n_in_baseline: int) -> dict:
    """Lift (relative risk) of the bad rate in a group vs. a baseline group,
    with a 95% CI on the log scale."""
    rate_group = bad_in_group / n_in_group if n_in_group else float("nan")
    rate_baseline = bad_in_baseline / n_in_baseline if n_in_baseline else float("nan")
    lift = rate_group / rate_baseline if rate_baseline else float("nan")

    if bad_in_group == 0 or bad_in_baseline == 0 or n_in_group == 0 or n_in_baseline == 0:
        return {"lift": lift, "rate_group": rate_group, "rate_baseline": rate_baseline,
                "ci_lower": float("nan"), "ci_upper": float("nan")}

    log_lift = np.log(lift)
    se_log = np.sqrt(
        (1 / bad_in_group) - (1 / n_in_group) + (1 / bad_in_baseline) - (1 / n_in_baseline)
    )
    ci_lower = float(np.exp(log_lift - 1.96 * se_log))
    ci_upper = float(np.exp(log_lift + 1.96 * se_log))

    return {
        "lift": float(lift),
        "rate_group": float(rate_group),
        "rate_baseline": float(rate_baseline),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
