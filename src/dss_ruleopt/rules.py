"""Traffic-light rules.

A Rule turns a column into GREEN (0) / YELLOW (1) / RED (2). A RuleSet holds
several rules plus AND-combinations ("A and B are both red").
example_ruleset() is a demo on the Home Credit columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

GREEN, YELLOW, RED = 0, 1, 2
FLAG_NAMES = {GREEN: "green", YELLOW: "yellow", RED: "red"}


@dataclass
class Rule:
    """One traffic-light rule.

    Cut points are quantiles of the column itself, not fixed numbers.
    direction="low_is_risky" is for things like scores where low is bad.
    """
    name: str
    description: str
    column: str
    direction: str = "high_is_risky"  # or "low_is_risky"
    yellow_quantile: float = 0.75
    red_quantile: float = 0.90
    transform: Callable[[pd.Series], pd.Series] | None = None

    def flags(self, df: pd.DataFrame) -> pd.Series:
        series = df[self.column]
        if self.transform is not None:
            series = self.transform(series)
        series = series.astype(float)

        if self.direction == "high_is_risky":
            q_yellow = series.quantile(self.yellow_quantile)
            q_red = series.quantile(self.red_quantile)
            out = np.where(series >= q_red, RED, np.where(series >= q_yellow, YELLOW, GREEN))
        elif self.direction == "low_is_risky":
            q_yellow = series.quantile(1 - self.yellow_quantile)
            q_red = series.quantile(1 - self.red_quantile)
            out = np.where(series <= q_red, RED, np.where(series <= q_yellow, YELLOW, GREEN))
        else:
            raise ValueError(f"unknown direction: {self.direction!r}")

        return pd.Series(out, index=df.index, name=self.name)


@dataclass
class CombinationRule:
    """Fires (RED) only when both base rules are RED."""
    name: str
    description: str
    rule_a: str
    rule_b: str

    def flags(self, flags_df: pd.DataFrame) -> pd.Series:
        fired = (flags_df[self.rule_a] == RED) & (flags_df[self.rule_b] == RED)
        return pd.Series(np.where(fired, RED, GREEN), index=flags_df.index, name=self.name)


@dataclass
class RuleSet:
    rules: list[Rule] = field(default_factory=list)
    combinations: list[CombinationRule] = field(default_factory=list)

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a DataFrame with one ordinal (0/1/2) column per rule and combination."""
        flags = pd.DataFrame({rule.name: rule.flags(df) for rule in self.rules}, index=df.index)
        for combo in self.combinations:
            flags[combo.name] = combo.flags(flags)
        return flags


# --- example rules (Home Credit columns) ---
def _credit_to_income(df: pd.DataFrame) -> pd.Series:
    return df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)


def _annuity_to_income(df: pd.DataFrame) -> pd.Series:
    return df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)


def _years_employed(df: pd.DataFrame) -> pd.Series:
    # DAYS_EMPLOYED is negative; 365243 is a placeholder used for pensioners /
    # unemployed. I treat it as a very long tenure so it ends up green.
    days = df["DAYS_EMPLOYED"].copy()
    days = days.where(days != 365243, -365243 * 20)
    return -days / 365.25


def _external_score_composite(df: pd.DataFrame) -> pd.Series:
    return df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)


def example_ruleset() -> RuleSet:
    """5 rules + 3 combinations on Home Credit columns, quantile cut points.
    Just an example to run the pipeline on, not a credit policy."""
    rules = [
        Rule(
            name="R1_credit_to_income",
            description="Requested credit amount relative to declared income (AMT_CREDIT / AMT_INCOME_TOTAL).",
            column="_credit_to_income",
            direction="high_is_risky",
            yellow_quantile=0.75,
            red_quantile=0.90,
        ),
        Rule(
            name="R2_annuity_to_income",
            description="Annual repayment burden relative to declared income (AMT_ANNUITY / AMT_INCOME_TOTAL).",
            column="_annuity_to_income",
            direction="high_is_risky",
            yellow_quantile=0.75,
            red_quantile=0.90,
        ),
        Rule(
            name="R3_employment_tenure",
            description="Years of continuous employment at time of application (short tenure = higher risk).",
            column="_years_employed",
            direction="low_is_risky",
            yellow_quantile=0.75,
            red_quantile=0.90,
        ),
        Rule(
            name="R4_external_score",
            description="Composite of the three normalized external credit-bureau scores (low score = higher risk).",
            column="_external_score",
            direction="low_is_risky",
            yellow_quantile=0.75,
            red_quantile=0.90,
        ),
        Rule(
            name="R5_social_default_exposure",
            description="Count of the applicant's social circle observed to have defaulted within 30 days (DEF_30_CNT_SOCIAL_CIRCLE).",
            column="DEF_30_CNT_SOCIAL_CIRCLE",
            direction="high_is_risky",
            yellow_quantile=0.75,
            red_quantile=0.90,
        ),
    ]

    combinations = [
        CombinationRule(
            name="C1_credit_and_annuity_burden",
            description="Both the credit-to-income AND annuity-to-income rules fired red simultaneously.",
            rule_a="R1_credit_to_income",
            rule_b="R2_annuity_to_income",
        ),
        CombinationRule(
            name="C2_score_and_tenure",
            description="Both the external-score AND employment-tenure rules fired red simultaneously.",
            rule_a="R4_external_score",
            rule_b="R3_employment_tenure",
        ),
        CombinationRule(
            name="C3_burden_and_score",
            description="Both the annuity-burden AND external-score rules fired red simultaneously.",
            rule_a="R2_annuity_to_income",
            rule_b="R4_external_score",
        ),
    ]

    return RuleSet(rules=rules, combinations=combinations)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the ratio / score columns used by example_ruleset()."""
    df = df.copy()
    df["_credit_to_income"] = _credit_to_income(df)
    df["_annuity_to_income"] = _annuity_to_income(df)
    df["_years_employed"] = _years_employed(df)
    df["_external_score"] = _external_score_composite(df)
    return df
