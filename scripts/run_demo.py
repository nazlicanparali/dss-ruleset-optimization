#!/usr/bin/env python3
"""Runs the example rule set and prints the decision table.

    python scripts/run_demo.py
    python scripts/run_demo.py --export decision_table.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dss_ruleopt.data import load_application_data
from dss_ruleopt.rules import add_derived_columns, example_ruleset
from dss_ruleopt.pipeline import run_full_validation

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=str, default=None, help="optional path to write the decision table as CSV")
    args = parser.parse_args()

    df, is_real = load_application_data()
    print(f"{len(df):,} rows ({'Kaggle data' if is_real else 'synthetic data, data/application_train.csv not found'})\n")

    df = add_derived_columns(df)
    ruleset = example_ruleset()
    flags = ruleset.evaluate(df)

    descriptions = {r.name: r.description for r in ruleset.rules}
    descriptions.update({c.name: c.description for c in ruleset.combinations})

    decision_table = run_full_validation(
        rule_flags=flags,
        outcome=df["TARGET"],
        descriptions=descriptions,
        segment=df["NAME_INCOME_TYPE"],
    )

    display_cols = [
        "rule",
        "n_total",
        "association_method",
        "association_p_value",
        "cramers_v",
        "trend_status",
        "lift",
        "odds_ratio",
        "or_mantel_haenszel",
        "recommended_action",
    ]
    show = decision_table[display_cols].copy()
    show["association_p_value"] = show["association_p_value"].map(lambda p: f"{p:.1e}")
    show = show.round(3)
    print(show.to_string(index=False))

    if args.export:
        decision_table.to_csv(args.export, index=False)
        print(f"\nsaved to {args.export}")


if __name__ == "__main__":
    main()
