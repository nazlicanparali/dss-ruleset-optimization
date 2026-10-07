"""End-to-end test on synthetic data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dss_ruleopt.data import _generate_synthetic, load_application_data
from dss_ruleopt.rules import add_derived_columns, example_ruleset
from dss_ruleopt.pipeline import run_full_validation


def test_synthetic_generator_produces_expected_schema():
    # call the generator directly, load_application_data() would pick up the real csv if it exists
    from dss_ruleopt.data import REQUIRED_COLUMNS

    df = _generate_synthetic(n=5000, random_state=1)
    assert len(df) == 5000
    assert set(REQUIRED_COLUMNS) <= set(df.columns)
    assert df["TARGET"].isin([0, 1]).all()
    assert 0 < df["TARGET"].mean() < 1


def test_full_pipeline_runs_end_to_end():
    df, _is_real = load_application_data(n_synthetic=5000, random_state=1)

    df = add_derived_columns(df)
    ruleset = example_ruleset()
    flags = ruleset.evaluate(df)

    for rule in ruleset.rules:
        assert set(flags[rule.name].unique()) <= {0, 1, 2}
    for combo in ruleset.combinations:
        assert set(flags[combo.name].unique()) <= {0, 2}

    table = run_full_validation(flags, df["TARGET"], segment=df["NAME_INCOME_TYPE"])

    assert len(table) == len(ruleset.rules) + len(ruleset.combinations)
    assert set(table["recommended_action"]) <= {
        "automation_candidate",
        "manual_review",
        "recalibrate_thresholds",
        "do_not_use",
    }
    # synthetic data has a real signal, at least one rule should pick it up
    assert table["cramers_v"].max() > 0.02
