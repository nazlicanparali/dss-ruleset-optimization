# Rule set validation for traffic-light decision systems

Many credit and fraud processes use simple business rules that put each
application into green / yellow / red. This project checks such rules
against a known outcome (default or not) and suggests what to do with each
rule.

The idea comes from a credit rule validation project I worked on. The code
here is rebuilt on public data (Kaggle Home Credit Default Risk). The rules
in `rules.py` are my own examples with quantile cut points, not real
business rules.

## What it checks

For every rule and every AND-combination of rules:

- **Is it related to the outcome?** Chi-square test. If some expected counts
  are below 5, the p-value is simulated instead.
- **How strong is it?** Cramér's V, lift of red vs. green with a CI, and the
  odds ratio from a logistic regression (IRLS, written by hand).
- **Is the order right?** Cochran-Armitage trend test. A rule can be
  significant but still have yellow riskier than red.
- **Does it hold within segments?** Mantel-Haenszel odds ratio by income
  type, compared to the raw odds ratio.

Then `decision.py` maps the results to one of four actions:
`automation_candidate`, `manual_review`, `recalibrate_thresholds`,
`do_not_use`. The thresholds are in `DecisionThresholds` and can be changed.

## Structure

```
src/dss_ruleopt/
    stats_utils.py   statistical tests
    rules.py         Rule / RuleSet / CombinationRule + example rule set
    pipeline.py      runs all tests per rule, builds the decision table
    decision.py      test results -> recommended action
    data.py          loads the Kaggle file, or makes synthetic data if it's missing
scripts/
    download_kaggle_data.py
    run_demo.py
tests/
```

## Running it

```bash
git clone https://github.com/nazlicanparali/dss-ruleset-optimization.git
cd dss-ruleset-optimization
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python scripts/run_demo.py
pytest
```

Without `data/application_train.csv` the demo uses synthetic data with the
same columns, so it runs without a download. To use the real data see
`data/README.md`.

## Results

### Synthetic data (20,000 rows, seed 42)

The synthetic data has a signal I put in on purpose, so this only shows that
the pipeline finds it.

```
                        rule  n_total association_method association_p_value  cramers_v         trend_status  lift  odds_ratio  or_mantel_haenszel   recommended_action
           R4_external_score    20000               chi2             4.8e-86      0.140 monotonic_increasing 2.694       2.698               2.694 automation_candidate
         R1_credit_to_income    20000               chi2             4.7e-65      0.122 monotonic_increasing 2.355       2.305               2.306 automation_candidate
        R2_annuity_to_income    20000               chi2             1.7e-61      0.118 monotonic_increasing 2.390       2.429               2.442 automation_candidate
C1_credit_and_annuity_burden    20000               chi2             3.5e-36      0.089         undetermined 2.442       2.856               2.863        manual_review
         C3_burden_and_score    20000               chi2             1.3e-35      0.088         undetermined 3.672       5.083               5.042        manual_review
         C2_score_and_tenure    20000               chi2             1.4e-08      0.040         undetermined 2.248       2.589               2.606        manual_review
        R3_employment_tenure    20000               chi2             6.2e-05      0.031 monotonic_increasing 1.179       1.145               1.146           do_not_use
  R5_social_default_exposure    20000               chi2             9.1e-02      0.012         undetermined 1.085       1.095               1.096           do_not_use
```

Combination rules have only two levels (fired / not fired), so the trend
test can't run on them and they show `undetermined`.

### Kaggle data (application_train.csv, 307,511 rows)

```
                        rule  n_total association_method  association_p_value  cramers_v              trend_status   lift  odds_ratio  or_mantel_haenszel   recommended_action
           R4_external_score   307511                chi2               <0.001      0.213     monotonic_increasing  4.582       4.335                4.209 automation_candidate
         C2_score_and_tenure   307511                chi2               <0.001      0.077             undetermined  3.224       3.974                3.635        manual_review
        R3_employment_tenure   307511                chi2               <0.001      0.066     monotonic_increasing  1.554       1.460                1.340 recalibrate_thresholds
         C3_burden_and_score   307511                chi2               <0.001      0.056             undetermined  2.966       3.570                3.614        manual_review
  R5_social_default_exposure   307511                chi2               <0.001      0.033     monotonic_increasing  2.977       1.394                1.398 automation_candidate
         R1_credit_to_income   307511                chi2               <0.001      0.016 monotonic_non_increasing  0.851       0.854                0.880           do_not_use
        R2_annuity_to_income   307511                chi2               <0.001      0.012     monotonic_increasing  1.036       1.018                1.044           do_not_use
C1_credit_and_annuity_burden   307511                chi2                0.002      0.006             undetermined  0.918       0.912                0.948           do_not_use
```

What I found interesting:

- `R1_credit_to_income` goes the wrong way. A high credit-to-income ratio
  has a slightly *lower* default rate (lift 0.85). With 300k rows every
  p-value is tiny, so the p-value alone would not show this. The trend test
  and the lift do.
- The external score rule is much stronger than the others (V = 0.21, the
  rest are 0.01–0.08). Its Mantel-Haenszel OR (4.21) is close to the raw OR
  (4.34), so the effect is not coming from the income type mix.
- `R5` comes out as an automation candidate (lift 2.98 red vs. green), but
  its V is small and its odds ratio vs. all other applications is only 1.39.
  Before automating it I would check how many applications are actually red.
  Coverage is not part of the decision rules yet.

## Using it with other rules

```python
from dss_ruleopt.pipeline import run_full_validation

table = run_full_validation(
    rule_flags=flags_df,      # one column per rule, values 0/1/2
    outcome=y,                # 1 = bad
    segment=segment_series,   # optional
)
```

## License

MIT
