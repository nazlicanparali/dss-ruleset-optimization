"""Loads data/application_train.csv (Home Credit, Kaggle) if it exists.
Otherwise generates a synthetic dataset with the same columns so the demo
and the tests run without downloading anything."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REAL_FILE = DATA_DIR / "application_train.csv"

REQUIRED_COLUMNS = [
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "NAME_INCOME_TYPE",
]


def load_application_data(n_synthetic: int = 20_000, random_state: int = 42) -> tuple[pd.DataFrame, bool]:
    """Returns (df, is_real). is_real is False for synthetic data."""
    if REAL_FILE.exists():
        df = pd.read_csv(REAL_FILE)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{REAL_FILE} is missing columns {missing}, "
                "maybe the download was incomplete"
            )
        return df, True

    return _generate_synthetic(n_synthetic, random_state), False


def _generate_synthetic(n: int, random_state: int) -> pd.DataFrame:
    """Random data with Home Credit columns and a built-in risk signal,
    so the pipeline has something to find."""
    rng = np.random.default_rng(random_state)

    income = rng.lognormal(mean=11.8, sigma=0.5, size=n)
    credit = income * rng.uniform(1.5, 8.0, size=n)
    annuity = credit * rng.uniform(0.03, 0.12, size=n)

    days_employed = -rng.exponential(scale=1800, size=n).astype(int)
    pensioner_mask = rng.random(n) < 0.18
    days_employed = np.where(pensioner_mask, 365243, days_employed)

    ext_1 = np.clip(rng.normal(0.5, 0.18, size=n), 0, 1)
    ext_2 = np.clip(rng.normal(0.5, 0.18, size=n), 0, 1)
    ext_3 = np.clip(rng.normal(0.5, 0.18, size=n), 0, 1)

    social_default_exposure = rng.poisson(lam=0.3, size=n)

    income_type = rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
        size=n,
        p=[0.51, 0.23, 0.18, 0.07, 0.01],
    )

    # default prob goes up with credit/income, annuity/income, short tenure, low score
    credit_to_income = credit / income
    annuity_to_income = annuity / income
    years_employed = np.where(days_employed == 365243, 40, -days_employed / 365.25)
    score_composite = (ext_1 + ext_2 + ext_3) / 3

    logit = (
        -2.6
        + 0.35 * (credit_to_income - credit_to_income.mean()) / credit_to_income.std()
        + 0.25 * (annuity_to_income - annuity_to_income.mean()) / annuity_to_income.std()
        - 0.30 * (years_employed - years_employed.mean()) / years_employed.std()
        - 0.55 * (score_composite - score_composite.mean()) / score_composite.std()
        + 0.15 * (social_default_exposure - social_default_exposure.mean())
        + rng.normal(0, 0.4, size=n)
    )
    p_default = 1 / (1 + np.exp(-logit))
    target = (rng.random(n) < p_default).astype(int)

    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100_000, 100_000 + n),
            "TARGET": target,
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "DAYS_EMPLOYED": days_employed,
            "EXT_SOURCE_1": ext_1,
            "EXT_SOURCE_2": ext_2,
            "EXT_SOURCE_3": ext_3,
            "DEF_30_CNT_SOCIAL_CIRCLE": social_default_exposure,
            "NAME_INCOME_TYPE": income_type,
        }
    )
