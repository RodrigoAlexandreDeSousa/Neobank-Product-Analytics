"""
Unit tests for the feature engineering in ml/train_activation_model.py.

These run against data/raw_users.csv + data/raw_events.csv, so they need the
dataset to exist first:  python scripts/generate_data.py
They're skipped automatically if it doesn't (e.g. on a fresh checkout before
Phase 1 of the README has been run).

Run:  pytest tests/ -v
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("sklearn")  # the module imports sklearn at the top level

from ml.train_activation_model import CATEGORICAL, NUMERIC, TARGET, features_from_local

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "raw_users.csv").exists(),
    reason="data/raw_users.csv missing -- run `python scripts/generate_data.py` first",
)


@pytest.fixture(scope="module")
def features():
    # features_from_local() reads relative paths ("data/raw_users.csv"), so run
    # it from the project root and restore the original cwd afterwards. (Not
    # using the `monkeypatch` fixture here -- it's function-scoped and this
    # fixture is module-scoped.)
    original_cwd = os.getcwd()
    os.chdir(DATA_DIR.parent)
    try:
        yield features_from_local()
    finally:
        os.chdir(original_cwd)


def test_population_is_subset_of_all_users(features):
    import pandas as pd

    users = pd.read_csv(DATA_DIR / "raw_users.csv")
    assert len(features) <= len(users)
    assert len(features) > 0


def test_no_leakage_columns_present(features):
    leaky = {"reached_kyc_submitted", "reached_first_deposit", "kyc_submitted", "first_deposit"}
    assert leaky.isdisjoint(features.columns)


def test_required_columns_present(features):
    for col in CATEGORICAL + NUMERIC + [TARGET]:
        assert col in features.columns


def test_target_is_binary(features):
    assert set(features[TARGET].unique()) <= {0, 1}


def test_no_missing_values(features):
    cols = CATEGORICAL + NUMERIC + [TARGET]
    assert features[cols].isnull().sum().sum() == 0


def test_minutes_signup_to_complete_is_non_negative(features):
    assert (features["minutes_signup_to_complete"] >= 0).all()


def test_target_prevalence_is_reasonable(features):
    """Sanity bound, not a precise check -- catches a badly broken join/filter."""
    prevalence = features[TARGET].mean()
    assert 0.05 < prevalence < 0.95
