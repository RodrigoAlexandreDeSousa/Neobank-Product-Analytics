"""
Unit tests for scripts/generate_data.py.

These pin down the data-generating contract the rest of the project relies on:
  - the experiment (variant) affects ONLY signup_completed
  - country/platform/time-of-day affect ONLY the downstream steps
  - the funnel and event log are internally consistent (ordering, uniqueness)

Run:  pytest tests/ -v
"""

import pandas as pd
import pytest

from scripts.generate_data import FUNNEL, generate

N = 50_000  # matches the project's canonical default; large enough to keep segment-level noise small
SEED = 42


@pytest.fixture(scope="module")
def dataset():
    return generate(n_users=N, seed=SEED)


def test_shapes(dataset):
    users, events = dataset
    assert len(users) == N
    assert users["user_id"].nunique() == N
    assert set(events["event_name"].unique()) <= set(FUNNEL)
    assert len(events) >= len(users)  # everyone reaches at least signup_started


def test_reproducibility():
    u1, e1 = generate(n_users=5_000, seed=7)
    u2, e2 = generate(n_users=5_000, seed=7)
    pd.testing.assert_frame_equal(u1, u2)
    pd.testing.assert_frame_equal(e1, e2)


def test_different_seed_gives_different_data():
    u1, _ = generate(n_users=5_000, seed=1)
    u2, _ = generate(n_users=5_000, seed=2)
    assert not u1["country"].equals(u2["country"])


def test_variant_split_is_roughly_balanced(dataset):
    users, _ = dataset
    share = users["variant"].value_counts(normalize=True)
    assert abs(share["control"] - 0.5) < 0.02
    assert abs(share["treatment"] - 0.5) < 0.02


def test_funnel_counts_are_monotonically_decreasing(dataset):
    users, events = dataset
    counts = [events.loc[events.event_name == step, "user_id"].nunique() for step in FUNNEL]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == len(users)


def test_event_timestamps_are_ordered_per_user(dataset):
    """Each user's own events must appear in funnel order with non-decreasing timestamps."""
    _, events = dataset
    order = {step: i for i, step in enumerate(FUNNEL)}
    events = events.copy()
    events["step_order"] = events["event_name"].map(order)
    events = events.sort_values(["user_id", "step_order"])
    is_increasing = events.groupby("user_id")["event_ts"].apply(lambda s: s.is_monotonic_increasing)
    assert is_increasing.all()


def test_event_ids_are_unique(dataset):
    _, events = dataset
    assert events["event_id"].is_unique


def test_treatment_lifts_signup_completed(dataset):
    """The core baked-in A/B effect: treatment must clearly beat control."""
    users, events = dataset
    completed = set(events.loc[events.event_name == "signup_completed", "user_id"])
    users = users.copy()
    users["completed"] = users["user_id"].isin(completed)
    rates = users.groupby("variant")["completed"].mean()
    assert rates["treatment"] > rates["control"]
    assert rates["treatment"] - rates["control"] > 0.03  # well above sampling noise at n=50k


def test_signup_completed_is_not_driven_by_country_or_platform(dataset):
    """
    Only `variant` should move signup_completed -- the country/platform modifiers
    in generate_data.py apply strictly downstream (kyc/deposit/transaction).
    """
    users, events = dataset
    completed = set(events.loc[events.event_name == "signup_completed", "user_id"])
    users = users.copy()
    users["completed"] = users["user_id"].isin(completed)

    by_country = users.groupby("country")["completed"].mean()
    by_platform = users.groupby("platform")["completed"].mean()
    # true effect is zero here; at n=50k the spread should stay small (observed ~1.5pp / ~0.8pp)
    assert by_country.max() - by_country.min() < 0.03
    assert by_platform.max() - by_platform.min() < 0.03


def test_downstream_activation_does_vary_by_segment(dataset):
    """Sanity check the opposite: country/platform DO move the downstream steps."""
    users, events = dataset
    activated = set(events.loc[events.event_name == "first_transaction", "user_id"])
    users = users.copy()
    users["activated"] = users["user_id"].isin(activated)

    by_country = users.groupby("country")["activated"].mean()
    by_platform = users.groupby("platform")["activated"].mean()
    assert by_country.max() - by_country.min() > 0.02
    assert by_platform.max() - by_platform.min() > 0.02
