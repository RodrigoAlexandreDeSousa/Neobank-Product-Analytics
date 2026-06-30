"""
Unit tests for the statistics in analysis/ab_test.py.

Run:  pytest tests/ -v
"""

import pytest

from analysis.ab_test import two_proportion_z_test


def test_identical_rates_give_no_effect():
    p_c, p_t, diff, z, p_value, ci_low, ci_high, power = two_proportion_z_test(500, 1000, 500, 1000)
    assert diff == pytest.approx(0.0)
    assert z == pytest.approx(0.0, abs=1e-9)
    assert p_value == pytest.approx(1.0, abs=1e-9)
    assert ci_low < 0 < ci_high


def test_known_result_matches_manual_calculation():
    """Regression test pinned to this project's own results (50k users, seed 42)."""
    p_c, p_t, diff, z, p_value, ci_low, ci_high, power = two_proportion_z_test(
        15387, 24814, 17102, 25186
    )
    assert p_c == pytest.approx(0.6201, abs=1e-3)
    assert p_t == pytest.approx(0.6790, abs=1e-3)
    assert diff == pytest.approx(0.0589, abs=1e-3)
    assert z == pytest.approx(13.81, abs=0.05)
    assert p_value < 0.001
    assert ci_low == pytest.approx(0.0506, abs=1e-3)
    assert ci_high == pytest.approx(0.0673, abs=1e-3)
    assert power > 0.99


def test_confidence_interval_widens_with_smaller_samples():
    *_, ci_low_big, ci_high_big, _ = two_proportion_z_test(620, 1000, 680, 1000)
    *_, ci_low_small, ci_high_small, _ = two_proportion_z_test(62, 100, 68, 100)
    assert (ci_high_small - ci_low_small) > (ci_high_big - ci_low_big)


def test_same_observed_rates_not_significant_on_tiny_sample():
    """The same ~6pp gap that is highly significant at n=25k should not be at n=10."""
    *_, p_value, ci_low, ci_high, _ = two_proportion_z_test(6, 10, 7, 10)
    assert p_value > 0.05 or ci_low < 0


def test_negative_lift_gives_negative_diff_and_ci_below_zero_when_clear():
    p_c, p_t, diff, z, p_value, ci_low, ci_high, power = two_proportion_z_test(
        7000, 10000, 6000, 10000
    )
    assert diff < 0
    assert ci_high < 0
    assert p_value < 0.001
