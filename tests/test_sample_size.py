"""
Unit tests for analysis/sample_size.py.

Run:  pytest tests/ -v
"""

import pytest

from analysis.sample_size import minimum_detectable_effect, n_per_group


def test_known_textbook_value():
    """p1=0.62, p2=0.65, alpha=0.05, power=0.80 -> ~4039 per group (verified by hand with
    the closed-form Wald formula; see analysis/sample_size.py docstring)."""
    n = n_per_group(0.62, 0.65, alpha=0.05, power=0.80)
    assert n == pytest.approx(4039, rel=0.01)


def test_larger_effect_needs_fewer_users():
    n_small_effect = n_per_group(0.62, 0.64, alpha=0.05, power=0.80)
    n_large_effect = n_per_group(0.62, 0.70, alpha=0.05, power=0.80)
    assert n_large_effect < n_small_effect


def test_higher_power_needs_more_users():
    n_80 = n_per_group(0.62, 0.65, alpha=0.05, power=0.80)
    n_95 = n_per_group(0.62, 0.65, alpha=0.05, power=0.95)
    assert n_95 > n_80


def test_stricter_alpha_needs_more_users():
    n_05 = n_per_group(0.62, 0.65, alpha=0.05, power=0.80)
    n_01 = n_per_group(0.62, 0.65, alpha=0.01, power=0.80)
    assert n_01 > n_05


def test_unequal_allocation_changes_control_n():
    n_equal = n_per_group(0.62, 0.65, alpha=0.05, power=0.80, ratio=1.0)
    n_skewed = n_per_group(0.62, 0.65, alpha=0.05, power=0.80, ratio=2.0)
    # giving treatment a bigger share lets the control arm shrink
    assert n_skewed < n_equal


def test_minimum_detectable_effect_round_trips_with_n_per_group():
    n = n_per_group(0.62, 0.65, alpha=0.05, power=0.80)
    mde = minimum_detectable_effect(0.62, n, alpha=0.05, power=0.80)
    assert mde == pytest.approx(0.03, abs=0.001)
