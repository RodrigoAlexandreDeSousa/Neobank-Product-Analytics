"""
A priori sample-size / power calculator for a two-proportion experiment.

This is the calculation a data scientist runs *before* launching an A/B test:
"given our baseline conversion rate, how many users per arm do we need to
reliably detect a lift of X, at significance alpha and power (1 - beta)?"

It is the natural companion to analysis/ab_test.py, which evaluates the
experiment *after* the data has been collected (achieved power, post-hoc).
Running this first is what makes a test design defensible: it stops you from
either under-powering a test (false negative risk) or running it for far
longer than necessary (wasted exposure / opportunity cost).

Formula (two-sided, equal allocation by default; Wald / normal approximation):

    n_per_group = (z_(alpha/2) + z_(power))^2 * (p1*(1-p1) + p2*(1-p2)) / (p1 - p2)^2

For unequal allocation (--ratio = n_treatment / n_control != 1) the standard
Wald correction is applied. See e.g. Wikipedia: "Sample size determination",
or Kohavi et al., "Trustworthy Online Controlled Experiments" (2020), ch. 3.

Run
---
  # Absolute MDE: "can we detect a +2pp lift over a 62% baseline?"
  python analysis/sample_size.py --baseline 0.62 --mde 0.02

  # Relative MDE: "can we detect a +5% relative lift?"
  python analysis/sample_size.py --baseline 0.62 --mde 0.05 --mde-type relative

  # Stricter alpha, higher power, unequal allocation
  python analysis/sample_size.py --baseline 0.62 --mde 0.02 --alpha 0.01 \
      --power 0.9 --ratio 2.0

  # Reverse use: "we only have 8,000 users/arm and 30 days runtime -- what is
  # the smallest lift we could reliably detect?"
  python analysis/sample_size.py --baseline 0.62 --n-per-group 8000 --solve-mde
"""

import argparse
import math

from scipy import stats


def n_per_group(p1: float, p2: float, alpha: float, power: float, ratio: float = 1.0) -> float:
    """Required control-group n for a two-proportion z-test (Wald approximation)."""
    z_alpha2 = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    pooled_var = p1 * (1 - p1) + (p2 * (1 - p2)) / ratio
    return ((z_alpha2 + z_beta) ** 2) * pooled_var / (p1 - p2) ** 2


def minimum_detectable_effect(p1: float, n1: float, alpha: float, power: float, ratio: float = 1.0) -> float:
    """Inverse problem: smallest |p2 - p1| detectable with a fixed n1 (control size)."""
    # Solve numerically: at p2 = p1 + d, does n_per_group(p1, p2) <= n1?
    lo, hi = 1e-5, 1 - p1 - 1e-5
    for _ in range(100):
        mid = (lo + hi) / 2
        p2 = p1 + mid
        required = n_per_group(p1, p2, alpha, power, ratio)
        if required > n1:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=float, required=True, help="control conversion rate, e.g. 0.62")
    parser.add_argument("--mde", type=float, default=None, help="minimum detectable effect")
    parser.add_argument("--mde-type", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--alpha", type=float, default=0.05, help="two-sided significance level")
    parser.add_argument("--power", type=float, default=0.80, help="target statistical power (1 - beta)")
    parser.add_argument("--ratio", type=float, default=1.0, help="n_treatment / n_control allocation ratio")
    parser.add_argument("--daily-signups", type=int, default=None, help="optional: estimate runtime in days")
    parser.add_argument("--n-per-group", type=float, default=None, help="solve for MDE given a fixed sample size")
    parser.add_argument("--solve-mde", action="store_true", help="solve for MDE instead of sample size")
    args = parser.parse_args()

    p1 = args.baseline

    if args.solve_mde or (args.n_per_group and not args.mde):
        if not args.n_per_group:
            parser.error("--n-per-group is required with --solve-mde")
        mde = minimum_detectable_effect(p1, args.n_per_group, args.alpha, args.power, args.ratio)
        p2 = p1 + mde
        print(f"Baseline conversion rate : {p1:.2%}")
        print(f"Fixed sample size        : {args.n_per_group:,.0f} per group (control)")
        print(f"alpha = {args.alpha} (two-sided)  ·  power = {args.power:.0%}\n")
        print(f"Minimum detectable absolute lift : {mde:+.2%}  (-> {p2:.2%})")
        print(f"Minimum detectable relative lift : {mde / p1:+.2%}")
        return

    if args.mde is None:
        parser.error("--mde is required unless --solve-mde is used")

    mde_abs = args.mde if args.mde_type == "absolute" else args.mde * p1
    p2 = p1 + mde_abs
    if not (0 < p2 < 1):
        parser.error(f"baseline + mde = {p2:.4f} is out of (0, 1) range")

    n_control = n_per_group(p1, p2, args.alpha, args.power, args.ratio)
    n_treatment = n_control * args.ratio
    n_control, n_treatment = math.ceil(n_control), math.ceil(n_treatment)
    total = n_control + n_treatment

    print(f"Baseline conversion rate (control) : {p1:.2%}")
    print(f"Target conversion rate (treatment) : {p2:.2%}")
    print(f"Minimum detectable effect          : {mde_abs:+.2%} absolute  ({mde_abs / p1:+.2%} relative)")
    print(f"alpha = {args.alpha} (two-sided)  ·  power = {args.power:.0%}  ·  ratio (T/C) = {args.ratio}\n")
    print(f"  required n (control)   : {n_control:,}")
    print(f"  required n (treatment) : {n_treatment:,}")
    print(f"  required n (total)     : {total:,}")

    if args.daily_signups:
        days = math.ceil(total / args.daily_signups)
        print(f"\n  at {args.daily_signups:,} new signups/day -> ~{days} days to reach this sample size")
        print("  (add 1-2 full weeks on top to cover weekday/weekend seasonality)")


if __name__ == "__main__":
    main()
