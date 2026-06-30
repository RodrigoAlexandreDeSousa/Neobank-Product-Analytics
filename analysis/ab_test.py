"""
Evaluate the onboarding A/B test: does the treatment flow lift signup completion?

Computes, for the chosen funnel step:
  - conversion rate per variant
  - absolute & relative lift
  - two-proportion z-test (p-value)
  - 95% confidence interval for the difference
  - achieved statistical power
and prints a plain-English ship / no-ship recommendation.

Two data sources
----------------
  --source local      reads data/raw_users.csv + data/raw_events.csv directly
                      (works immediately, before BigQuery/dbt are set up)
  --source bigquery   reads the fct_ab_experiment mart built by dbt
                      (the "production" path; requires the BigQuery pipeline)

Run
---
  python analysis/ab_test.py --source local --step signup_completed
  python analysis/ab_test.py --source bigquery --project my-gcp-project \
         --dataset neobank_analytics --step signup_completed

Stats note: scipy is used so the script runs with minimal dependencies.
statsmodels (proportions_ztest, NormalIndPower) is a cleaner production choice.
"""

import argparse
import math

from scipy import stats


def counts_from_local(step: str) -> dict:
    import pandas as pd

    users = pd.read_csv("data/raw_users.csv")
    events = pd.read_csv("data/raw_events.csv")

    reached = events[events.event_name == step]["user_id"].unique()
    users["reached"] = users["user_id"].isin(reached)

    grouped = users.groupby("variant")["reached"].agg(["sum", "count"])
    return {
        v: {"converted": int(row["sum"]), "total": int(row["count"])}
        for v, row in grouped.iterrows()
    }


def counts_from_bigquery(project: str, dataset: str, step: str) -> dict:
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    query = f"""
        SELECT variant, converted, total
        FROM `{project}.{dataset}.fct_ab_experiment`
        WHERE funnel_step = @step
    """
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("step", "STRING", step)]
        ),
    )
    return {
        row["variant"]: {"converted": int(row["converted"]), "total": int(row["total"])}
        for row in job
    }


def two_proportion_z_test(c_conv, c_tot, t_conv, t_tot):
    """Manual two-proportion z-test (control vs treatment)."""
    p_c = c_conv / c_tot
    p_t = t_conv / t_tot
    p_pool = (c_conv + t_conv) / (c_tot + t_tot)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / c_tot + 1 / t_tot))
    z = (p_t - p_c) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # 95% CI for the difference (unpooled SE)
    se_diff = math.sqrt(p_c * (1 - p_c) / c_tot + p_t * (1 - p_t) / t_tot)
    diff = p_t - p_c
    ci_low = diff - 1.96 * se_diff
    ci_high = diff + 1.96 * se_diff

    # achieved power for the observed effect (alpha = 0.05, two-sided)
    effect = abs(diff)
    z_alpha = stats.norm.ppf(1 - 0.05 / 2)
    z_beta = effect / se_diff - z_alpha
    power = stats.norm.cdf(z_beta)

    return p_c, p_t, diff, z, p_value, ci_low, ci_high, power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["local", "bigquery"], default="local")
    parser.add_argument("--step", default="signup_completed")
    parser.add_argument("--project", default=None)
    parser.add_argument("--dataset", default="neobank_analytics")
    args = parser.parse_args()

    if args.source == "local":
        counts = counts_from_local(args.step)
    else:
        if not args.project:
            parser.error("--project is required when --source bigquery")
        counts = counts_from_bigquery(args.project, args.dataset, args.step)

    c, t = counts["control"], counts["treatment"]
    p_c, p_t, diff, z, p_value, ci_low, ci_high, power = two_proportion_z_test(
        c["converted"], c["total"], t["converted"], t["total"]
    )

    print(f"Funnel step under test : {args.step}\n")
    print(f"  control   : {p_c:7.2%}  ({c['converted']:,}/{c['total']:,})")
    print(f"  treatment : {p_t:7.2%}  ({t['converted']:,}/{t['total']:,})\n")
    print(f"  absolute lift : {diff:+.2%}")
    print(f"  relative lift : {diff / p_c:+.2%}")
    print(f"  95% CI (diff) : [{ci_low:+.2%}, {ci_high:+.2%}]")
    print(f"  z-statistic   : {z:.3f}")
    print(f"  p-value       : {p_value:.5f}")
    print(f"  power         : {power:.2%}\n")

    significant = p_value < 0.05 and ci_low > 0
    if significant:
        print("RECOMMENDATION: SHIP the new onboarding flow.")
        print("The lift is positive and statistically significant (p < 0.05, CI > 0).")
    else:
        print("RECOMMENDATION: DO NOT SHIP yet.")
        print("The effect is not statistically convincing; keep iterating / collect more data.")


if __name__ == "__main__":
    main()
