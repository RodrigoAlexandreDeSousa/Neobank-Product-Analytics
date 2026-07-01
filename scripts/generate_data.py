"""
Generate a synthetic neobank product-event dataset with an embedded A/B test.

---Story:

A neobank ran an experiment on its onboarding flow. Users are split 50/50:
  - control   : the existing onboarding flow
  - treatment : a new, "lighter" onboarding flow

We bake in a real effect: the treatment flow lifts the `signup_completed`
conversion step. Because the downstream steps are conditional on completing
signup, that lift propagates through the whole funnel (realistic behaviour).

The A/B variant affects ONLY signup_completed. Downstream steps (kyc_submitted,
first_deposit, first_transaction) have their own small, realistic dependence on
country / platform / time-of-day -- e.g. KYC pass rates vary by country
(document-verification friction), deposit rates vary by platform, and
late-night signups follow through to a first transaction slightly less often.
This is what gives ml/train_activation_model.py real, recoverable signal to
find: a model predicting post-signup activation should learn that `variant`
carries no information (correct -- it doesn't, by design), while country,
platform and signup hour do.

---Funnel (in order):

  1. signup_started      (everyone)
  2. signup_completed
  3. kyc_submitted
  4. first_deposit
  5. first_transaction

---Outputs:

  data/raw_users.csv   : one row per user
  data/raw_events.csv  : one row per (user, event) -- the raw event log

---Run:

  python scripts/generate_data.py --n-users 50000 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Conversion probabilities.
# Control vs treatment differ ONLY at the signup_completed step (the experiment).
# Everything downstream is conditional on the previous step being reached.
# ----------------------------------------------------------------------------
P_SIGNUP_COMPLETED_CONTROL = 0.62
P_SIGNUP_COMPLETED_TREATMENT = 0.68    # the baked-in lift (~+6pp) -- the ONLY variant effect
P_KYC_SUBMITTED = 0.75                 # conditional on signup_completed
P_FIRST_DEPOSIT = 0.55                 # conditional on kyc_submitted
P_FIRST_TRANSACTION = 0.70             # conditional on first_deposit

COUNTRIES = ["PT", "ES", "PL", "GB", "AE"]
COUNTRY_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]
PLATFORMS = ["ios", "android", "web"]
PLATFORM_WEIGHTS = [0.45, 0.40, 0.15]

# ----------------------------------------------------------------------------
# Realistic, non-variant signal on the DOWNSTREAM steps only (never on
# signup_completed -- that stays a clean, single-cause A/B effect).
# Weighted across COUNTRY_WEIGHTS / PLATFORM_WEIGHTS these roughly net out, so
# the headline funnel rates above are still close to the true marginal rates.
# ----------------------------------------------------------------------------
KYC_COUNTRY_MODIFIER = {"PT": 0.05, "ES": 0.03, "GB": 0.02, "PL": 0.00, "AE": -0.10}
DEPOSIT_PLATFORM_MODIFIER = {"ios": 0.05, "android": 0.00, "web": -0.08}
NIGHT_HOURS = set(range(0, 6))           # 00:00-05:59 local
TXN_NIGHT_SIGNUP_MODIFIER = -0.06        # night-time signups follow through less
TXN_DAY_SIGNUP_MODIFIER = 0.01

FUNNEL = [
    "signup_started",
    "signup_completed",
    "kyc_submitted",
    "first_deposit",
    "first_transaction",
]


def generate(n_users: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    user_id = np.arange(1, n_users + 1)
    variant = rng.choice(["control", "treatment"], size=n_users, p=[0.5, 0.5])
    country = rng.choice(COUNTRIES, size=n_users, p=COUNTRY_WEIGHTS)
    platform = rng.choice(PLATFORMS, size=n_users, p=PLATFORM_WEIGHTS)

    # Signup happens at a random moment over a 60-day window.
    start = np.datetime64("2026-01-01T00:00:00")
    offset_minutes = rng.integers(0, 60 * 24 * 60, size=n_users)  # 60 days in minutes
    signup_ts = start + offset_minutes.astype("timedelta64[m]")

    # ---- walk the funnel with conditional random draws ----
    p_complete = np.where(variant == "treatment",
                          P_SIGNUP_COMPLETED_TREATMENT,
                          P_SIGNUP_COMPLETED_CONTROL)

    # Per-user downstream probabilities: country affects KYC pass rate, platform
    # affects deposit rate, time-of-day-of-signup affects follow-through to a
    # first transaction. None of these depend on `variant`.
    signup_hour = pd.Series(signup_ts).dt.hour.to_numpy()
    is_night = np.isin(signup_hour, list(NIGHT_HOURS))

    p_kyc = np.clip(
        P_KYC_SUBMITTED + np.vectorize(KYC_COUNTRY_MODIFIER.get)(country), 0.01, 0.99
    )
    p_deposit = np.clip(
        P_FIRST_DEPOSIT + np.vectorize(DEPOSIT_PLATFORM_MODIFIER.get)(platform), 0.01, 0.99
    )
    p_txn = np.clip(
        P_FIRST_TRANSACTION + np.where(is_night, TXN_NIGHT_SIGNUP_MODIFIER, TXN_DAY_SIGNUP_MODIFIER),
        0.01, 0.99,
    )

    reached_started = np.ones(n_users, dtype=bool)
    reached_completed = reached_started & (rng.random(n_users) < p_complete)
    reached_kyc = reached_completed & (rng.random(n_users) < p_kyc)
    reached_deposit = reached_kyc & (rng.random(n_users) < p_deposit)
    reached_txn = reached_deposit & (rng.random(n_users) < p_txn)

    reached = {
        "signup_started": reached_started,
        "signup_completed": reached_completed,
        "kyc_submitted": reached_kyc,
        "first_deposit": reached_deposit,
        "first_transaction": reached_txn,
    }

    users = pd.DataFrame(
        {
            "user_id": user_id,
            "variant": variant,
            "country": country,
            "platform": platform,
            "signup_ts": signup_ts,
        }
    )

    # ---- build the long event log ----
    # Each step happens a random number of minutes after the *previous* step
    # (sequential gaps, each strictly positive), not a fixed offset from
    # signup -- so time-to-activate is a real distribution, not a constant.
    # This is what gives the "time to activate" panel in the dashboard/EDA an
    # actual shape to show, instead of a single spike.
    step_gap_range_minutes = {
        "signup_completed": (3, 20),     # filling in the signup form
        "kyc_submitted": (10, 90),       # finding + uploading ID documents
        "first_deposit": (30, 360),      # linking a card / bank transfer
        "first_transaction": (15, 600),  # deciding to actually spend/transfer
    }

    event_ts = {"signup_started": signup_ts}
    prev_step = "signup_started"
    for step in FUNNEL[1:]:
        lo, hi = step_gap_range_minutes[step]
        gap_minutes = rng.integers(lo, hi + 1, size=n_users)
        event_ts[step] = event_ts[prev_step] + gap_minutes.astype("timedelta64[m]")
        prev_step = step

    frames = []
    for step in FUNNEL:
        mask = reached[step]
        frames.append(
            pd.DataFrame(
                {
                    "user_id": user_id[mask],
                    "event_name": step,
                    "event_ts": event_ts[step][mask],
                }
            )
        )

    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values(["user_id", "event_ts"]).reset_index(drop=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1))

    return users, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    users, events = generate(args.n_users, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    users.to_csv(out / "raw_users.csv", index=False)
    events.to_csv(out / "raw_events.csv", index=False)

    # quick sanity print of the baked-in effect
    print(f"users:  {len(users):,}  ->  {out / 'raw_users.csv'}")
    print(f"events: {len(events):,}  ->  {out / 'raw_events.csv'}")
    conv = (
        events[events.event_name == "signup_completed"]
        .merge(users[["user_id", "variant"]], on="user_id")
        .groupby("variant")["user_id"].nunique()
        / users.groupby("variant")["user_id"].nunique()
    )
    print("\nsignup_completed conversion by variant (the experiment):")
    print(conv.round(4).to_string())

    # quick sanity print of the downstream, non-variant signal
    activated = events[events.event_name == "first_transaction"]["user_id"].unique()
    users["activated"] = users["user_id"].isin(activated)
    print("\nfirst_transaction rate by country (downstream signal, not variant-driven):")
    print(users.groupby("country")["activated"].mean().round(4).sort_values(ascending=False).to_string())
    print("\nfirst_transaction rate by platform (downstream signal, not variant-driven):")
    print(users.groupby("platform")["activated"].mean().round(4).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
