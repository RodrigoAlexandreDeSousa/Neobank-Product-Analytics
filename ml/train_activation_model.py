"""
Predict, right after `signup_completed`, which users will go on to reach
`first_transaction` ("activate") -- so Growth/CRM can target at-risk users
with a nudge (KYC reminder, deposit incentive) before they drop off.

This is a genuinely useful, separate question from the A/B test: the
experiment (analysis/ab_test.py) tells you the new onboarding flow lifts
signup completion. This model tells you, among people who already signed up,
who is unlikely to fully activate on their own -- a targeting problem, not a
causal one.

Important -- read before quoting this in an interview
-------------------------------------------------------
`variant` is included as a feature because it is genuinely known at
prediction time. But by construction of this dataset (see
scripts/generate_data.py) it carries NO information about *post-signup*
activation -- the A/B effect lives entirely in whether a user reaches
signup_completed in the first place. Expect this model to assign `variant`
~zero importance. That's the model correctly recovering a known fact about
the data-generating process, not a bug -- and a good thing to say out loud if
asked "isn't variant leaking the experiment into your model?".

Population   : users who reached signup_completed (the model is only useful
               once there's something to predict).
Target       : reached_first_transaction (1) vs not (0).
Features     : variant, country, platform, signup_hour, signup_dayofweek,
               minutes_signup_to_complete (an engagement/friction signal).
Leakage guard: kyc_submitted / first_deposit are deliberately excluded -- they
               sit *between* the prediction point and the target on the
               funnel, so including them would leak the outcome.

Two data sources (mirrors analysis/ab_test.py)
-----------------------------------------------
  --source local      data/raw_users.csv + data/raw_events.csv
  --source bigquery    dim_users + fct_funnel marts (requires the dbt pipeline)

Run
---
  python ml/train_activation_model.py --source local
"""

import argparse

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["variant", "country", "platform", "signup_dayofweek"]
NUMERIC = ["signup_hour", "minutes_signup_to_complete"]
TARGET = "reached_first_transaction"


def features_from_local() -> pd.DataFrame:
    users = pd.read_csv("data/raw_users.csv", parse_dates=["signup_ts"])
    events = pd.read_csv("data/raw_events.csv", parse_dates=["event_ts"])

    # Population: users who reached signup_completed (inner join).
    completed = events.loc[events.event_name == "signup_completed", ["user_id", "event_ts"]]
    completed = completed.rename(columns={"event_ts": "completed_ts"})
    df = users.merge(completed, on="user_id", how="inner")

    txn_users = set(events.loc[events.event_name == "first_transaction", "user_id"])
    df[TARGET] = df["user_id"].isin(txn_users).astype(int)

    df["signup_hour"] = df["signup_ts"].dt.hour
    df["signup_dayofweek"] = df["signup_ts"].dt.dayofweek.astype(str)
    df["minutes_signup_to_complete"] = (df["completed_ts"] - df["signup_ts"]).dt.total_seconds() / 60

    return df[["user_id"] + CATEGORICAL + NUMERIC + [TARGET]]


def features_from_bigquery(project: str, dataset: str) -> pd.DataFrame:
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    query = f"""
        select
            f.user_id, f.variant, f.country, f.platform,
            extract(hour from u.signup_ts)                       as signup_hour,
            cast(extract(dayofweek from u.signup_ts) as string)  as signup_dayofweek,
            f.reached_first_transaction
        from `{project}.{dataset}.fct_funnel` f
        join `{project}.{dataset}.dim_users` u using (user_id)
        where f.reached_signup_completed = 1
    """
    # minutes_signup_to_complete isn't exposed by the marts yet -- extend
    # fct_funnel with ts_signup_completed if you want it server-side; for now
    # we drop it from the BigQuery path rather than silently fabricate it.
    return client.query(query).to_dataframe()


def build_pipeline(model, categorical: list, numeric: list) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(name: str, pipe: Pipeline, X_test, y_test) -> tuple[float, float]:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    print(f"\n=== {name} ===")
    print(f"  ROC-AUC      : {auc:.3f}  (0.5 = no better than random)")
    print(f"  PR-AUC       : {ap:.3f}  (baseline = positive rate = {y_test.mean():.3f})")
    print(f"  Brier score  : {brier:.4f}  (lower = better-calibrated probabilities)")
    print(classification_report(y_test, pred, target_names=["did_not_activate", "activated"]))
    return auc, ap


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["local", "bigquery"], default="local")
    parser.add_argument("--project", default=None)
    parser.add_argument("--dataset", default="neobank_analytics")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.source == "local":
        df = features_from_local()
    else:
        if not args.project:
            parser.error("--project is required when --source bigquery")
        df = features_from_bigquery(args.project, args.dataset)

    categorical = [c for c in CATEGORICAL if c in df.columns]
    numeric = [c for c in NUMERIC if c in df.columns]

    print(f"Population (reached signup_completed) : {len(df):,}")
    print(f"Activation rate (target prevalence)    : {df[TARGET].mean():.2%}")
    print(f"Features used                          : {categorical + numeric}")

    X = df[categorical + numeric]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # --- baseline: logistic regression (interpretable coefficients) ---
    logreg = build_pipeline(LogisticRegression(max_iter=1000), categorical, numeric)
    logreg.fit(X_train, y_train)
    evaluate("Logistic Regression (baseline)", logreg, X_test, y_test)

    ohe = logreg.named_steps["pre"].named_transformers_["cat"]
    feature_names = list(ohe.get_feature_names_out(categorical)) + numeric
    coefs = logreg.named_steps["model"].coef_[0]
    print("\n  Coefficients ranked by |effect| (log-odds; + = higher activation odds):")
    for name, coef in sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1])):
        print(f"    {name:35s} {coef:+.3f}")

    # --- stronger model: gradient boosting (sklearn-native, no extra dependency) ---
    gboost = build_pipeline(HistGradientBoostingClassifier(random_state=args.seed), categorical, numeric)
    gboost.fit(X_train, y_train)
    evaluate("HistGradientBoosting", gboost, X_test, y_test)

    print(
        "\nNote: by design (see scripts/generate_data.py) `variant` carries no true\n"
        "signal for post-signup activation -- only country, platform and signup_hour\n"
        "do. A near-zero `variant` coefficient/importance above is the model\n"
        "correctly recovering that fact, not an error."
    )


if __name__ == "__main__":
    main()
