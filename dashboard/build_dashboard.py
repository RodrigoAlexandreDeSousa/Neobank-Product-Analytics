"""
Build a single self-contained HTML dashboard straight from the local CSVs --
no BigQuery project, no Looker Studio account, no internet connection needed
to view it once built. Useful as a fallback/companion to the Looker Studio
path described in the README, and handy to have open during an interview.

Panels
------
  1. Cumulative funnel conversion by step, control vs treatment
  2. A/B lift per funnel step, with 95% confidence intervals
  3. Activation rate heatmap (country x platform)
  4. Time-to-activate distribution (signup_started -> first_transaction),
     control vs treatment

Run
---
  python dashboard/build_dashboard.py
  # writes dashboard/neobank_dashboard.html -- open it in a browser
"""

import argparse
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FUNNEL = ["signup_started", "signup_completed", "kyc_submitted", "first_deposit", "first_transaction"]


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    users = pd.read_csv(data_dir / "raw_users.csv", parse_dates=["signup_ts"])
    events = pd.read_csv(data_dir / "raw_events.csv", parse_dates=["event_ts"])
    return users, events


def funnel_table(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for step in FUNNEL:
        reached = set(events.loc[events.event_name == step, "user_id"])
        is_reached = users["user_id"].isin(reached)
        for variant, sub in users.groupby("variant"):
            converted = int(is_reached[sub.index].sum())
            total = len(sub)
            rows.append({"step": step, "variant": variant, "converted": converted, "total": total})
    df = pd.DataFrame(rows)
    df["rate"] = df["converted"] / df["total"]
    first_total = df.groupby("variant")["total"].transform("first")
    df["cumulative_rate"] = df["converted"] / first_total
    return df


def lift_table(funnel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for step in FUNNEL:
        sub = funnel[funnel.step == step]
        c = sub[sub.variant == "control"].iloc[0]
        t = sub[sub.variant == "treatment"].iloc[0]
        p_c, p_t = c.converted / c.total, t.converted / t.total
        se = math.sqrt(p_c * (1 - p_c) / c.total + p_t * (1 - p_t) / t.total)
        diff = p_t - p_c
        rows.append({"step": step, "lift_pp": diff * 100, "ci_half_pp": 1.96 * se * 100})
    return pd.DataFrame(rows)


def activation_heatmap_table(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    activated_users = set(events.loc[events.event_name == "first_transaction", "user_id"])
    users = users.copy()
    users["activated"] = users["user_id"].isin(activated_users)
    return users.pivot_table(index="country", columns="platform", values="activated", aggfunc="mean") * 100


def time_to_activate(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    start = events.loc[events.event_name == "signup_started", ["user_id", "event_ts"]].rename(
        columns={"event_ts": "t0"}
    )
    txn = events.loc[events.event_name == "first_transaction", ["user_id", "event_ts"]].rename(
        columns={"event_ts": "t1"}
    )
    df = start.merge(txn, on="user_id").merge(users[["user_id", "variant"]], on="user_id")
    df["minutes_to_activate"] = (df["t1"] - df["t0"]).dt.total_seconds() / 60
    return df


def build_figure(users: pd.DataFrame, events: pd.DataFrame) -> go.Figure:
    funnel = funnel_table(users, events)
    lift = lift_table(funnel)
    heat = activation_heatmap_table(users, events)
    tta = time_to_activate(users, events)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Cumulative funnel conversion (control vs treatment)",
            "A/B lift per funnel step (pp, 95% CI)",
            "Activation rate by country x platform (%)",
            "Time to activate -- signup to first transaction (minutes)",
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "heatmap"}, {"type": "xy"}]],
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
    )

    colors = {"control": "#7f8c8d", "treatment": "#2E86C1"}
    for variant in ["control", "treatment"]:
        sub = funnel[funnel.variant == variant]
        fig.add_trace(
            go.Bar(
                x=sub["step"],
                y=sub["cumulative_rate"] * 100,
                name=variant,
                marker_color=colors[variant],
                legendgroup=variant,
                opacity=0.85,
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=lift["step"],
            y=lift["lift_pp"],
            error_y=dict(type="data", array=lift["ci_half_pp"], visible=True),
            marker_color="#2E86C1",
            name="lift (pp)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)

    fig.add_trace(
        go.Heatmap(
            z=heat.values,
            x=list(heat.columns),
            y=list(heat.index),
            colorscale="Blues",
            text=heat.round(1).values,
            texttemplate="%{text}",
            colorbar=dict(title="%", len=0.4, y=0.18),
        ),
        row=2,
        col=1,
    )

    for variant in ["control", "treatment"]:
        sub = tta[tta.variant == variant]
        fig.add_trace(
            go.Histogram(
                x=sub["minutes_to_activate"],
                name=variant,
                marker_color=colors[variant],
                opacity=0.6,
                legendgroup=variant,
                showlegend=False,
                nbinsx=40,
            ),
            row=2,
            col=2,
        )
    fig.update_yaxes(title_text="% of users", row=1, col=1)
    fig.update_yaxes(title_text="lift (pp)", row=1, col=2)
    fig.update_xaxes(title_text="minutes", row=2, col=2)
    fig.update_yaxes(title_text="users", row=2, col=2)

    n = len(users)
    fig.update_layout(
        title=dict(
            text=(
                "Neobank onboarding funnel & A/B experiment<br>"
                f"<sup>n = {n:,} users · synthetic data · "
                "generated by scripts/generate_data.py</sup>"
            ),
            x=0.02,
        ),
        barmode="overlay",
        height=850,
        width=1150,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="dashboard/neobank_dashboard.html")
    args = parser.parse_args()

    users, events = load_data(args.data_dir)
    fig = build_figure(users, events)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Dashboard written to {out_path}  ({len(users):,} users, {len(events):,} events)")


if __name__ == "__main__":
    main()
