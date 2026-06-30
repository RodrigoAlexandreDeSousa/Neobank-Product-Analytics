-- ============================================================================
-- Ad-hoc analytical queries (BigQuery Standard SQL).
-- Run these against the dbt marts once they are built. They show the kind of
-- SQL a product analyst writes day to day: funnel drop-off, cohort retention,
-- and segment cuts. Replace `your_project.neobank_analytics` with your dataset.
-- ============================================================================


-- 1) FUNNEL DROP-OFF with a window function ----------------------------------
-- Step-over-step conversion: what % of the *previous* step survives to this one.
with steps as (
    select
        funnel_step,
        step_order,
        sum(converted) as users_at_step
    from `your_project.neobank_analytics.fct_ab_experiment`
    group by funnel_step, step_order
)
select
    funnel_step,
    users_at_step,
    lag(users_at_step) over (order by step_order)                          as users_prev_step,
    safe_divide(
        users_at_step,
        lag(users_at_step) over (order by step_order)
    )                                                                      as step_conversion,
    safe_divide(
        users_at_step,
        first_value(users_at_step) over (order by step_order)
    )                                                                      as cumulative_conversion
from steps
order by step_order;


-- 2) A/B LIFT per funnel step ------------------------------------------------
-- Side-by-side conversion of control vs treatment at every step.
select
    funnel_step,
    step_order,
    max(if(variant = 'control',   conversion_rate, null)) as control_rate,
    max(if(variant = 'treatment', conversion_rate, null)) as treatment_rate,
    max(if(variant = 'treatment', conversion_rate, null))
        - max(if(variant = 'control', conversion_rate, null))             as absolute_lift
from `your_project.neobank_analytics.fct_ab_experiment`
group by funnel_step, step_order
order by step_order;


-- 3) WEEKLY SIGNUP COHORTS -> activation rate --------------------------------
-- Group users by signup week and measure how many reached first_transaction.
select
    date_trunc(signup_date, week)                                          as signup_week,
    variant,
    count(*)                                                               as signups,
    sum(reached_first_transaction)                                         as activated,
    safe_divide(sum(reached_first_transaction), count(*))                  as activation_rate
from `your_project.neobank_analytics.fct_funnel`
group by signup_week, variant
order by signup_week, variant;


-- 4) ACTIVATION by acquisition segment ---------------------------------------
-- Where do the best-activating users come from? (country x platform)
select
    country,
    platform,
    count(*)                                                               as users,
    round(avg(reached_first_transaction) * 100, 1)                         as activation_pct,
    round(avg(minutes_to_activate))                                        as avg_minutes_to_activate
from `your_project.neobank_analytics.fct_funnel`
group by country, platform
having users >= 500
order by activation_pct desc;
