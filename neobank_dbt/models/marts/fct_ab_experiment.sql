-- Experiment table: for each (variant, funnel_step), how many users converted
-- out of the total. This is exactly what analysis/ab_test.py reads to run the
-- two-proportion z-test.

with funnel as (
    select * from {{ ref('fct_funnel') }}
),

unpivoted as (
    select variant, 'signup_started'    as funnel_step, 1 as step_order, reached_signup_started    as reached from funnel
    union all
    select variant, 'signup_completed',                     2,             reached_signup_completed           from funnel
    union all
    select variant, 'kyc_submitted',                        3,             reached_kyc_submitted              from funnel
    union all
    select variant, 'first_deposit',                        4,             reached_first_deposit             from funnel
    union all
    select variant, 'first_transaction',                    5,             reached_first_transaction         from funnel
)

select
    variant,
    funnel_step,
    step_order,
    sum(reached)                         as converted,
    count(*)                             as total,
    safe_divide(sum(reached), count(*))  as conversion_rate
from unpivoted
group by variant, funnel_step, step_order
order by step_order, variant
