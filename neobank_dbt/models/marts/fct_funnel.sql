-- User-grain funnel table.
-- For every user, flag which funnel steps they reached and when.
-- Conditional aggregation (MAX(CASE WHEN ...)) pivots the long event log
-- into one wide row per user.

with users as (
    select * from {{ ref('stg_users') }}
),

events as (
    select * from {{ ref('stg_events') }}
),

flags as (
    select
        user_id,

        max(case when event_name = 'signup_started'    then 1 else 0 end) as reached_signup_started,
        max(case when event_name = 'signup_completed'  then 1 else 0 end) as reached_signup_completed,
        max(case when event_name = 'kyc_submitted'     then 1 else 0 end) as reached_kyc_submitted,
        max(case when event_name = 'first_deposit'     then 1 else 0 end) as reached_first_deposit,
        max(case when event_name = 'first_transaction' then 1 else 0 end) as reached_first_transaction,

        min(case when event_name = 'signup_started'    then event_ts end) as ts_signup_started,
        min(case when event_name = 'signup_completed'  then event_ts end) as ts_signup_completed,
        min(case when event_name = 'first_transaction' then event_ts end) as ts_first_transaction
    from events
    group by user_id
)

select
    u.user_id,
    u.variant,
    u.country,
    u.platform,
    u.signup_date,

    f.reached_signup_started,
    f.reached_signup_completed,
    f.reached_kyc_submitted,
    f.reached_first_deposit,
    f.reached_first_transaction,

    -- minutes from signup_started to first_transaction (time-to-activate)
    timestamp_diff(f.ts_first_transaction, f.ts_signup_started, minute) as minutes_to_activate
from users u
left join flags f using (user_id)
