-- One row per user: the dimensional table joined onto facts.
select
    user_id,
    variant,
    country,
    platform,
    signup_ts,
    signup_date
from {{ ref('stg_users') }}
