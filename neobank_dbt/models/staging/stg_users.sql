-- Clean + typecast the raw users table.
with source as (
    select * from {{ source('neobank_raw', 'raw_users') }}
)

select
    cast(user_id as int64)              as user_id,
    lower(variant)                      as variant,
    upper(country)                      as country,
    lower(platform)                     as platform,
    cast(signup_ts as timestamp)        as signup_ts,
    date(cast(signup_ts as timestamp))  as signup_date
from source
