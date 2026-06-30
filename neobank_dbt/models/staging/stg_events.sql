-- Clean + typecast the raw events log.
with source as (
    select * from {{ source('neobank_raw', 'raw_events') }}
)

select
    cast(event_id as int64)       as event_id,
    cast(user_id as int64)        as user_id,
    lower(event_name)             as event_name,
    cast(event_ts as timestamp)   as event_ts
from source
