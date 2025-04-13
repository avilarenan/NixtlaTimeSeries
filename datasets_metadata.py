# 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'TrafficL', 'Weather'
from datasetsforecast.long_horizon2 import ETTh1, ETTh2, ETTm1, ETTm2, ECL, TrafficL, Weather


ts_metadata = {
    "ETTh1": {
        "target_ts": "OT",
        "exog_list": ['HUFL', 'HULL', 'MUFL', 'MULL', "LUFL", "LULL"],
        "freq": "h"
    },
    "ETTh2": {
        "target_ts": "OT",
        "exog_list": ['HUFL', 'HULL', 'MUFL', 'MULL', "LUFL", "LULL"],
        "freq": "h"
    },
    "ETTm1": {
        "target_ts": "OT",
        "exog_list": ['HUFL', 'HULL', 'MUFL', 'MULL', "LUFL", "LULL"],
        "freq": "min"
    },
    "ETTm2": {
        "target_ts": "OT",
        "exog_list": ['HUFL', 'HULL', 'MUFL', 'MULL', "LUFL", "LULL"],
        "freq": "min"
    }
}