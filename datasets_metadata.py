# 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'TrafficL', 'Weather'
from datasetsforecast.long_horizon2 import ETTh1, ETTh2, ETTm1, ETTm2, ECL, TrafficL, Weather


ts_metadata = {
    "ETTh1": {
        "target_ts": "OT",
        "exog_list": ['HUFL', 'HULL', 'MUFL', 'MULL', "LUFL", "LULL"],
        "freq": "h",
        "test_size": 2000,
        "valid_size": 1000,
    },
    "ETTh2": {
        "target_ts": "OT",
        "exog_list": ['HUFL', 'HULL', 'MUFL', 'MULL', "LUFL", "LULL"],
        "freq": "h",
        "test_size": 2000,
        "valid_size": 1000,
    },
    "ETTm1": {
        "target_ts": "OT",
        "exog_list": ['HUFL', 'HULL', 'MUFL', 'MULL', "LUFL", "LULL"],
        "freq": "min",
        "test_size": 2000,
        "valid_size": 1000,
    },
    "ETTm2": {
        "target_ts": "OT",
        "exog_list": ['HUFL', 'HULL', 'MUFL', 'MULL', "LUFL", "LULL"],
        "freq": "min",
        "test_size": 2000,
        "valid_size": 1000,
    }
}