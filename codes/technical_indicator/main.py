import pandas as pd
import numpy as np
import talib
from talib import abstract

import codes.technical_indicator.io as io
from codes.technical_indicator.indicator_function_map import run_indicator_function

df = io.read_raw()
json_indicators = io.read_indicator_json()

# print(run_indicator_function('rsi', {'period': 14, 'data': df.loc[0:18, 'Close']}))

for json_indicator in json_indicators:
    params = json_indicator['params_value'].copy()
    for (column_arg, column_name) in json_indicator['params_column'].items():
        params[column_arg] = np.array(df[column_name])
    result = run_indicator_function(json_indicator['func_name'], params)
    df[json_indicator['column_name']] = pd.Series(result)



io.to_excel(df)
print(df)
# print(json_indicators)
