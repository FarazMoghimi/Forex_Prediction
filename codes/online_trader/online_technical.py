import pandas as pd
import numpy as np

import codes.technical_indicator.io as io
from codes.technical_indicator.indicator_function_map import run_indicator_function

class Online_Technical:
    def __init__(self):
        self.df = io.read_raw()
        self.json_indicators = io.read_indicator_json()
        for json_indicator in self.json_indicators:
            params = json_indicator['params_value'].copy()
            for (column_arg, column_name) in json_indicator['params_column'].items():
                params[column_arg] = np.array(self.df[column_name])
            result = run_indicator_function(json_indicator['func_name'], params)
            self.df[json_indicator['column_name']] = pd.Series(result)



io.to_excel(df)
print(df)
# print(json_indicators)
