import codes.technical_indicator.io as io
import numpy as np

computed_df = io.read_computed_indicators()
manual_df = io.read_precomputed_indicators()

for column_name in computed_df.columns.tolist():
    if column_name in ['xau', 'Date'] or column_name not in manual_df.columns.tolist():
        continue
    result = np.allclose(computed_df.loc[30:, column_name], manual_df.loc[30:, column_name], atol=.05, equal_nan=True)
    print(column_name, result)
        # print(computed_df[indexes,column_name], manual_df[indexes,column_name])
    # row_ids = computed_df[
    #     computed_df[column_name].values + .5 >= manual_df[column_name].values and
    #                       computed_df[column_name].values + .5 <= manual_df[column_name].values
    # ].index
    # print(row_ids)