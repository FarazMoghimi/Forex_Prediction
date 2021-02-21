import codes.price_action.io as io
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

pd_data = io.read_indicators()
candle_column = 8
pd_result = pd.DataFrame()
pd_result['pre1'] = pd_data['candle_type']
pd_result['pre2'] = pd_data['candle_type'].shift(-1)
pd_result['pre3'] = pd_data['candle_type'].shift(-2)
# pd_result['post'] = pd_data['candle_type'].shift(-2)
pd_result = pd_result[pd_result['pre1'] == 33.0]
# for i in range(len(pd_data.index)-1):
#     pd_result.append({
#         'pre': pd_data.iloc[i, candle_column],
#         'post': pd_data.iloc[i+1, candle_column],
#     }, ignore_index=True)
pd_result = pd_result.pivot_table(index=['pre1', 'pre2'], columns='pre3', aggfunc=len, margins=True)
# pd_result = pd_result.div(pd_result.iloc[-1,:], axis=1)
pd_result = pd_result.div(pd_result.iloc[:,-1], axis=0)
sns.set(style="white")
ax = sns.heatmap(pd_result, cmap="RdYlBu", annot=True)
plt.show()