import pandas as pd
import os

input_filename = 'USDJPY_M1.csv'
time_frame = '1440Min'


pd_data = pd.read_csv(os.path.join('../../data/', input_filename),
                      header=None,
                      usecols=[0, 1, 2, 3, 4, 5, 6],
                      names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Vol'],
                      parse_dates={'DateTime': [0, 1]})

pd_data['WeekOfDay'] = pd_data['DateTime'].apply(lambda x: x.weekday())
pd_data['Hour'] = pd_data['DateTime'].apply(lambda x: x.hour)
pd_data = pd_data.set_index('DateTime')
conversion = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'WeekOfDay': 'first', 'Hour': 'first', 'Vol': 'sum'}
# data = pd_data.ix[:, ['Open', 'High', 'Low', 'Close']]
# pd_data.DateTime = pd.to_datetime(pd_data.DateTime, unit='s')
# print(data)
resample_data = pd_data.resample(time_frame, how=conversion)
resample_data = resample_data.dropna()

resample_data.to_csv("USDJPY_D1.csv")