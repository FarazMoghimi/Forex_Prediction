import pandas as pd
import os
import json

with open('../../config/price_action.json') as f:
    config = json.load(f)


def read_indicators():
    data = pd.read_csv(os.path.join('../../result/technical_indicator/', config['in_file']))
    # if 'Time' in data.keys():
    #     data = data[(data['Time'] < '22:00:00') & (data['Time'] > '10:00:00')]
    # print(data['Time'].size)
    return data


def to_excel(df):
    df.to_csv(os.path.join('../../result/prediction/', config['out_file']))
