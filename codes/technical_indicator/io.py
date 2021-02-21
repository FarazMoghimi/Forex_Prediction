import pandas as pd
import os
import json

with open('../../config/raw_input.json') as f:
    config = json.load(f)


def read_raw():
    parse_dates = [config['date_column_name']]
    if config['has_Time']:
        parse_dates.append('Time')

    if config['in_file'][-3:] == 'csv':
        data = pd.read_csv(os.path.join('../../data/', config['in_file']),
                             # usecols=6,
                             parse_dates=parse_dates,
                             dtype={
                                 config['close_column_name']: 'float',
                                 config['open_column_name']: 'float',
                                 config['high_column_name']: 'float',
                                 config['low_column_name']: 'float',
                                 config['volume_column_name']: 'float',
                             })
        print(data.dtypes)
    else:
        data = pd.read_excel(os.path.join('../../data/', config['in_file']),
                             usecols=6,
                             parse_dates=parse_dates,
                             dtype={
                                 config['close_column_name']: 'float',
                                 config['open_column_name']: 'float',
                                 config['high_column_name']: 'float',
                                 config['low_column_name']: 'float',
                                 config['volume_column_name']: 'float',
                             })

    print(data.dtypes)
    return data


def read_precomputed_indicators():
    data = pd.read_excel(os.path.join('../../data/', config['in_file']),
                         parse_dates=[config['date_column_name']])
    return data


def read_computed_indicators():
    data = pd.read_excel(os.path.join('../../result/technical_indicator/', config['out_file']),
                         parse_dates=[config['date_column_name']])
    return data


def to_excel(df):
    df.to_csv(os.path.join('../../result/technical_indicator/', config['out_file']))


def read_indicator_json():
    with open('../../config/indicator.json') as file:
        return json.load(file)
