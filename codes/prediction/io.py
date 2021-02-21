import pandas as pd
import os
import json

with open('../../config/prediction.json') as f:
    config = json.load(f)


def read_indicators():
    data = pd.read_csv(os.path.join('../../result/technical_indicator/', config['in_file']), parse_dates=['Date'])
    # if 'Time' in data.keys():
    #     data = data[(data['Time'] < '22:00:00') & (data['Time'] > '10:00:00')]
    # print(data['Time'].size)
    data['Time'] = '00:00:00'
    # data['Date'] = data['Date']
    return data


def get_train():
    return config['train_set']


def get_test():
    return config['test_set']


def get_y_period():
    return config['y_period']


def get_close_column_name():
    return config['close_column_name']


def get_x_column_names():
    return config['x_columns']


def get_score_columns():
    return config['score_columns']


def get_step_svm():
    return config['step_svm']


def get_risk_factor():
    return config['risk_factor']


def get_probability_cutoff():
    return config['probability_cutoff']


def to_excel(df):
    df.to_excel(os.path.join('../../result/prediction/', config['out_file']), sheet_name='prediction')
