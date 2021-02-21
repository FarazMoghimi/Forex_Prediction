from sklearn import svm
import pandas as pd
import codes.prediction.io as io
import numpy as np
from codes.technical_indicator.indicator_function_map import run_indicator_function
from codes.prediction.score import get_all_scores_name, get_all_scores, add_column_score
import json
import re
import datetime

###### our params
c_set = [0.001]
gamma = 0.01
kernel_set = ['rbf']


class SVM:
    def __init__(self):
        self.pd_data = io.read_indicators()
        self.x = self.pd_data.loc[:, io.get_x_column_names()]
        self.train_limit = io.get_train()
        self.test_limit = io.get_test()

        params = dict()
        self.y_period = io.get_y_period()
        params['period'] = io.get_y_period()
        params["data"] = np.array(self.pd_data[io.get_close_column_name()])
        self.y = np.array(run_indicator_function('y_period', params))

    def prepare_data(self):
        res = dict()
        x_train = np.array(self.x[self.train_limit['start']:self.train_limit['end']].values)
        res['x_train'] = x_train.astype(float)
        x_test = np.array(self.x[self.test_limit['start']:self.test_limit['end']].values)
        res['x_test'] = x_test.astype(int)

        y_train = self.y[self.train_limit['start']:self.train_limit['end']]
        res['y_train'] = y_train.astype(int)
        y_test = self.y[self.test_limit['start']:self.test_limit['end']]
        res['y_test'] = y_test.astype(int)

        res['Time'] = np.array(self.pd_data[self.test_limit['start']:self.test_limit['end']]['Time'])
        res['Time'] = res['Time']

        res['Date'] = np.array(self.pd_data[self.test_limit['start']:self.test_limit['end']]['Date'])
        res['Date'] = res['Date']

        res['BBAND(21)'] = np.array(self.pd_data[self.test_limit['start']:self.test_limit['end']]['BBAND(21)'])
        res['BBAND(21)'] = res['BBAND(21)']

        res['ADX(12)'] = np.array(self.pd_data[self.test_limit['start']:self.test_limit['end']]['ADX(12)'])
        res['ADX(12)'] = res['ADX(12)']
        return res

    def run_svm(self, x_train, y_train, x_test, c, kernel):
        clf = svm.SVC(kernel=kernel, degree=3, C=c, gamma=gamma)
        clf.fit(x_train, y_train)
        return clf.predict(x_test)

    def run_svm_prob(self, x_train, y_train, x_test, c, kernel):
        clf = svm.SVC(kernel=kernel, degree=3, C=c, probability=True, gamma=gamma)
        clf.fit(x_train, y_train)
        return clf.predict_proba(x_test)

    def train_test(self):
        pd_result = pd.DataFrame(columns=[
            'Y', 'Y_rbf', 'prob_rbf'
        ])
        data = self.prepare_data()
        for c in c_set:
            for kernel in kernel_set:
                pd_result['Date'] = data['Date']
                pd_result['Time'] = data['Time']
                pd_result['BBAND(21)'] = data['BBAND(21)']
                pd_result['ADX(12)'] = data['ADX(12)']
                pd_result['Y'] = data['y_test']
                y_pred = self.run_svm(data['x_train'], data['y_train'], data['x_test'], c, kernel)
                pd_result['Y_' + kernel] = y_pred
                y_prob = np.array(self.run_svm_prob(data['x_train'], data['y_train'], data['x_test'], c, kernel))
                pd_result['prob_' + kernel] = y_prob[np.arange(len(y_prob)), y_prob.argmax(axis=1).squeeze()]
        return pd_result

    def step_train_test(self):
        c = 3
        pd_result = pd.DataFrame(columns=[
            'c', 'kernel', 'test_start', 'test_end', 'train_start', 'train_end'
        ].extend(get_all_scores_name()))

        for i in range(io.get_step_svm()):
            print(i)
            self.train_limit['start'] += 1
            self.train_limit['end'] += 1
            self.test_limit['start'] = self.train_limit['end'] + self.y_period
            self.test_limit['end'] = self.train_limit['end'] + self.y_period + 1
            data = self.prepare_data()
            res = {}
            for kernel in kernel_set:
                y_pred = self.run_svm(data['x_train'], data['y_train'], data['x_test'], c, kernel)
                res['Y_' + kernel] = y_pred[0]
                y_prob = np.array(self.run_svm_prob(data['x_train'], data['y_train'], data['x_test'], c, kernel)[0])
                res['prob_' + kernel] = max(y_prob)
                res['Date'] = data['Date'][0]
                res['Time'] = data['Time'][0]
                res['BBAND(21)'] = data['BBAND(21)'][0]
                res['ADX(12)'] = data['ADX(12)'][0]
            pd_result = pd_result.append({
                'train_start': self.train_limit['start'],
                'train_end': self.train_limit['end'],
                'Y': data['y_test'][0],
                'c': c,
                **res,
            }, ignore_index=True)
        return pd_result


    def get_gain(self, true_y, predict_y, time, total, risk_factor, i):
        # time = time.strftime('%H:%M:%S')
        if predict_y == true_y:
            # if time > '23:00:00' or time < '10:00:00':
            #     return total * risk_factor * .5
            return total * risk_factor * .75
        return total * risk_factor * -1

    def json_array(self, x):
        x = re.findall(r'\d+\.*\d*', x)
        x = [float(t) for t in x]
        return x

    def should_get_position(self, pd_true, pd_predict, pd_prob, pb_bband, pb_adx, pd_time, i, last_positions):
         # cutoff=self.json_array(pd_prob[i])
        cutoff = pd_prob[i]

        # if len(last_positions) > 0 and last_positions[-1][0] + 6 > i:
        #     return False
        # if pb_adx[i] < 0:
        #     return False
        # bband = self.json_array(pb_bband[i])
        # if bband[1] - bband[0] <= 0.15:
        #     return False
        # if cutoff + .02 > io.get_probability_cutoff() and len(last_positions) > 2 and last_positions[-1][1] > 0 and last_positions[-2][1] > 0 and last_positions[-2][0] + 20 > i:
        #     return True
        # if len(last_positions) > 2 and last_positions[-1][1] < 0 and last_positions[-2][1] < 0 and last_positions[-2][0] + 20 > i:
        #     return False
        if cutoff < io.get_probability_cutoff():
            return False

        time = pd_time[i]
        # time = time.strftime('%H:%M:%S')
        # if time > '23:00:00' or time < '10:00:00':
        #     return False
        # if i > 7 and cutoff - 0.04 < io.get_probability_cutoff() and (pd_true[i-6] != pd_predict[i-6] or pd_true[i-7] != pd_predict[i-7]):
        #     return False
        return True

    def add_loss_gain(self, pd_data):
        risk_factor = io.get_risk_factor()
        for kernel in kernel_set:
            pd_series = pd.Series()
            pd_position = pd.Series()
            total = 100
            last_positions = []
            for i in range(pd_data['Y'].size):
                position = self.should_get_position(pd_data['Y'],
                                                    pd_data['Y_' + kernel],
                                                    pd_data['prob_' + kernel],
                                                    pd_data['BBAND(21)'],
                                                    pd_data['ADX(12)'],
                                                    pd_data['Time'],
                                                    i,
                                                    last_positions)
                if position:
                    gain = self.get_gain(
                        pd_data.at[i, 'Y'],
                        pd_data.at[i, 'Y_' + kernel],
                        pd_data.at[i, 'Time'],
                        total,
                        risk_factor,
                        i
                    )
                    total += gain
                    last_positions.append((i, gain))
                pd_series.set_value(i, total)
                pd_position.set_value(i, position)
            pd_data['total_' + kernel] = pd_series
            pd_data['position_' + kernel] = pd_position
        return pd_data

    def get_step_score(self, pd_data):
        for kernel in kernel_set:
            filter_data = pd_data[pd_data['position_'+kernel]]
            score = add_column_score(filter_data['Y'], filter_data['Y_' + kernel], kernel)
            total_position = filter_data['position_'+kernel].size
            # params = dict()
            # params['period'] = io.get_y_period()
            # params["data"] = np.array(self.pd_data[io.get_close_column_name()])
            # y = np.array(run_indicator_function('shift', params))
            # mean_true_diff = np.mean(abs(y[np.where(pd_data['position_'+kernel].values)[0]] - self.pd_data.loc[pd_data['position_'+kernel].values, :]['Close']))
            # mean_false_diff = np.mean(abs(y[np.where(np.logical_not(pd_data['position_' + kernel].values))[0]] - self.pd_data.loc[np.logical_not(pd_data['position_'+kernel].values), :]['Close']))
            return {**score, 'total_position': total_position}
