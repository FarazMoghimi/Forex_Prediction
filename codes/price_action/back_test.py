import pandas as pd
import os
from codes.price_action.trendline import Trendline
import plotly as py
import plotly.tools as pyTools
import plotly.graph_objs as go
import numpy as np
from codes.technical_indicator.indicator_function_map import run_indicator_function
pyTools.set_credentials_file(username='mbehroozikhah', api_key='evr0jZbxNMUeP5Hysifc')

class BackTest:
    def __init__(self, LOT=1e5):
        self.loaded_data = None
        self.trendline = Trendline(tolerance=200, Point=0.00001, Intersection_ab=3, Intersection_bc=3, fibo=15,
                                   InpDepth=5, InpDeviation=5, InpBackstep=3, Score_limit=2)
        self.SPREAD = self.trendline.Point * 10
        self.LOT = LOT
        self.NEAR_UPTREND = 1  # point
        self.NEAR_DOWNTREND = 1  # point
        self.RISK_FACTOR = 0.02
        self.MIN_POINT_FOR_POSITION = self.trendline.Point * 40
        self.stoploss_buy = 0
        self.breakeven_buy = 0
        self.tp2_buy = 0
        self.sum_stoploss_buy = 0
        self.sum_tp2_buy = 0
        self.stoploss_sell = 0
        self.breakeven_sell = 0
        self.tp2_sell = 0
        self.sum_stoploss_sell = 0
        self.sum_tp2_sell = 0
        self.balance = 1000
        self.buy_orders = []
        self.sell_orders = []

    def load_data(self, filename):
        self.loaded_data = pd.read_csv(os.path.join('../../data/', filename))

    def run_test(self, from_data, to_data, window_size=100, after_window=0, before_window=10):
        for start in range(from_data, to_data-window_size):
            pos = start + window_size # we are in pos and our data are valid until pos-1
            df = self.loaded_data[start:pos].reset_index()
            self.check_buy_orders(self.loaded_data.iloc[pos])
            self.check_sell_orders(self.loaded_data.iloc[pos])
            # if start == 1050:
            self.trendline.onCalculate(0, pos-1, window_size, df.Open.values[::-1], df.High.values[::-1], df.Low.values[::-1], df.Close.values[::-1])
            # print(len(self.trendline.UpTrend))
            # print(len(self.trendline.DownTrend))
            buy_signal = self.buy_signal(df, self.loaded_data.iloc[pos], pos)
            # sell_signal = False
            sell_signal = self.sell_signal(df, self.loaded_data.iloc[pos], pos)
            if (buy_signal or sell_signal) and from_data < start - before_window:
                df = self.loaded_data[(start - before_window):min(pos + after_window,
                                                                                to_data)].reset_index()
                self.update_plot(pos-1, df, before_window, after_window, buy_signal, sell_signal)
        print({
            'sell_stoploss': self.stoploss_sell,
            'sell_breakeven': self.breakeven_sell,
            'sell_tp2': self.tp2_sell,
            'sell_stoploss_diff_avg': self.sum_stoploss_sell / self.stoploss_sell,
            'sell_tp2_diff_avg': self.sum_tp2_sell
        })
        print({
            'buy_stoploss': self.stoploss_buy,
            'buy_breakeven': self.breakeven_buy,
            'buy_tp2': self.tp2_buy,
            'buy_stoploss_diff_avg': self.sum_stoploss_buy / self.stoploss_buy,
            'buy_tp2_diff_avg': self.sum_tp2_buy
        })
        print(self.balance)

    def buy_signal(self, history, current_price, pos):
        if self.is_inside_bar(history.iloc[-2], history.iloc[-1]):
            if self.is_insideBar_near_upTrend(history.iloc[-2], pos-2, history.iloc[-1], pos-1, history, pos):
                if current_price['High'] > history.iloc[-1]['High']:
                    if not self.trendline.is_trendline_down():
                        buy_price = history.iloc[-1]['High']
                        stoploss = self.trendline.get_last_low() - self.trendline.Point
                        position = {
                            'buy_price': buy_price + self.SPREAD,
                            'fix_stoploss': stoploss,
                            'stoploss': stoploss,
                            'tp1': buy_price + (buy_price - stoploss),
                            'tp2': buy_price + 2 * (buy_price - stoploss)
                        }
                        if buy_price - stoploss > self.SPREAD and buy_price - stoploss > self.MIN_POINT_FOR_POSITION and self.is_tp2_below_high_zigzag(position['tp2']):
                            self.buy_order(position)
                            return True
        return False

    def sell_signal(self, history, current_price, pos):
        if self.is_inside_bar(history.iloc[-2], history.iloc[-1]):
            if self.is_insideBar_near_downTrend(history.iloc[-2], pos-2, history.iloc[-1], pos-1, history, pos):
                if current_price['Low'] < history.iloc[-1]['Low']:
                    if not self.trendline.is_trendline_up():
                        sell_price = history.iloc[-1]['Low']
                        stoploss = self.trendline.get_last_high() + self.trendline.Point
                        position = {
                            'sell_price': sell_price,
                            'fix_stoploss': stoploss + self.SPREAD,
                            'stoploss': stoploss + self.SPREAD,
                            'tp1': sell_price - (stoploss + self.SPREAD - sell_price) + self.SPREAD,
                            'tp2': sell_price - 2 * (stoploss + self.SPREAD - sell_price) + self.SPREAD
                        }
                        if stoploss - sell_price > self.SPREAD and stoploss - sell_price > self.MIN_POINT_FOR_POSITION and self.is_tp2_above_low_zigzag(position['tp2']):
                            self.sell_order(position)
                            return True

        return False

    def buy_order(self, position):
        self.buy_orders.append(position)

    def sell_order(self, position):
        self.sell_orders.append(position)

    def check_buy_orders(self, current_price):
        should_close = []
        for i, order in enumerate(self.buy_orders):
            if order['stoploss'] >= current_price['Low']:
                should_close.append(i)
                if order['stoploss'] == order['buy_price']:
                    self.breakeven_buy += 1
                else:
                    self.stoploss_buy += 1
                    self.sum_stoploss_buy += order['buy_price'] - order['fix_stoploss']
                    lotage = (self.balance * self.RISK_FACTOR) / ((order['buy_price'] - order['fix_stoploss']) * self.LOT)
                    self.balance -= lotage * (order['buy_price'] - order['stoploss']) * self.LOT
                    print('stoploss buy', self.balance)
            elif order['tp2'] <= current_price['High']:
                lotage = (self.balance * self.RISK_FACTOR) / ((order['buy_price'] - order['fix_stoploss']) * self.LOT)
                self.balance += lotage * (order['tp2'] - order['buy_price']) * self.LOT
                should_close.append(i)
                self.tp2_buy += 1
                self.sum_tp2_buy += order['tp2'] - order['buy_price']
                print('tp2 buy', self.balance)
            elif order['tp1'] <= current_price['High']:
                order['stoploss'] = order['buy_price']
        self.buy_orders = [v for i, v in enumerate(self.buy_orders) if i not in should_close]

    def check_sell_orders(self, current_price):
        should_close = []
        for i, order in enumerate(self.sell_orders):
            if order['stoploss'] <= current_price['High']:
                should_close.append(i)
                if order['stoploss'] == order['sell_price']:
                    self.breakeven_sell += 1
                else:
                    self.stoploss_sell += 1
                    self.sum_stoploss_sell += order['fix_stoploss'] - order['sell_price']
                    lotage = (self.balance * self.RISK_FACTOR) / ((order['fix_stoploss'] - order['sell_price']) * self.LOT)
                    self.balance -= lotage * (order['fix_stoploss'] - order['sell_price']) * self.LOT
                    print('stoploss sell', self.balance)
            elif order['tp2'] >= current_price['Low']:
                lotage = (self.balance * self.RISK_FACTOR) / ((order['fix_stoploss'] - order['sell_price']) * self.LOT)
                self.balance += lotage * (order['sell_price'] - order['tp2']) * self.LOT
                print('tp2 sell', self.balance)
                should_close.append(i)
                self.tp2_sell += 1
                self.sum_tp2_sell += order['sell_price'] - order['tp2']
            elif order['tp1'] >= current_price['Low']:
                order['stoploss'] = order['sell_price']
        self.sell_orders = [v for i, v in enumerate(self.sell_orders) if i not in should_close]

    @staticmethod
    def is_inside_bar(last_2, last_1):
        if last_2['Low'] < last_1['Low'] < last_2['High'] and \
            last_2['Low'] < last_1['High'] < last_2['High']:
            return True
        return False

    def is_insideBar_near_upTrend(self, price_pos1, pos1, price_pos2, pos2, history, pos):
        upTrend = np.array(self.trendline.UpTrend)[::-1]
        for line in upTrend:
            line_price = self.trendline.get_price_line_on_pos(line, pos1)
            if 0 <= price_pos1['High'] - line_price and (price_pos1['Low'] <= line_price <= price_pos1['High'] or price_pos1['Low'] - line_price < self.NEAR_UPTREND * self.trendline.Point):
                if self.trendline.get_price_line_on_pos(line, pos2) <= price_pos2['Close']:
                    if self.compare_ma_slope_with_trendLine_slope(pos, history, line):
                        return True

    def is_insideBar_near_downTrend(self, price_pos1, pos1, price_pos2, pos2, history, pos):
        downTrend = np.array(self.trendline.DownTrend)[::-1]
        for line in downTrend:
            line_price = self.trendline.get_price_line_on_pos(line, pos)
            if 0 <= line_price - price_pos1['Low'] and (price_pos1['Low'] <= line_price <= price_pos1['High'] or line_price - price_pos1['High'] < self.NEAR_DOWNTREND * self.trendline.Point):
                if self.trendline.get_price_line_on_pos(line, pos2) >= price_pos2['Close']:
                    if self.compare_ma_slope_with_trendLine_slope(pos, history, line):
                        return True

    def is_tp2_below_high_zigzag(self, tp2):
        return tp2 <= self.trendline.get_last_high()

    def is_tp2_above_low_zigzag(self, tp2):
        return tp2 >= self.trendline.get_last_low()

    def compare_ma_slope_with_trendLine_slope(self, pos, history, line):
        # start = (pos - line['prev_i'] - 1) + line['ax'] + 1
        # stop = 0
        # mid = (start + stop) // 2
        # past_ma = self.get_ma_slope(history.Close.values[-start:-mid])
        # last_ma = self.get_ma_slope(history.Close.values[-mid:])
        # print(past_ma, last_ma, line['coef'])
        return True


    def get_ma_slope(self, close):
        ma1 = run_indicator_function('ma', {
            'data': close,
            'period': len(close)
        })
        ma2 = run_indicator_function('ma', {
            'data': close[:-1],
            'period': len(close[:-1])
        })
        return ma1[-1] - ma2[-1]

    def update_plot(self, start, df, before_window, after_window, buy_signal, sell_signal):
        trace = go.Candlestick(x=df.index,
                               open=df.Open,
                               high=df.High,
                               low=df.Low,
                               close=df.Close)
        data = [trace]
        layout = {
            'title': 'The Great Recession',
            'yaxis': {'title': 'AAPL Stock'},
            'shapes': [],
        }
        upTrend = np.array(self.trendline.UpTrend)[::-1]
        downTrend = np.array(self.trendline.DownTrend)[::-1]
        zizag = np.array(self.trendline.zizag.ExtZigzagBuffer)[::-1]
        if sell_signal:
            for i in range(downTrend.size):
                # if start == downTrend[i]['prev_i']:
                    x0_pos = df.Close.size - downTrend[i]['ax'] - (start - downTrend[i]['prev_i']) - after_window - 1
                    x1_pos = df.Close.size - 1
                    # x1_pos = df.Close.size - 1
                    layout['shapes'].append({
                        'type': 'line',
                        'x0': df.index[x0_pos],
                        'x1': df.index[x1_pos],
                        'y0': downTrend[i]['ay'],
                        # 'y1': downTrend[i]['by'],
                        'y1': downTrend[i]['beta'] + downTrend[i]['coef'] * ((downTrend[i]['prev_i'] - start) - after_window),
                        'xref': 'x', 'yref': 'y',
                        'line': {'color': 'rgb(255,0,0)', 'width': 1}
                    })
        # print(downTrend)
        # print(upTrend)
        # print(start)
        # print(df)
        if buy_signal:
            for i in range(upTrend.size):
                # if start == upTrend[i]['prev_i']:
                    x0_pos = df.Close.size - upTrend[i]['ax'] - (start - upTrend[i]['prev_i']) - after_window - 1
                    x1_pos = df.Close.size - 1
                    # x1_pos = df.Close.size - 1
                    layout['shapes'].append({
                        'type': 'line',
                        'x0': df.index[x0_pos],
                        'x1': df.index[x1_pos],
                        'y0': upTrend[i]['ay'],
                        # 'y1': upTrend[i]['by'],
                        'y1': upTrend[i]['beta'] + upTrend[i]['coef'] * ((upTrend[i]['prev_i'] - start) - after_window),
                        'xref': 'x', 'yref': 'y',
                        'line': {'color': 'rgb(0,255,0)', 'width': 1}
                    })

        for i in range(zizag.size):
            if zizag[i] != 0:
                layout['shapes'].append({
                    'type': 'circle',
                    'x0': df.index[before_window + i] -1,
                    'x1': df.index[before_window + i] +1,
                    'y0': zizag[i]-.000001,
                    'y1': zizag[i]+.000001,
                    # 'y1': upTrend[i]['by'],
                    'xref': 'x', 'yref': 'y',
                    'line': {'color': 'rgb(0,0,255)', 'width': 1}
                })
        # print(layout)
        fig = dict(data=data, layout=layout)
        py.offline.plot(fig, filename='plot/' + str(df.Open[0]) + '.html')


if __name__ == '__main__':
    backtest = BackTest()
    backtest.load_data('GBPUSD5.csv')
    backtest.run_test(25000, 36700, 100, 50, 100)