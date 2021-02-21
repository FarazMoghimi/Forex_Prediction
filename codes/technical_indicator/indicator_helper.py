import numpy as np
import talib


def adx(high, low, close, timeperiod=12):
    return talib.ADX(high, low, close, timeperiod) - 30


def sar(high, low, acceleration, maximum):
    return talib.SAR(high, low, acceleration, maximum)


def stoch(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3):
    slowk, slowd = talib.STOCH(high, low, close, fastk_period, slowk_period, 0, slowd_period, 0)
    return slowk


def bbands(close, timeperiod=21):
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod, nbdevup=2, nbdevdn=2, matype=0)
    return list(zip(lowerband, upperband))


def bbands_lower(close, timeperiod=21):
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod, nbdevup=2, nbdevdn=2, matype=0)
    return lowerband


def bbands_upper(close, timeperiod=21):
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod, nbdevup=2, nbdevdn=2, matype=0)
    return upperband


def bband_ma(close, bband):
    res = []
    for i in range(len(close)):
        if bband[i][0] is None or np.isnan(bband[i][0]):
            res.append(None)
            continue
        if close[i] < bband[i][1]:
            res.append(+1)
        elif close[i] > bband[i][0]:
            res.append(-1)
        else:
            res.append(0)
    return res


def sar_ma(close, sar):
    res = []
    for i in range(len(close)):
        if sar[i] is None or np.isnan(sar[i]):
            res.append(None)
            continue
        if close[i] > sar[i]:
            res.append(+1)
        elif close[i] < sar[i]:
            res.append(-1)
        else:
            res.append(0)
    return res


def ma_ma(close, short_ma, long_ma):
    res = []
    for i in range(len(close)):
        if close[i] > short_ma[i] and close[i] > long_ma[i] and short_ma[i] > long_ma[i]:
            res.append(+1)
        elif close[i] < short_ma[i] and close[i] < long_ma[i] and short_ma[i] < long_ma[i]:
            res.append(-1)
        else:
            res.append(0)
    return res


def mfi_ma(close, mfi, mfi_bound=20):
    res = []
    for i in range(len(close)):
        if mfi[i] is None or np.isnan(mfi[i]):
            res.append(None)
            continue
        if mfi[i] < mfi_bound:
            res.append(+1)
        elif mfi[i] > mfi_bound:
            res.append(-1)
        else:
            res.append(0)
    return res


def stoch_ma(close, stoch, stoch_bound=20):
    res = []
    for i in range(len(close)):
        if stoch[i] is None or np.isnan(stoch[i]):
            res.append(None)
            continue
        if stoch[i] < stoch_bound:
            res.append(+1)
        elif stoch[i] > stoch_bound:
            res.append(-1)
        else:
            res.append(0)
    return res


def rsi_ma(close, rsi, rsi_bound=30):
    res = []
    for i in range(len(close)):
        if rsi[i] is None or np.isnan(rsi[i]):
            res.append(None)
            continue
        if rsi[i] < rsi_bound:
            res.append(+1)
        elif rsi[i] > rsi_bound:
            res.append(-1)
        else:
            res.append(0)
    return res


def y_period(data, period):
    res = []
    for i in range(len(data)):
        if i >= len(data) - period:
            res.append(None)
            continue
        if data[i] < data[i+period]:
            res.append(+1)
        else:
            res.append(-1)
    return res


def rsi(data, period):
    return talib.RSI(data, timeperiod=period)


def macd(data, short_period, long_period):
    _macd, _, _ = talib.MACD(data, fastperiod=short_period, slowperiod=long_period)
    return _macd


def ma(data, period):
    return talib.MA(data, timeperiod=period)


def sma(data, period):
    return talib.SMA(data, timeperiod=period)


def mfi(close, high, low, volume, period):
    return talib.MFI(high, low, close, volume, timeperiod=period)


def slope(data, period=1):
    res = []
    for i in range(len(data)):
        res.append(_period_slope(data, period, i))
    return res


def momentum(avg_data, data):
    res = []
    for i in range(len(data)):
        if i < 1 or np.isnan(avg_data[i]):
            res.append(None)
            continue
        diff = data[i] - data[i-1]
        if abs(diff) > avg_data[i]:
            if diff > 0:
                res.append(1)
            else:
                res.append(-1)
        else:
            res.append(0)
    return res


def diff_func(plus_data, minus_data):
    res = []
    for i in range(len(plus_data)):
        res.append(plus_data[i]-minus_data[i])
    return res


def divergence(data1, data2, period):
    res = []
    for i in range(len(data1)):
        data1_slope = _period_slope(data1, period, i)
        data2_slope = _period_slope(data2, period, i)
        if data1_slope == 1 and data2_slope == -1:
            res.append(1)
        elif data2_slope == 1 and data1_slope == -1:
            res.append(-1)
        elif data1_slope is None or data1_slope is None:
            res.append(None)
        else:
            res.append(0)
    return res


def _period_slope(data, period, i):
    if i < period:
        return None
    count_plus = 0
    count_minus = 0
    for j in range(period):
        if data[i-j] > data[i-j-1]:
            count_plus += 1
        elif data[i-j] < data[i-j-1]:
            count_minus += 1
    if count_plus > 0 and count_minus > 0:
        return 0
    elif count_plus > 0:
        return +1
    else:
        return -1


def border(data, upper_bound, lower_bound):
    res = []
    for rsi in data:
        if np.isnan(rsi):
            res.append(None)
        elif rsi > upper_bound:
            res.append(-1)
        elif rsi < lower_bound:
            res.append(1)
        else:
            res.append(0)
    return res


def hcl(close, high, low):
    res = []
    for i in range(len(close)):
        if close[i] > high[i - 1]:
            res.append(1)
        elif close[i] < low[i - 1]:
            res.append(-1)
        else:
            res.append(0)
    return res


def range_function(close, high, low):
    res = []
    for i in range(len(close)):
        diff = (high[i] - low[i]) / 3
        if close[i] <= low[i] + diff:
            res.append(-1)
        elif close[i] < low[i] + 2 * diff:
            res.append(0)
        else:
            res.append(1)
    return res


def compare(high_plus, high_minus):
    res = []
    for i in range(len(high_plus)):
        if high_plus[i] > high_minus[i]:
            res.append(1)
        elif high_plus[i] == high_minus[i]:
            res.append(0)
        else:
            res.append(-1)
    return res


def shift(data, period):
    res = []
    for i in range(len(data)):
        if i < period:
            res.append(None)
        else:
            res.append(data[i-period])
    return res


def swing(max_data, min_data):
    res = []
    for i in range(len(max_data)):
        if max_data[i-4] == 1:
            res.append(-1)
        elif min_data[i-4] == 1:
            res.append(1)
        else:
            res.append(0)
    return res


def step(data, bound):
    res = []
    for i in range(len(data)):
        if data[i] > bound:
            res.append(1)
        elif data[i] < bound:
            res.append(-1)
        else:
            res.append(0)

    return res


def distance_max_local(data):
    res = []
    for i in range(len(data)):
        last_index = index_last_maxima(data, i)
        res.append(i - last_index)
    return logistic(res, .25, 11)


def index_last_maxima(data, i):
    for j in range(3, i):
        if i-j > 0 and data[i-j] > data[i-j+1] and data[i-j] > data[i-j+2] and data[i-j] > data[i-j-1] and data[i-j] > data[i-j-2]:
            return i-j
    return 0


def index_maxima(data, how_far=1):
    res = []
    for i in range(len(data)):
        last_index = i
        for j in range(how_far):
            last_index = index_last_maxima(data, last_index)
        res.append(last_index)
    return res


def index_minima(data, how_far=1):
    res = []
    for i in range(len(data)):
        last_index = i
        for j in range(how_far):
            last_index = index_last_minima(data, last_index)
        res.append(last_index)
    return res


def distance_min_local(data):
    res = []
    for i in range(len(data)):
        last_index = index_last_minima(data, i)
        res.append(i - last_index)
    return logistic(res, .25, 11)


def index_last_minima(data, i):
    for j in range(3, i):
        if i-j > 0 and data[i-j] < data[i-j+1] and data[i-j] < data[i-j+2] and data[i-j] < data[i-j-1] and data[i-j] < data[i-j-2]:
            return i-j
    return 0


def logistic(data, k, x0):
    data = np.array(data)
    for i in range(len(data)):
        if data[i] is None or np.isnan(data[i]):
            data[i] = 0
    return 1 / (1 + np.exp(-k*(data-x0)))


def doji(open, high, low, close):
    res = talib.CDLDOJI(open, high, low, close)
    return np.array(res) == 100


def engulfing(open, high, low, close):
    res = talib.CDLENGULFING(open, high, low, close)
    return np.array(res) == 100


def abandoned_baby(open, high, low, close):
    res = talib.CDLABANDONEDBABY(open, high, low, close, penetration=0)
    return np.array(res) == 100


def gravestone_doji(open, high, low, close):
    res = talib.CDLGRAVESTONEDOJI(open, high, low, close)
    return np.array(res) == 100


def hammer(open, high, low, close):
    res = talib.CDLHAMMER(open, high, low, close)
    return np.array(res) == 100


def hanging_man(open, high, low, close):
    res = talib.CDLHANGINGMAN(open, high, low, close)
    return np.array(res) == 100


def inverted_hammer(open, high, low, close):
    res = talib.CDLINVERTEDHAMMER(open, high, low, close)
    return np.array(res) == 100


def candle_type(open, high, low, close):
    res = []
    for i in range(len(close)):
        scale = 3
        percent = (high[i] - low[i]) / scale
        digit_1 = ((open[i] - low[i]) // percent) + 1
        if digit_1 == scale+1:
            digit_1 = digit_1-1
        digit_2 = ((close[i] - low[i]) // percent) + 1
        if digit_2 == scale+1:
            digit_2 = digit_2-1
        sign = +1
        if open[i] > close[i]:
            sign = -1
        res.append((digit_1 * 10 + digit_2) * sign)
    return res
# def macd_rule(data, close_price):
#     res = []
#     for i in range(len(data)):
#         if data[i] > data[i - 4] and close_price[i] < close_price[i - 4]:
#             res.append(1)
#         elif data[i] < data[i - 4] and close_price[i] > close_price[i - 4]:
#             res.append(-1)
#         else:
#             res.append(0)
#     return res
#
#
# def ma_rule(data, close_price):
#     res = []
#     for i in range(len(data)):
#         if i >= len(close_price):
#             res.append(0)
#         else:
#             if data[i] < close_price[i]:
#                 res.append(1)
#             else:
#                 res.append(-1)
#     return res
#
#
# def k_percent(high, low, close):
#     res = []
#     for i in range(len(close)):
#         if (i < 13):
#             res.append(0)
#         else:
#             res.append((close[i] - min(low[i - 13:i + 1])) / (max(high[i - 13:i + 1]) - min(low[i - 13:i + 1])) * 100)
#     return res
#
#
# def d_percent(data):
#     res = []
#     for i in range(len(data)):
#         res.append(sum(data[i - 2:i + 1]) / 3)
#     return res
#
#
# def k_percent_d_percent(d_percent, k_percent):
#     res = []
#     for i in range(len(d_percent)):
#         if k_percent[i] > d_percent[i]:
#             res.append(1)
#         elif k_percent[i] < d_percent[i]:
#             res.append(-1)
#         else:
#             res.append(1)
#     return res
#
#
# def fix_missing(data):
#     res = []
#     for i in range(len(data)):
#         if data[i] == 0:
#             res[i] = res[i - 1]
#         else:
#             res.append(data[i])
#     return res
#
#
#
#
# def max_local(close, neighbor=1):
#     res = [0 for _ in range(neighbor)]
#     for i in range(neighbor, len(close)-neighbor):
#         ans = 1
#         for j in range(1, neighbor + 1):
#             if close[i] < close[i - j] or close[i] < close[i + j]:
#                 ans = 0
#                 break
#         res.append(ans)
#     res.extend([0 for _ in range(neighbor)])
#     return res
# def k_percent_rule(data):
#     res = []
#     for k in data:
#         if k > 79:
#             res.append(-1)
#         elif k < 21:
#             res.append(1)
#         else:
#             res.append(0)
#     return res
#
#
# def mfi_bound(data):
#     res = []
#     for k in data:
#         if k > 80:
#             res.append(-1)
#         elif (k < 21):
#             res.append(1)
#         else:
#             res.append(0)
#     return res
#
#
# def ma_slope(ma):
#     res = []
#     for i in range(len(ma)):
#         if ma[i] > ma[i - 1]:
#             res.append(1)
#         elif ma[i] < ma[i - 1]:
#             res.append(-1)
#         else:
#             res.append(0)
#     return res
#
#
# def y_predict(data, period):
#     res = []
#     for i in range(len(data)):
#         if i + period >= len(data):
#             res.append(0)
#         else:
#             if data[i] > data[i + period]:
#                 res.append(-1)
#             else:
#                 res.append(1)
#     return res
