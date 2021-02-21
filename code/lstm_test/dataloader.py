import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_Y(data, forecast, target, window_size):
    close = np.array(data.loc[window_size:, target])
    close_forecast = data.loc[window_size:, target].shift(-forecast).values
    Y = []
    for n in range(0, len(close)):
        if close_forecast[n] > close[n]:
            Y.append([1, 0])
        else:
            Y.append([0, 1])
    return np.array(Y)


def prepare_data(o_n, h_n, l_n, c_n, window_size, split_point):
    X = []
    for i in range(window_size, len(c_n)):
        try:
            o = o_n[i - window_size:i]
            h = h_n[i - window_size:i]
            l = l_n[i - window_size:i]
            c = c_n[i - window_size:i]

            x_i = np.stack((o, h, l, c), axis=-1)
            X.append(x_i)

        except Exception as e:
            break
    # FIXME: pretty sure the X_train has data about the first 95 X_test data points ...
    # Plus, the data are highly correlated, but I don't have a better idea
    X = np.array(X)
    X_train = X[:split_point]
    X_test = X[split_point:]
    return X_train, X_test


def load_data(path, window_size, nb_forecast_steps):
    # import data
    data_original = pd.read_csv('../../config/EURUSD_15m_BID_01.01.2010-31.12.2016.csv')
    #data_original = pd.read_csv('/dat_mt_usdjpy_m1_2017/DAT_MT_USDJPY_M1_2017.csv')
    data = data_original.iloc[:, 1:6]
    data = data.astype(np.float64)

    # check for zeros in data
    zeros = np.where(data.values == 0)[0]
    nans = np.where(data.values == np.nan)[0]
    nulls = np.where(data.values == None)[0]

    print('zeros, nans, nulls:', zeros, nans, nulls)

    # split data
    split_point = int(len(data) * 0.8)
    X_train_original = data[:split_point]
    X_test_original = data[split_point:]
    Y = get_Y(data, nb_forecast_steps, 'Close', window_size)
    Y_train = Y[:split_point]
    Y_test = Y[split_point:]

    # get collumn arrays
    openp_train = np.array(X_train_original.loc[:, 'Open']).reshape(-1, 1)
    highp_train = np.array(X_train_original.loc[:, 'High']).reshape(-1, 1)
    lowp_train = np.array(X_train_original.loc[:, 'Low']).reshape(-1, 1)
    closep_train = np.array(X_train_original.loc[:, 'Close']).reshape(-1, 1)
    openp_test = np.array(X_test_original.loc[:, 'Open']).reshape(-1, 1)
    highp_test = np.array(X_test_original.loc[:, 'High']).reshape(-1, 1)
    lowp_test = np.array(X_test_original.loc[:, 'Low']).reshape(-1, 1)
    closep_test = np.array(X_test_original.loc[:, 'Close']).reshape(-1, 1)

    # normalize X: nope, normalize train, Y is already calculated and doesn't need normalization
    sc_o = StandardScaler()
    o_train = sc_o.fit_transform(openp_train).flatten()
    sc_h = StandardScaler()
    h_train = sc_h.fit_transform(highp_train).flatten()
    sc_l = StandardScaler()
    l_train = sc_l.fit_transform(lowp_train).flatten()
    sc_c = StandardScaler()
    c_train = sc_c.fit_transform(closep_train).flatten()

    # normalize Y: nope, normalize test based on statistics of train
    o_test = sc_o.transform(openp_test).flatten()
    h_test = sc_h.transform(highp_test).flatten()
    l_test = sc_o.transform(lowp_test).flatten()
    c_test = sc_h.transform(closep_test).flatten()

    # Now we have to join them again for data preparation
    o = np.append(o_train, o_test)
    h = np.append(h_train, h_test)
    l = np.append(l_train, l_test)
    c = np.append(c_train, c_test)

    X_train, X_test = prepare_data(o, h, l, c, window_size, split_point)

    return (X_train, Y_train), (X_test, Y_test)