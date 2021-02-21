import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.layers import LSTM
from keras import regularizers
from keras import optimizers
from keras import losses
from keras import activations
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Bidirectional 
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger 
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import json
from keras import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from keras.models import load_model


# import data
data_original = pd.read_csv('../../data/USDJPY_M5_2016.csv')
#data_original = pd.read_csv('/my_data/EURUSD_15m_BID_01.01.2010-31.12.2016.csv')
data = data_original.iloc[:,1:5]
data = data.astype(np.float64)

# check for zeros in data
zeros = np.where(data.values == 0)[0]
nans = np.where(data.values == np.nan)[0]
nulls = np.where(data.values == None)[0]

def get_Y(data, forecast, target):
    close = np.array(data.loc[96:, target])
    close_forecast = data.loc[96:, target].shift(-forecast).values
    Y = []  
    for n in range(0,len(close)):
        if close_forecast[n] > close[n]:
            Y.append([1,0])
        elif n in range(len(close)-forecast, len(close)):
            break
        else:
            Y.append([0,1])
    return np.array(Y)

# split data
split_point = int(len(data)*0.8)
X_train_original = data[:split_point]
X_test_original = data[split_point:-1]
Y = get_Y(data, 1, 'Close')
Y_train = Y[:split_point]
Y_test = Y[split_point:]

# get collumn arrays
openp_train = np.array(X_train_original.loc[:, 'Open']).reshape(-1,1)
highp_train = np.array(X_train_original.loc[:, 'High']).reshape(-1,1)
lowp_train = np.array(X_train_original.loc[:, 'Low']).reshape(-1,1)
closep_train = np.array(X_train_original.loc[:, 'Close']).reshape(-1,1)
openp_test = np.array(X_test_original.loc[:, 'Open']).reshape(-1,1)
highp_test = np.array(X_test_original.loc[:, 'High']).reshape(-1,1)
lowp_test = np.array(X_test_original.loc[:, 'Low']).reshape(-1,1)
closep_test = np.array(X_test_original.loc[:, 'Close']).reshape(-1,1)

# normalize X
sc_o = StandardScaler()
o_train = sc_o.fit_transform(openp_train).flatten()
sc_h = StandardScaler()
h_train = sc_h.fit_transform(highp_train).flatten()
sc_l = StandardScaler()
l_train = sc_l.fit_transform(lowp_train).flatten()
sc_c = StandardScaler()
c_train = sc_c.fit_transform(closep_train).flatten()

# normalize Y
o_test = sc_o.transform(openp_test).flatten()
h_test = sc_h.transform(highp_test).flatten()
l_test = sc_o.transform(lowp_test).flatten()
c_test = sc_h.transform(closep_test).flatten()

# Now we have to join them again for data preparation
o = np.append(o_train, o_test)
h = np.append(h_train, h_test)
l = np.append(l_train, l_test)
c = np.append(c_train, c_test)

def prepare_data(o_n, h_n, l_n, c_n, window_size, split_point):
    X = []  
    for i in range(window_size, len(c_n)): 
        try:
            o=o_n[i-window_size:i]
            h=h_n[i-window_size:i]
            l=l_n[i-window_size:i]
            c=c_n[i-window_size:i]

            x_i = np.stack((o, h, l, c), axis=-1)
            X.append(x_i)
        
        except Exception as e:
            break
    X = np.array(X)
    X_train = X[:split_point]
    X_test = X[split_point:]
    return X_train, X_test

X_train, X_test = prepare_data(o, h, l, c, 96, split_point)

########################## LSTM network ######################################
model = Sequential()
model.add(BatchNormalization(input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(10, return_sequences=True))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(LSTM(10,return_sequences=False))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(optimizer=optimizers.Adam(0.1), 
			  loss='binary_crossentropy',
              metrics = ['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001, epsilon=0.0003, verbose=1)
model_history = model.fit(X_train, Y_train, 
                          validation_data=[X_test, Y_test], 
                          epochs = 100,     
                          batch_size = 512,
                          shuffle = True, verbose=1, callbacks=[reduce_lr])
model.save('my_model2.h5')  # creates a HDF5 file 'my_model.h5'

