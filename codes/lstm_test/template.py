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
from sklearn.preprocessing import MinMaxScaler

y_period = 6
window_size = 60
predict_column = 'Close'
batch_size = 1024
# import data
data_original = pd.read_excel('../../result/technical_indicator/EURUSD M5_included_ma.xls')
data = data_original.iloc[100:, 3:40]
data = data.reset_index(drop=True)
data = data.astype(np.float64)

def create_dataset(dataset, look_back, target, forecast):
    dataY, dataX = [], []
    for i in range(look_back-1, len(dataset)-forecast):
        x = dataset.loc[(i-look_back+1):i, :].values
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # x = scaler.fit_transform(x)
        # print(x)
        if dataset.loc[i+forecast, target] > dataset.loc[i, target]:
            y = [1, 0]
        else:
            y = [0, 1]
        # y = dataset.loc[i+forecast, target]
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)




all_x, all_y = create_dataset(data, window_size, predict_column, y_period)
split_point = int(len(data) * 0.9)
x_train = all_x[:split_point]
x_test = all_x[split_point:]
y_train = all_y[:split_point]
y_test = all_y[split_point:]

########################## LSTM network ######################################
model = Sequential()
model.add(BatchNormalization(input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(20, return_sequences=True))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
# model.add(BatchNormalization())
model.add(LSTM(20, return_sequences=False))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# model.add(BatchNormalization())
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(optimizer=optimizers.Adam(lr=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.000001, verbose=1, epsilon=0.0002)
model_history = model.fit(x_train, y_train,
                          validation_data=[x_test, y_test],
                          epochs=100,
                          batch_size=batch_size,
                          shuffle=False, verbose=1, callbacks=[reduce_lr])
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

