import pandas as pd
import numpy as np
import keras.models as models
import keras.layers as layers
# from keras.models import Model
# from keras.layers import Input, LSTM, Dense
import random
# import ipdb
import os

delay = 4
latent_dim = 8
epochs = 1000
validation_split = 0.2
test_split = 0.2

data_path = os.path.join('../../config', 'USDJPY5.xls')
data = pd.read_excel(data_path)
column_names = ['Open', 'High', 'Low', 'Close']
series = data[column_names].values
input_dim = series.shape[1]

### normalization
means = series.mean(axis=0)
stds  = series.std(axis=0)
normalized = (series - means) / stds


### test/val/test split
## This is obviously not the best split, but it's simple given the sequential nature of the data
split_test = int(normalized.shape[0] * (1 - test_split))
split_validation = int(normalized.shape[0] * (1 - test_split - validation_split))
X = normalized[:split_validation-delay]
Y = normalized[delay:split_validation]
assert(X.shape == Y.shape)
X_val = normalized[split_validation:split_test-delay]
Y_val = normalized[split_validation+delay:split_test]
assert(X_val.shape == Y_val.shape)
# won't be using X_tst for now
X_tst = normalized[split_test:-delay]
Y_tst = normalized[split_test+delay:]
assert(X_tst.shape == Y_tst.shape)
print('train size: %d, validation size: %d, test size: %d' % (X.shape[0], X_val.shape[0], X_tst.shape[0]))


print('series shapes:', normalized.shape)
# ipdb.set_trace()

### defining the neural network
inputs = layers.Input(shape=(None, input_dim))
## I think None refers to the size of the time-index in input

x = inputs
## we can add Dense layers at the start and the end
# x = layers.Dense(latent_dim, activation='relu')(x)
x = layers.LSTM(latent_dim, return_sequences=True)(x)
## not setting the value of initial_state for now
x = layers.LSTM(latent_dim, return_sequences=True)(x)
x = layers.Dense(input_dim, activation='relu')(x)
outputs = x

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse') #, metrics=['accuracy'])
## X[None,] just adds an extra dimension to the input
## we need it since we only have a single data point ...
model.fit(X[None,], Y[None,], epochs=epochs, verbose=2)

### validating the result
yhat = model.predict(X_val[None,])
print('MSE: %7.4f' % ((Y_val - yhat) ** 2).mean())
sample = random.sample(range(Y_val.shape[0]), 10)
val_df = pd.DataFrame(
    data=np.concatenate([yhat[0][sample], Y_val[sample]], axis=1),
    columns=column_names + [x+'_hat' for x in column_names]
)
