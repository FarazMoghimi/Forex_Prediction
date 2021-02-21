from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.layers import LSTM
from keras import optimizers


def get_model(window_length, nb_features, lr, hidden_size=10, use_dropout=False):
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(window_length, nb_features), return_sequences=True))
    model.add(Activation('relu'))
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Activation('relu'))
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(LSTM(hidden_size, return_sequences=False))
    model.add(Activation('relu'))
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(optimizer=optimizers.Adam(lr=lr),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model