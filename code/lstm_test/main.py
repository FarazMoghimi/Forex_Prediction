import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras import losses
from keras import activations
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import json
from keras import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

from argparser import Args
from dataloader import load_data
from model import get_model


def main():
    args = Args(
        nb_forecast_steps=5,
        window_size=64,
        net_layer_size=10,
        lr=1e-4,
        batch_size=1024,
        nb_epochs=100,
        use_dropout=False,
        shuffle=True,
        load_path='',
        store_path='my_model.h5',
        data_path='../../config/EURUSD_15m_BID_01.01.2010-31.12.2016.csv',
        # data_path='/dat_mt_usdjpy_m1_2017/DAT_MT_USDJPY_M1_2017.csv',
    ).parse()

    (X_train, Y_train), (X_test, Y_test) = load_data(
                                                args.data_path,
                                                window_size=args.window_size,
                                                nb_forecast_steps=args.nb_forecast_steps)

    if args.load_path:
        model = load_model(args.load_path)
    else:
        model = get_model(
            window_length=args.window_size,
            nb_features=4,
            lr=args.lr,
            hidden_size=args.net_layer_size,
            use_dropout=args.use_dropout,
        )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=0.000001, verbose=1)

    model_history = model.fit(
                            X_train, Y_train,
                            validation_data=[X_test, Y_test],
                            epochs=args.nb_epochs,
                            batch_size=args.batch_size,
                            shuffle=args.shuffle,
                            verbose=1, callbacks=[reduce_lr])

    model.save(args.store_path)  # stores net as a HDF5 file


if __name__ == '__main__':
    main()