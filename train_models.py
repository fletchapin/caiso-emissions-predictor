import os
import pandas as pd
import tensorflow as tf
import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow import keras
from keras.layers import Layer
import keras.backend as K

# Following this example: https://keras.io/examples/timeseries/timeseries_weather_forecasting/
def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

def create_model(input_data, design, dataset_train):
    for batch in dataset_train.take(1):
        inputs, targets = batch
    if design == "24-neuron LSTM":
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        lstm_out = keras.layers.LSTM(
            24,
            kernel_regularizer=keras.regularizers.l1(0.00001),
            activation="tanh"
        )(inputs)
        outputs = keras.layers.Dense(1)(lstm_out)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
          optimizer=keras.optimizers.Adam(learning_rate=0.000001),
          loss="huber",
          metrics=[
            keras.metrics.MeanAbsoluteError(name='abs'),
            keras.metrics.RootMeanSquaredError(name='rmse')
          ]
        )
    elif design == "48-neuron GRU":
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        dropout = keras.layers.Dropout(0.8)(inputs)
        gru_out = keras.layers.GRU(48, activation="tanh")(dropout)
        outputs = keras.layers.Dense(1)(gru_out)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
          optimizer=keras.optimizers.Adam(learning_rate=0.001),
          loss="mae",
          metrics=[
            keras.metrics.MeanAbsoluteError(name='abs'),
            keras.metrics.RootMeanSquaredError(name='rmse')
          ]
        )
    elif design == "72-layer LSTM":
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        dropout = keras.layers.Dropout(0.8)(inputs)
        lstm_out = keras.layers.LSTM(72, activation="tanh")(dropout)
        outputs = keras.layers.Dense(1)(lstm_out)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
          optimizer=keras.optimizers.Adam(learning_rate=0.01),
          loss="huber",
          metrics=[
            keras.metrics.MeanAbsoluteError(name='abs'),
            keras.metrics.RootMeanSquaredError(name='rmse')
          ]
        )
    return model

def select_input_data(input_data, weather_data):
    if weather_data == "all":
        if input_data == "normalized":
            df = pd.read_csv("merged_data_normalized_v2.csv").fillna(0)
        elif input_data == "unnormalized":
            df = pd.read_csv("merged_data_v2.csv").fillna(0)
    elif weather_data == "gilroy":
        if input_data == "normalized":
            df = pd.read_csv("merged_data_normalized.csv").fillna(0)
        elif input_data == "unnormalized":
            df = pd.read_csv("merged_data.csv").fillna(0)
    elif weather_data == "none":
        if input_data == "normalized":
            df = pd.read_csv("eia_data_normalized.csv").fillna(0)
        elif input_data == "unnormalized":
            df = pd.read_csv("eia_data.csv").fillna(0)

    return df

input_types = ["normalized", "unnormalized"]
weather_types = ["all", "gilroy", "none"]
designs = ["24-neuron LSTM", "48-neuron GRU", "72-layer LSTM"]

for i in range(len(input_types)):
    for j in range(len(weather_types)):
        for k in range(len(designs)):
            for h in range(8):
                df = select_input_data(input_types[i], weather_types[j])

                dev_split_idx = df[df["DateTime"] == "5/1/2021 0:00"].index.tolist()[0]
                test_split_idx = df[df["DateTime"] == "11/1/2021 0:00"].index.tolist()[0]
                features = df.drop([x for x in df.columns if "Soil Temp" in x], axis=1)
                features["DateTime"] = pd.to_datetime(features["DateTime"])
                features.set_index("DateTime", inplace=True)
                if input_types[i] == "unnormalized":
                    features = normalize(features, dev_split_idx)

                # Save hour, month, and weekday as features
                features["hour"] = features.index.hour
                features["month"] = features.index.month
                features["weekday"] = features.index.weekday

                train_data = features.iloc[0:dev_split_idx]
                dev_data = features.iloc[dev_split_idx:test_split_idx]
                test_data = features.iloc[test_split_idx:]

                # Pin these hyperparameters
                past = 72
                future = 24
                batch_size = 256
                epochs = 100

                start = past + future
                end = start + dev_split_idx

                X_train = train_data.values
                Y_train = features.iloc[start:end, h] # selecting just solar generation for now

                dataset_train = keras.preprocessing.timeseries_dataset_from_array(
                    X_train,
                    Y_train,
                    sequence_length=72, # use 72 hours of historical data
                    sampling_rate=1, # make a prediction every 6 hours
                    batch_size=batch_size,
                )

                x_end = len(dev_data) - past - future

                x_dev = dev_data.iloc[:x_end].values
                y_dev = features.iloc[end:, h] # selecting just solar for now

                dataset_dev = keras.preprocessing.timeseries_dataset_from_array(
                    x_dev,
                    y_dev,
                    sequence_length=72, # use 72 hours of historical data
                    sampling_rate=1, # make a prediction every 6 hours
                    batch_size=batch_size,
                )

                model = create_model(input_types[i], designs[k], dataset_train)
                model.summary()

                # Use ModelCheckpoint callback to regularly save checkpoints,
                # and EarlyStopping callback to interrupt training when validation loss is not improving
                path_checkpoint = "model_checkpoint_" + str(i) + "_" + str(j) + "_" + str(k) + "_" + str(h) + ".h5"
                es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10)

                modelckpt_callback = keras.callbacks.ModelCheckpoint(
                  monitor="val_loss",
                  filepath=path_checkpoint,
                  verbose=1,
                  save_weights_only=True,
                  save_best_only=True,
                )

                history = model.fit(
                  dataset_train,
                  epochs=epochs,
                  validation_data=dataset_dev,
                  callbacks=[es_callback, modelckpt_callback],
                )
