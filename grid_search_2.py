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

# assign 0 to null values based on this argument: https://stackoverflow.com/questions/52570199/multivariate-lstm-with-missing-values
df = pd.read_csv("merged_data.csv").fillna(0)
dev_split_idx = df[df["DateTime"] == "2021-05-01 00:00:00"].index.tolist()[0]
test_split_idx = df[df["DateTime"] == "2021-11-01 00:00:00"].index.tolist()[0]
features = df.drop([x for x in df.columns if "Soil Temp" in x], axis=1)
features["DateTime"] = pd.to_datetime(features["DateTime"])
features.set_index("DateTime", inplace=True)
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
epochs = 50

start = past + future
end = start + dev_split_idx

X_train = train_data.values
Y_train = features.iloc[start:end, 6] # selecting just solar generation for now

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    X_train,
    Y_train,
    sequence_length=72, # use 72 hours of historical data
    sampling_rate=1, # make a prediction every 6 hours
    batch_size=batch_size,
)

x_end = len(dev_data) - past - future

x_dev = dev_data.iloc[:x_end].values
y_dev = features.iloc[end:, 6] # selecting just solar for now

dataset_dev = keras.preprocessing.timeseries_dataset_from_array(
    x_dev,
    y_dev,
    sequence_length=72, # use 72 hours of historical data
    sampling_rate=1, # make a prediction every 6 hours
    batch_size=batch_size,
)

for batch in dataset_train.take(1):
    inputs, targets = batch

dropout_rate = [0.2, 0.4, 0.6, 0.8]
reg_penalty = [0.00001, 0.0001, 0.001, 0.01]
reg_method = ["l1", "l2", "dropout"]
designs = ["24-neuron LSTM", "48-neuron GRU", "72-layer LSTM"]
grid_search_results = np.zeros((len(reg_penalty), len(reg_method), len(designs)))

for i in range(len(reg_penalty)):
  for j in range(len(reg_method)):
    dropout = False
    if reg_method[j] == "l1":
        kr = keras.regularizers.l2(reg_penalty[i])
    elif reg_method[j] == "l2":
        kr = keras.regularizers.l2(reg_penalty[i])
    elif reg_method[j] == "dropout":
        dropout = True
    else:
        raise ValueError("Invalid regularization method")

    for k in range(len(designs)):
      if designs[k] == "24-neuron LSTM":
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        if dropout:
            inputs = keras.layers.Dropout(dropout_rate[i])(inputs)
            lstm_out = keras.layers.LSTM(24)(inputs)
        else:
            lstm_out = keras.layers.LSTM(24, kernel_regularizer=kr)(inputs)

        outputs = keras.layers.Dense(1)(lstm_out)
        lr = 0.01
      elif designs[k] == "48-neuron GRU":
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        if dropout:
            inputs = keras.layers.Dropout(dropout_rate[i])(inputs)
            gru_out = keras.layers.GRU(48)(inputs)
        else:
            gru_out = keras.layers.GRU(48, kernel_regularizer=kr)(inputs)

        outputs = keras.layers.Dense(1)(gru_out)
        lr = 0.001
      elif designs[k] == "72-layer LSTM":
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        if dropout:
            inputs = keras.layers.Dropout(dropout_rate[i])(inputs)
            lstm_out = keras.layers.LSTM(72)(inputs)
        else:
            lstm_out = keras.layers.LSTM(72, kernel_regularizer=kr)(inputs)

        outputs = keras.layers.Dense(1)(lstm_out)
        lr = 0.01
      else:
        raise ValueError("Invalid architecture")

      model = keras.Model(inputs=inputs, outputs=outputs)
      model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="huber")
      model.summary()

      # Use ModelCheckpoint callback to regularly save checkpoints,
      # and EarlyStopping callback to interrupt training when validation loss is not improving
      path_checkpoint = "model_checkpoint_" + str(i) + "_" + str(j) + "_" + str(k) + ".h5"
      es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

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

      grid_search_results[i,j,k] = np.min(history.history["val_loss"])

# np.savetxt("grid_search_architecture.csv", grid_search_results, delimiter=",")
np.save("grid_search_reg.npy", grid_search_results)
