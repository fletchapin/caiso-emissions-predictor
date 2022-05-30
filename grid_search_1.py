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

# Idea for attention layer: https://arxiv.org/pdf/2203.09170.pdf
# Attention layer code: https://machinelearningmastery.com/adding-a-custom-attention-layer-to-recurrent-neural-network-in-keras/
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1),
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1),
                               initializer='zeros', trainable=True)
        super(attention, self).build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

learning_rates = [0.00001, 0.0001, 0.001, 0.01]
num_neurons = [24, 48, 72, 96, 120, 144]
designs = ["LSTM", "LSTM_attention", "GRU", "GRU_attention"]
grid_search_results = np.zeros((len(learning_rates), len(num_neurons), len(designs)))

for i in range(len(learning_rates)):
  for j in range(len(num_neurons)):
    for k in range(len(designs)):
      if designs[k] == "LSTM":
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        lstm_out = keras.layers.LSTM(num_neurons[j])(inputs)
        outputs = keras.layers.Dense(1)(lstm_out)
      elif designs[k] == "LSTM_attention":
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        lstm = keras.layers.LSTM(num_neurons[j], return_sequences=True)(inputs)
        attention_layer = attention()(lstm)
        outputs = keras.layers.Dense(1)(attention_layer)
      elif designs[k] == "GRU":
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        gru_out = keras.layers.GRU(num_neurons[j])(inputs)
        outputs = keras.layers.Dense(1)(gru_out)
      elif designs[k] == "GRU_attention":
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        gru = keras.layers.GRU(num_neurons[j], return_sequences=True)(inputs)
        attention_layer = attention()(gru)
        outputs = keras.layers.Dense(1)(attention_layer)
      else:
        raise ValueError("Invalid architecture")

      model = keras.Model(inputs=inputs, outputs=outputs)
      model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rates[i]), loss="huber")
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
np.save("data/grid_search_architecture.npy", grid_search_results)
