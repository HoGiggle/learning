# !/usr/bin/python
# coding:utf8
from aetypes import end

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


boston_house = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_house.load_data()
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

# use DataFrame explore data
df = pd.DataFrame(train_data, columns=column_names)
print df.head()

# standardization
mean = train_data.mean(axis=0)
print(train_data.shape)
print(type(mean), mean.shape)
std = train_data.std(axis=0)
print(type(std), std.shape)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
print(type(train_data), train_data.shape)


def build():
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation=tf.nn.relu,
                                 input_shape=(train_data.shape[1],)))
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='mse',
                  metrics=['mae'])
    return model


class PrintDot(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            print('')
        print '.', end('')


model1 = build()
model1.summary()

EPOCHS = 500
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model1.fit(train_data, train_labels, epochs=EPOCHS,
                     validation_split=0.2, verbose=0,
                     callbacks=[PrintDot(), early_stop])

print("history type: {}".format(type(history)))


# plot
def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()


plot_history(history)


# test data
[loss, mae] = model1.evaluate(test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

test_predictions = model1.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")
plt.show()






