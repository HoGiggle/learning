# !/usr/bin/python
# coding:utf8
import numpy as np
from tensorflow import keras
import tensorflow as tf


def multi_hot_seq(seq, dim):
    results = np.zeros((len(seq), dim))
    for i, word in enumerate(seq):
        results[i, word] = 1.0
    return results


def model_build(num_word):
    model = keras.Sequential([
        keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(num_word,)),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    return model



if __name__ == '__main__':
    NUM_WORDS = 10000
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
    # print type(train_data)
    # print train_data.shape
    # print type(enumerate(train_data))
    # print type(train_data[0])
    # multi_hot_seq(train_data[0:2], NUM_WORDS)

    # transform data to one_hot
    train_data = multi_hot_seq(train_data, dim=NUM_WORDS)
    test_data = multi_hot_seq(test_data, dim=NUM_WORDS)

    # build model
    model = model_build(NUM_WORDS)
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc', 'binary_crossentropy'])
    model.summary()

    history = model.fit(train_data, train_labels, batch_size=512,
                        epochs=20, verbose=2,
                        validation_data=(test_data, test_labels))
