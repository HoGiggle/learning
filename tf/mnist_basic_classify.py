# !/usr/bin/python
# coding:utf8

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.get_cmap('binary'))

    pre_label = np.argmax(predictions_array)
    if pre_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[pre_label], 100*np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    bar_plot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    pre_label = np.argmax(predictions_array)

    bar_plot[pre_label].set_color('red')
    bar_plot[true_label].set_color('blue')


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print type(train_images)
print train_images.shape
print len(train_labels)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0
# train_image show RGB
# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# train_image show binary
# plt.figure(figsize=(6, 6))
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_images[i], cmap=plt.get_cmap('binary'))
#     plt.xlabel(class_names[test_labels[i]])
# plt.show()

# model set
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(train_images, train_labels, epochs=5)

model.save('')
# test acc
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test acc = ', test_acc)
print('Test loss = ', test_loss)
pre = model.predict(test_images)
# print type(pre)
# for i in range(9):
#     print('test[%d] = %s' % (i, class_names[int(np.argmax(pre[i]))]))

n = 0
while n < 10:
    i = raw_input()
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(int(i), pre, test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(int(i), pre, test_labels)
    plt.show()
    n = n + 1





