# !/usr/bin/python
# -*- coding: utf-8 -*-

from nn.activations import relu_forward, relu_backward
# 定义前向传播和反向传播
from nn.layers import *
from nn.losses import cross_entropy_loss
from nn.optimizers import SGD, RmsProp
from nn.load_mnist import load_mnist_datasets
from nn.utils import to_categorical
from sklearn.datasets import fetch_mldata


# 定义前向传播
def forward(X):
    neurons["conv1"] = conv_forward(X.astype(np.float64), weights["K1"], weights["b1"])
    neurons["conv1_relu"] = relu_forward(neurons["conv1"])
    neurons["maxp1"] = max_pooling_forward(neurons["conv1_relu"].astype(np.float64), pooling=(2, 2))

    neurons["flatten"] = flatten_forward(neurons["maxp1"])

    neurons["fc2"] = fc_forward(neurons["flatten"], weights["W2"], weights["b2"])
    neurons["fc2_relu"] = relu_forward(neurons["fc2"])

    neurons["y"] = fc_forward(neurons["fc2_relu"], weights["W3"], weights["b3"])

    return neurons["y"]


# 定义反向传播
def backward(X, y_true):
    loss, dy = cross_entropy_loss(neurons["y"], y_true)
    gradients["W3"], gradients["b3"], gradients["fc2_relu"] = fc_backward(dy, weights["W3"], neurons["fc2_relu"])
    gradients["fc2"] = relu_backward(gradients["fc2_relu"], neurons["fc2"])

    gradients["W2"], gradients["b2"], gradients["flatten"] = fc_backward(gradients["fc2"], weights["W2"],
                                                                         neurons["flatten"])

    gradients["maxp1"] = flatten_backward(gradients["flatten"], neurons["maxp1"])

    gradients["conv1_relu"] = max_pooling_backward(gradients["maxp1"].astype(np.float64),
                                                   neurons["conv1_relu"].astype(np.float64), pooling=(2, 2))
    gradients["conv1"] = relu_backward(gradients["conv1_relu"], neurons["conv1"])
    gradients["K1"], gradients["b1"], _ = conv_backward(gradients["conv1"], weights["K1"], X)
    return loss


# 获取精度
def get_accuracy(X, y_true):
    y_predict = forward(X)
    return np.mean(np.equal(np.argmax(y_predict, axis=-1),
                            np.argmax(y_true, axis=-1)))


# 定义权重、神经元、梯度
import numpy as np

weights = {}
weights_scale = 1e-2
filters = 1
fc_units = 64
weights["K1"] = weights_scale * np.random.randn(1, filters, 3, 3).astype(np.float64)
weights["b1"] = np.zeros(filters).astype(np.float64)
weights["W2"] = weights_scale * np.random.randn(filters * 13 * 13, fc_units).astype(np.float64)
weights["b2"] = np.zeros(fc_units).astype(np.float64)
weights["W3"] = weights_scale * np.random.randn(fc_units, 10).astype(np.float64)
weights["b3"] = np.zeros(10).astype(np.float64)

# 初始化神经元和梯度
neurons = {}
gradients = {}


train_set, val_set, test_set = load_mnist_datasets('/Users/hujinjun/work/ml_data_set/mnist.pkl.gz')
print len(train_set[0]), len(val_set[0]), len(test_set[0])
train_x, val_x, test_x = np.reshape(train_set[0], (-1, 1, 28, 28)), np.reshape(val_set[0][:500], (-1, 1, 28, 28)), np.reshape(
    test_set[0][:500], (-1, 1, 28, 28))
train_y, val_y, test_y = to_categorical(train_set[1]), to_categorical(val_set[1][:500]), to_categorical(test_set[1][:500])

# 随机选择训练样本
train_num = train_x.shape[0]
def next_batch(batch_size):
    idx=np.random.choice(train_num,batch_size)
    return train_x[idx],train_y[idx]

x,y= next_batch(16)
print("x.shape:{},y.shape:{}".format(x.shape,y.shape))


# 初始化变量
batch_size = 2
steps = 3000

# 更新梯度
sgd = SGD(weights, lr=0.01)

for s in range(steps):
    X, y = next_batch(batch_size)

    # 前向过程
    forward(X)
    # 反向过程
    loss = backward(X, y)

    sgd.iterate(weights, gradients)

    if s % 100 == 0:
        print("\n step:{} ; loss:{}".format(s, loss))
        idx = np.random.choice(len(val_x), 50)
        print(" train_acc:{};  val_acc:{}".format(get_accuracy(X, y), get_accuracy(val_x[idx], val_y[idx])))

print("\n final result test_acc:{};  val_acc:{}".
      format(get_accuracy(test_x, test_y), get_accuracy(val_x, val_y)))

# 随机查看预测结果
import matplotlib.pyplot as plt

idx=np.random.choice(test_x.shape[0],3)
x,y=test_x[idx],test_y[idx]
y_predict = forward(x)
for i in range(3):
    plt.figure(figsize=(3,3))
    plt.imshow(np.reshape(x[i],(28,28)))
    plt.show()
    print("y_true:{},y_predict:{}".format(np.argmax(y[i]),np.argmax(y_predict[i])))
