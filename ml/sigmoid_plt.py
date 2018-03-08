#!/usr/bin/python
# coding=utf-8
from numpy import *
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


if __name__ == '__main__':
    X = arange(-5.0, 5.0, 0.1)
    shape(X)
    Y = sigmoid(X)
    plt.plot(X, Y)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('sigmoid')
    plt.show()

