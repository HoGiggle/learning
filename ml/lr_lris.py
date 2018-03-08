#!/usr/bin/python
# coding=utf-8

from numpy import *
from sklearn.datasets import load_iris     # import datasets
from sklearn.linear_model import LogisticRegression

# load the dataset: iris
iris = load_iris()
samples = iris.data
target = iris.target

# import the LogisticRegression
classifier = LogisticRegression()  # 使用类，参数全是默认的
classifier.fit(samples, target)  # 训练数据来学习，不需要返回值

x = classifier.predict([5, 3, 5, 2.5])  # 测试数据，分类返回标记
print x