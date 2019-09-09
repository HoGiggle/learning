# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class LR():
    def __init__(self):
        self.learn_rate = 0.03
        self.limit = 0.001
        self.epoch_size = 128
        self.l1_lambda = 0.2
        self.l2_lambda = 0

    def fit(self, train_x, train_y):
        self.feature_size = len(train_x[0])
        self.train_size = len(train_x)
        self._update(train_x, train_y)
        return self

    def _update(self, train_x, train_y):
        if self.l2_lambda > 0:
            self.w, self.b = self._updater_l2(train_x, train_y)
        elif self.l1_lambda > 0:
            self.w, self.b = self._updater_l1(train_x, train_y)
        else:
            self.w, self.b = self._updater(train_x, train_y)
        return self

    def _updater(self, train_x, train_y):
        w, b = np.array([0.0] * self.feature_size), 0.0
        for epoch in range(self.epoch_size):
            alpha = self.learn_rate / np.sqrt(epoch + 1)   # 学习率缩减
            grad_w, grad_b = self._bgd(train_x, train_y, w, b)
            w = w - alpha * grad_w
            b = b - alpha * grad_b
        return w, b

    def _updater_l2(self, train_x, train_y):
        w, b = np.array([0.0] * self.feature_size), 0.0
        for epoch in range(self.epoch_size):
            alpha = self.learn_rate / np.sqrt(epoch + 1)   # 学习率缩减
            grad_w, grad_b = self._bgd(train_x, train_y, w, b)
            w = w - alpha * (grad_w + self.l2_lambda * w)
            b = b - alpha * grad_b
        return w, b


    def _updater_l1(self, train_x, train_y):
        w, b = np.array([0.0] * self.feature_size), 0.0
        for epoch in range(self.epoch_size):
            alpha = self.learn_rate / np.sqrt(epoch + 1)
            grad_w, grad_b = self._bgd(train_x, train_y, w, b)
            w = self._proximal_batch(w - alpha * grad_w, self.l1_lambda * alpha)
            b = b - alpha * grad_b
        return w, b

    def _bgd(self, train_x, train_y, w, b):
        grad_w, grad_b = np.array([0.0] * len(w)), 0.0
        for i, x in enumerate(train_x):
            pt = self._predict(w, x, b)
            grad_w += (pt - train_y[i]) * x
            grad_b += (pt - train_y[i])
        grad_w /= self.train_size
        grad_b /= self.train_size
        return grad_w, grad_b

    def _predict(self, w, x, b):
        h = np.dot(w, x) + b
        return self._sigmoid(h)

    def predict_proba(self, X):
        return np.array([self._predict(self.w, x, self.b) for x in X])

    def score(self, X, Y):
        pred = self.predict_proba(X)
        count = 0
        for i, y in enumerate(Y):
            if y == 0 and pred[i] <= 0.5:
                count += 1
            elif y == 1 and pred[i] > 0.5:
                count += 1

        return count * 1.0 / (len(X) * 1.0)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _proximal_batch(self, w, shrinkage):
        return [self._proximal(wi, shrinkage) for wi in w]

    def _proximal(self, wi, shrinkage):

        if wi > shrinkage:
            return wi - shrinkage
        elif wi < -shrinkage:
            return wi + shrinkage
        else:
            return 0.0


if __name__ == '__main__':
    # 1.加载数据
    iris = datasets.load_breast_cancer()
    X = iris.data
    Y = iris.target
    # np.unique(Y)   # out: array([0, 1, 2])

    # 2.拆分测试集、训练集。
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # 设置随机数种子，以便比较结果。

    # 3.标准化特征值
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    print "train: %s" % X_train[0]
    print "standard train: %s" % X_train_std[0]

    # 4. 训练逻辑回归模型
    # model = linear_model.LogisticRegression()
    model = LR()
    model.fit(X_train_std, Y_train)

    # 5. 预测
    # predict = model.predict_proba(X_test_std)[:, 1]
    predict = model.predict_proba(X_test_std)
    acc = model.score(X_test_std, Y_test)

    print "acc: ", acc
    print "auc: ", roc_auc_score(Y_test, predict)
    print "w: ", model.w
    print "b: ", model.b
