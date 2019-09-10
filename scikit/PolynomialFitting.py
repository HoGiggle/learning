# !/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from matplotlib.figure import SubplotParams
from sklearn.linear_model import LinearRegression


def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    linear_regression = LinearRegression(normalize=True)  # normalize=True对数据归一化处理
    pipeline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
    return pipeline

#
# n_dots = 200
# X = np.linspace(-2 * np.pi, 2 * np.pi, n_dots)
# Y = np.sin(X) + 0.2 * np.random.rand(n_dots) - 0.1
# X = X.reshape(-1, 1)
# Y = Y.reshape(-1, 1)


# 视频完成率拟合
arr = np.loadtxt('/Users/hujinjun/work/data/readRate/videoRate.txt')
X = arr[:, 0]
Y = arr[:, 1]
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

X2 = np.hstack([X,X**2,X**3])
reg = LinearRegression()
reg.fit(X2,Y)
y_predict = reg.predict(X2)

print(reg.coef_)#x的系数和X**2的系数[0.99158261 0.51495067]
print(reg.intercept_)#截距1.9865604994175838
print "y = %.4f + (%.4g)x + (%.4g)x^2 + (%.4g)x^3" % (reg.intercept_[0],reg.coef_[0][0],reg.coef_[0][1],reg.coef_[0][2])

plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=0.3))
plt.scatter(X, Y, c='b', alpha=0.5)
plt.title("video read rate", color='red')
plt.text(50, 0.8, "y = %.4f + (%.4g)x + (%.4g)x^2 + (%.4g)x^3" %
         (reg.intercept_[0],reg.coef_[0][0],reg.coef_[0][1],reg.coef_[0][2]))
plt.plot(X,y_predict, color = 'r')
plt.show()


# 图文完成率拟合
arr = np.loadtxt('/Users/hujinjun/work/data/readRate/docRate.txt')
X = arr[:, 0]
Y = arr[:, 1]
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

X2 = np.hstack([X,X**2,X**3])
reg = LinearRegression()
reg.fit(X2,Y)
y_predict = reg.predict(X2)

print(reg.coef_)#x的系数和X**2的系数
print(reg.intercept_)#截距1.
print type(reg.coef_)
print type(reg.intercept_)
print "y = %.4f + (%.4g)x + (%.4g)x^2 + (%.4g)x^3" % (reg.intercept_[0],reg.coef_[0][0],reg.coef_[0][1],reg.coef_[0][2])

plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=0.3))
plt.title("doc read rate", color='red')
plt.text(80, 0.8, "y = %.4f + (%.4g)x + (%.4g)x^2 + (%.4g)x^3" %
         (reg.intercept_[0],reg.coef_[0][0],reg.coef_[0][1],reg.coef_[0][2]))
plt.scatter(X, Y, c='b', alpha=0.5)
plt.plot(X,y_predict, color = 'r')
plt.show()

# 图集完成率拟合
arr = np.loadtxt('/Users/hujinjun/work/data/readRate/atlasRate.txt')
X = arr[:, 0]
Y = arr[:, 1]
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

X2 = np.hstack([X,X**2])
reg = LinearRegression()
reg.fit(X2,Y)
y_predict = reg.predict(X2)

print(reg.coef_)#x的系数和X**2的系数
print(reg.intercept_)#截距1.
print type(reg.coef_)
print type(reg.intercept_)
print "y = %.4f + (%.4g)x + (%.4g)x^2" % (reg.intercept_[0],reg.coef_[0][0],reg.coef_[0][1])

plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=0.3))
plt.title("atlas read rate", color='red')
plt.text(15, 0.8, "y = %.4f + (%.4g)x + (%.4g)x^2" %
         (reg.intercept_[0],reg.coef_[0][0],reg.coef_[0][1]))
plt.scatter(X, Y, c='b', alpha=0.5)
plt.plot(X,y_predict, color = 'r')
plt.show()


# degrees = [2, 3, 5]
# results = []
#
# # todo 参数输出
# for d in degrees:
#     model = polynomial_model(degree=d)
#     model.fit(X, Y)
#
#     train_score = model.score(X, Y)  # 训练集上拟合的怎么样
#     mse = mean_squared_error(Y, model.predict(X))  # 均方误差 cost
#     results.append({"model": model, "degree": d, "score": train_score, "mse": mse})
# for r in results:
#     print("degree: {}; train score: {}; mean squared error: {}".format(r["degree"], r["score"], r["mse"]))
#
#
# plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=0.3))
# for i, r in enumerate(results):
#     # fig = plt.subplot(2, 2, i+1)
#     plt.title("LinearRegression degree={}".format(r["degree"]))
#     plt.plot(X, Y, c='b')
#     # plt.scatter(X, Y, s=5, c='b', alpha=0.5)
#     plt.plot(X, r["model"].predict(X), 'r-')
#     plt.show()
