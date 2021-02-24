# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from matplotlib.figure import SubplotParams
from sklearn import linear_model
np.set_printoptions(suppress=True)
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

'''
图集分段
'''
# raw = np.loadtxt('/Users/hujinjun/work/data/readRateDis/atlas_dis.txt',delimiter='~')
# raw = np.column_stack((raw, raw[:, 1] / raw[:, 0])) # 完播率特征
# raw = np.column_stack((raw, raw[:, 0] // 12)) # 总时长分钟分段特征
#
# lowRateFlag = [0 if x < 0.1 else 1 for x in raw[:, 6]]
# highRateFlag = [1 if x > 0.8 else 0 for x in raw[:, 6]]
# raw = np.column_stack((raw, lowRateFlag)) # 完播率 <= 15%, 0/1特征
# raw = np.column_stack((raw, highRateFlag)) # 完播率 <= 15%, 0/1特征
#
# # arr = raw[(raw[:, 2] < 1) & (raw[:, 0] <= 12)]  # 图片数 [1, 12)
# # arr = raw[(raw[:, 2] < 1) & (raw[:, 0] > 12) & (raw[:, 0] < 23)]  # 图片数 [12, 23)
# arr = raw[(raw[:, 2] < 1) & (raw[:, 0] >= 23)]  # 图片数 [23, )
# arr = arr[:, [5, 0, 1, 6, 8, 9]]



'''
 视频分段
'''
# raw = np.loadtxt('/Users/hujinjun/work/data/readRateDis/video_dis.txt', delimiter='~')
# raw = np.column_stack((raw, raw[:, 1] / raw[:, 0])) # 完播率特征
# raw = np.column_stack((raw, raw[:, 0] // 60)) # 总时长分钟分段特征
#
# lowRateFlag = [0 if x < 0.2 else 1 for x in raw[:, 6]]
# highRateFlag = [1 if x > 0.8 else 0 for x in raw[:, 6]]
# raw = np.column_stack((raw, lowRateFlag)) # 完播率 <= 15%, 0/1特征
# raw = np.column_stack((raw, highRateFlag)) # 完播率 <= 15%, 0/1特征
#
# time10 = [0 if x <= 10 else 1 for x in raw[:, 1]]
# raw = np.column_stack((raw, time10)) # 阅读时长 <= 10s, 0/1特征
#
# arr = raw[(raw[:, 1] > 3) & (raw[:, 2] < 1) & (raw[:, 0] > 5) & (raw[:, 0] <= 60)] # 视频时长 (5, 60]  阅读时长 > 3
# # arr = raw[(raw[:, 1] > 3) & (raw[:, 2] < 1) & (raw[:, 0] > 60) & (raw[:, 0] <= 360)] # 视频时长 (5, 60]  阅读时长 > 3
# # arr = raw[(raw[:, 1] > 5) & (raw[:, 2] < 1) & (raw[:, 0] > 360) & (raw[:, 0] <= 900)] # 视频时长 (5, 60]  阅读时长 > 3
# arr = arr[:, [5, 0, 1, 6, 8, 9]]

'''
 图文分段
'''
raw = np.loadtxt('/Users/hujinjun/work/data/readRateDis/doc_dis.txt', delimiter='~')
raw = np.column_stack((raw, raw[:, 1] / raw[:, 0])) # 完播率特征
raw = np.column_stack((raw, raw[:, 0] // 60)) # 总时长分钟分段特征

lowRateFlag = [0 if x > 0.1 else 1 for x in raw[:, 6]]
highRateFlag = [1 if x > 0.8 else 0 for x in raw[:, 6]]
raw = np.column_stack((raw, lowRateFlag)) # 完播率 <= 15%, 0/1特征
raw = np.column_stack((raw, highRateFlag)) # 完播率 <= 15%, 0/1特征

# arr = raw[(raw[:, 2] <= 10) & (raw[:, 1] > 3) & (raw[:, 0] <= 90) & (raw[:, 5] <= 0.9)] # 视频时长 (5, 60]  阅读时长 > 3
# arr = raw[(raw[:, 2] <= 10) & (raw[:, 1] > 3) & (raw[:, 0] > 90) & (raw[:, 0] <= 180) & (raw[:, 5] <= 0.9)] # 视频时长 (5, 60]  阅读时长 > 3
arr = raw[(raw[:, 2] <= 10) & (raw[:, 1] > 3) & (raw[:, 0] > 180) & (raw[:, 5] <= 0.9)] # 视频时长 (5, 60]  阅读时长 > 3
arr = arr[:, [5, 0, 1, 6, 8, 9]]

# Train data gen & shuffle
print "train data size = %d" % len(arr)
sfl = np.random.permutation(len(arr))
train_x = arr[:, 1:]

train_x = train_x[sfl, :]
train_y = arr[:, 0]
train_y = train_y[sfl]

# model train
clf = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
clf.fit(train_x, train_y)
sampleLen = 50
xy = np.column_stack((train_x[0:sampleLen,[0, 1]], train_y[0:sampleLen]))
xyp = np.column_stack((xy, np.around(clf.predict(train_x[0:sampleLen]), decimals=3)))

print "total\t", "readRate\t", "percent\t", "pPercent\t", "p - y"
print np.column_stack((xyp, np.around(clf.predict(train_x[0:sampleLen]) - train_y[0:sampleLen], 3)))

print clf.score(train_x, train_y)
print "rmse", np.sqrt(np.mean((train_y - clf.predict(train_x))**2))
print clf.coef_
print clf.intercept_

print "avg = %f, len = %d" % (np.average(clf.predict(train_x)), len(train_x))

'''
 拟合效果
'''

# samplePlot = 5000
# x, y, z, p = train_x[:samplePlot, 0], train_x[:samplePlot, 1], train_y[:samplePlot], clf.predict(train_x[:samplePlot])
# ax = plt.subplot(111, projection='3d')
# ax.scatter(x, y, z, cmap=plt.get_cmap('green'))
# # ax.scatter(x, y, p, cmap='green')
# ax.set_xlabel('total')
# ax.set_ylabel('read')
# ax.set_zlabel('percentile')
# plt.show()


# while(True):
#     total = sys.stdin.readline()
#     total = float(total)
#     read = sys.stdin.readline()
#     read = float(read)
#     rate = read / total
#     lowRateFlag = 0 if rate > 0.1 else 1
#     highRateFlag = 1 if rate > 0.8 else 0
#     print total, read, read / total, clf.predict([[total, read, read / total,  lowRateFlag, highRateFlag]])

    # fig = plt.figure(figsize=(20, 8), dpi=80)
    # ax = fig.add_subplot(111)
    # data = arr[arr[:, 1] == total]
    # x = data[:, 2]
    # y = data[:, 0]
    # t1 = ax.scatter(x, y, alpha=0.8, c='r')
    # t2 = ax.scatter(x, clf.predict(data[:, 1:]), c='g')
    # plt.show()


# X = arr[:, 0]
# Y = arr[:, 1]
# X = X.reshape(-1, 1)
# Y = Y.reshape(-1, 1)
#
# X2 = np.hstack([X,X**2,X**3])
# reg = LinearRegression()
# reg.fit(X2,Y)
# y_predict = reg.predict(X2)
#
# print(reg.coef_)#x的系数和X**2的系数[0.99158261 0.51495067]
# print(reg.intercept_)#截距1.9865604994175838
#
# plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=0.3))
# plt.scatter(X, Y, c='b', alpha=0.5)
# plt.plot(X,y_predict, color = 'r')
# plt.show()





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
