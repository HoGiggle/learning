# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys


# 自定义函数 e指数形式
def func(x, a, b, c):
    return (a / (x + b)) + c


# 定义x、y散点坐标
# raw = np.loadtxt('/Users/hujinjun/work/data/readRateDis/doc_dis.txt', delimiter='~')
videoData = np.loadtxt('/Users/hujinjun/work/data/readRate/videoRate.txt', delimiter=' ')
docData = np.loadtxt('/Users/hujinjun/work/data/readRate/docRate.txt', delimiter='\t')
atlasData = np.loadtxt('/Users/hujinjun/work/data/readRate/atlasRate.txt', delimiter=' ')

durList = []
aList = []
bList = []
cList = []

# for i in np.arange(12, 301, 1):
#     data = raw[raw[:, 0] == i]
#     x = data[:, 1]
#     y = data[:, 5]
#
#     # 非线性最小二乘法拟合
#     popt, pcov = curve_fit(func, x, y)
#     # 获取popt里面是拟合系数
#     a = popt[0]
#     b = popt[1]
#     c = popt[2]
#     durList.append(i)
#     aList.append(a)
#     bList.append(b)
#     cList.append(c)
#     yvals = func(x, a, b, c)  # 拟合y值
#     print u'total:', i
#     print u'系数a:', a
#     print u'系数b:', b
#     print u'系数c:', c
#
#     # 绘图
#     plot1 = plt.plot(x, y, 's', label='original values')
#     plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend(loc=4)  # 指定legend的位置右下角
#     plt.title('curve_fit')
#     plt.show()


# 非线性最小二乘法拟合
# raw = docData
# x = raw[:, 0]
# y = raw[:, 1]

x = videoData[:, 0]
y = np.arange(len(x)).astype(np.float64)

print x.shape, y.shape
docLen = len(docData)
atlasLen = len(atlasData)
for i in np.arange(len(x)):
    sumV = videoData[i, 1]
    count = 1
    if i < atlasLen:
        sumV += atlasData[i, 1]
        count += 1
    if i < docLen:
        sumV += docData[i, 1]
        count += 1
    y[i] = sumV / (count * 1.0)
popt, pcov = curve_fit(func, x, y)
# 获取popt里面是拟合系数
a = popt[0]
b = popt[1]
c = popt[2]

xx = videoData[:, 0]
yvals = func(xx, a, b, c)  # 拟合y值
print u'系数a:', a
print u'系数b:', b
print u'系数c:', c

# 绘图
plot1 = plt.plot(x, y, '^', c='r', label='avg')
plt.plot(docData[:, 0], docData[:, 1], 's', c='g', label='doc')
plt.plot(videoData[:, 0], videoData[:, 1], 'c', label='video')
plt.plot(atlasData[:, 0], atlasData[:, 1], 'y', label='atlas')
plot2 = plt.plot(xx, yvals, 'b', label='polyfit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)  # 指定legend的位置右下角
plt.title('curve_fit')
plt.show()