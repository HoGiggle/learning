# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

percentile = 0.8
# raw = np.loadtxt('/Users/hujinjun/Desktop/v_%s' % percentile, delimiter=',')
raw = np.loadtxt('/Users/hujinjun/Desktop/doc_sample', delimiter=',')
print type(raw)
np.random.shuffle(raw)
print raw[0]
sample = int(len(raw) * 0.1)
raw = raw[:sample]

x = raw[:, 0]
y = raw[:, 1]
z = raw[:, 2]

fig = plt.figure(figsize=(20, 8), dpi=80)
plt.scatter(x, y, c='b')

plt.xlabel('percentile', fontsize=18)
plt.ylabel('r_score', fontsize=18)
plt.title('Doc percentile sample', fontsize=18, fontweight='bold')
fig.show()



# raw = np.loadtxt('/Users/hujinjun/work/data/readRateDis/doc_dis.txt', delimiter='~')
# total = sys.stdin.readline()
# total = int(total)
# colorMap = {0:'r', 1:'g', 2:'b', 3:'y', 4:'pink'}
# axList = []
# argList = []
#
# fig = plt.figure(figsize=(20, 8), dpi=80)
# ax = fig.add_subplot(111)
#
# for i in range(5):
#     data = raw[raw[:, 0] == total]
#     x = data[:, 1]
#     y = data[:, 5]
#     t = ax.scatter(x, y, alpha=0.5, c=colorMap[i])
#     axList.append(t)
#     argList.append(total)
#     # ax.legend((t), (total), loc=0)
#     total = total + 5
# ax.legend(axList, argList, loc=0)


# data = raw[raw[:, 0] == 12]
# x = data[:, 1]
# y = data[:, 5]
# t1 = ax.scatter(x, y, alpha=0.5)
#
#
# data = raw[raw[:, 0] == 20]
# x = data[:, 1]
# y = data[:, 5]
# t2 = ax.scatter(x, y, alpha=0.5, c='r')
#
# ax.legend((t1, t2), ("12", "20"), loc=0)

plt.show()
