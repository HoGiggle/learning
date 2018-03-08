#!/usr/bin/python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
import tensorflow as tf

# x ** x        x > 0
# (-x) ** (-x)  x < 0

def f(x):
    y = np.ones_like(x)
    i = x > 0
    y[i] = np.power(x[i], x[i])
    i = x < 0
    y[i] = np.power(-x[i], -x[i])
    return y

class A:
    pass


class B(A):
    pass

def func1(list):
    list.extend([1, 2])
    return list


if __name__ == '__main__':
    # # 1 numpy
    # x = np.linspace(0.0001, 1.3, 101)
    # y = f(x)
    # plt.plot(x, y, '-g', label='x^x', linewidth=2)
    # plt.grid()
    # plt.legend(loc='upper right')
    # plt.show()

    # 2 test
    # print random.randrange(1, 10, 2)
    #
    # print math.sin(math.pi / 6)
    # print math.degrees(math.asin(0.5))
    # print math.hypot(3, 4)
    #
    # print math.degrees(math.pi)
    # print math.radians(90 / math.pi)
    #
    # print ord(u'\u0061')
    # print ord('a')
    # print chr(101)
    # print unichr(200)
    # print hex(ord('a'))
    # print oct(ord('a'))
    # print bin(ord('a'))
    # print list(("hello", "world"))
    #
    # a = "lydia"
    # b = "lydia"
    # print a is b
    #
    # print range(0, 10, 1)[::-1]

    # a = "abc"
    # print a.count('y')
    # print a.center(10, '*')
    # print a.center(11, '*')
    #
    # s = '中文'
    # print s, type(s)
    # print s.decode('utf-8').encode('gbk').decode('gbk').encode('utf-8')


    list = [3, 4, 5]
    print(list[::-1])






