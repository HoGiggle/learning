# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from enum import Enum

class Params:
    def __init__(self, p=0.0, n=0):
        self.p = p
        self.n = n

class PDF(Enum):
    Binomial = 0
    Geometric = 1


def pdf(name):
    """
    :type name: PDF
    :return:
    """
    switch = {
        PDF.Binomial: lambda pm, k: comb(pm.n, k) * np.power(pm.p, k) * np.power(1-pm.p, pm.n-k),
        PDF.Geometric: lambda pm, k: pm.p * np.power(1 - pm.p, k - 1)
    }
    return switch[name]


def helper(x, params_list, pdf_name, labels, title, x_label, y_label):
    """
    :param x:
    :type params_list: List[Params]
    :type labels: List[str]
    :return:
    """
    y = []
    for param in params_list:
        y.append(map(lambda i: pdf(pdf_name)(param, i), x))

    color_map = {0:'r', 1:'b', 2:'y', 3:'g'}
    for idx, yi in enumerate(y):
        plt.plot(x, yi, color=color_map[idx], label=labels[idx])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


x = range(1, 20)
helper(x, [Params(p=0.9), Params(p=0.5), Params(p=0.25)], PDF.Geometric,
       ['p=0.9', 'p=0.5', 'p=0.25'], "Geometric distribution", 'X', 'P(X=K)')

helper(x, [Params(p=0.9, n=20), Params(p=0.5, n=20), Params(p=0.25, n=20)], PDF.Binomial,
       ['p=0.9 n=20', 'p=0.5 n=20', 'p=0.25 n=20'], "Binomial distribution", 'X', 'P(X=K)')


