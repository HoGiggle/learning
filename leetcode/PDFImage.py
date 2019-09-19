# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, factorial as fac
from enum import Enum
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class Params:
    def __init__(self, p=0.0, n=0, r=0, lambda_0=0.0, alpha=0.0, beta=0.0, mu=0.0, b=0.0):
        self.p = p
        self.n = n
        self.r = r
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.b = b

class PDF(Enum):
    Binomial = 0
    Geometric = 1
    NegBinomial = 2
    Poisson = 3
    Exponential = 4
    Gamma = 5
    Beta = 6
    Laplace = 7


def pdf(name):
    """
    :type name: PDF
    :return:
    """
    switch = {
        PDF.Binomial: lambda pm, k: comb(pm.n, k) * np.power(pm.p, k) * np.power(1-pm.p, pm.n-k),
        PDF.Geometric: lambda pm, k: pm.p * np.power(1 - pm.p, k - 1),
        PDF.NegBinomial: lambda pm, k: comb(k-1, pm.r) * np.power(pm.p, pm.r-1) * np.power(1-pm.p, k-pm.r),
        PDF.Poisson: lambda pm, k: np.exp(-pm.lambda_0) * np.power(pm.lambda_0, k) / fac(k),
        PDF.Exponential: lambda pm, x: pm.lambda_0 * np.exp(-pm.lambda_0 * x),
        PDF.Gamma: lambda pm, x: np.power(x, pm.alpha-1) * np.power(pm.lambda_0, pm.alpha) * np.exp(-pm.lambda_0*x),
        PDF.Beta: lambda pm, x: (fac(pm.alpha + pm.beta) * 1.0 / (fac(pm.alpha) * fac(pm.beta) * 1.0)) * np.power(x, pm.alpha - 1) * np.power(1-x, pm.beta-1),
        PDF.Laplace: lambda pm, x: (1.0 / (2 * pm.b)) * np.exp(-np.abs(x - pm.mu) / pm.b)
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

    color_map = {0:'r', 1:'b', 2:'y', 3:'g', 4:'black'}
    for idx, yi in enumerate(y):
        plt.plot(x, yi, color=color_map[idx], label=labels[idx])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # x = np.arange(1, 50)
    x = np.arange(-10, 10.01, 0.01)
    # 几何分布
    # helper(x, [Params(p=0.9), Params(p=0.5), Params(p=0.25)], PDF.Geometric, ['p=0.9', 'p=0.5', 'p=0.25'], "Geometric distribution", 'X', 'P(X=K)')

    # 二项分布
    # helper(x, [Params(p=0.8, n=50), Params(p=0.5, n=50), Params(p=0.2, n=50)], PDF.Binomial, ['p=0.8 n=50', 'p=0.5 n=50', 'p=0.2 n=50'], "Binomial distribution", 'X', 'P(X=K)')

    # 负二项分布
    # helper(x, [Params(p=0.4, r=5), Params(p=0.4, r=10), Params(p=0.8, r=10)], PDF.NegBinomial, ['p=0.4 r=5', 'p=0.4 r=10', 'p=0.8 r=10'], "Negative Binomial distribution", 'X', 'P(X=K)')

    # 泊松分布
    # helper(x, [Params(lambda_0=1), Params(lambda_0=5), Params(lambda_0=10)], PDF.Poisson, ['λ=1', 'λ=5', 'λ=10'], "Poisson distribution", 'X', 'P(X=K)')

    # 指数分布
    # helper(x, [Params(lambda_0=0.5), Params(lambda_0=1.0), Params(lambda_0=2)], PDF.Exponential, ['λ=0.5', 'λ=1.0', 'λ=2.0'], "Poisson distribution", 'X', 'P(X=K)')

    # Gamma分布
    # helper(x, [Params(lambda_0=0.5, alpha=1), Params(lambda_0=0.5, alpha=2), Params(lambda_0=0.5, alpha=3), Params(lambda_0=1.0, alpha=3)], PDF.Gamma,
    #        ['λ=0.5 α=1', 'λ=0.5 α=2', 'λ=0.5 α=3', 'λ=1.0 α=3'], "Gamma distribution", 'X', 'P(X=K)')

    # Beta分布
    # helper(x, [Params(alpha=1, beta=1), Params(alpha=1, beta=5), Params(alpha=5, beta=1), Params(alpha=5, beta=2), Params(alpha=2, beta=5)], PDF.Beta,
    #        ['α=1 β=1', 'α=1 β=5', 'α=5 β=1', 'α=5 β=2', 'α=2 β=5'], "Beta distribution", 'X', 'P(X=K)')

    # Laplace分布
    helper(x, [Params(mu=0, b=1), Params(mu=0, b=2), Params(mu=0, b=4), Params(mu=-5, b=4)], PDF.Laplace,
           ['μ=0 b=1', 'μ=0 b=2', 'μ=0 b=4', 'μ=-5 b=4'], "Laplace distribution", 'X', 'P(X=K)')



