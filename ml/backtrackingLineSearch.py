# !/usr/bin/python
# -*- coding: utf-8 -*-
# optimization test, y = (x-3)^2
from matplotlib.pyplot import figure, hold, plot, show, xlabel, ylabel, legend
import math


def backtracking_Armijo(alpha, beta, step, f, grad_f, d, x):
    """
    :param alpha:
    :param beta:
    :param step:
    :param f: loss function
    :param grad_f: gradient function
    :param d: descent direction
    :param x:
    :return: step

    Armijo rule: f(x - step * d(x)) <= f(x) - alpha * step * grad_f(x) * d(x), 0 < alpha <= 0.5, 0 < beta < 1
    """
    epoch = 64
    fx, dx, gx = f(x), d(x), grad_f(x)
    next = f(x - step * dx)

    # 寻找初始点, 使得next >= fx, 以满足下一循环条件
    while next < fx and epoch > 0:
        step *= 2
        next = f(x - step * dx)
        epoch -= 1

    # 寻找第一个满足Armijo准则的步长
    epoch = 64
    while next > fx - alpha * step * gx * dx and epoch > 0:
        step *= beta
        next = f(x - step * dx)
        epoch -= 1

    return step


if __name__ == '__main__':
    # 损失函数, 梯度方向, 下降优化方向
    loss_f = lambda x: (x - 2)**2 - 4
    grad_f = lambda x: 2 * x - 4
    desc_f = grad_f

    # 初始化
    init_x = 10
    x, y = init_x,loss_f(init_x)
    maxIter = 300

    # 随机梯度下降
    err, it, learning_rate = 1.0, 0, 0.2
    curve1 = [y]
    while err > 1e-4 and it < maxIter:
        step = learning_rate / math.sqrt(it + 1)
        x = x - step * grad_f(x)
        next_y = loss_f(x)
        err = abs(y - next_y)
        y = next_y

        curve1.append(y)
        it += 1

    # backtracking line search with sgd
    err, it, alpha, beta = 1.0, 0, 0.25, 0.8
    x, y = init_x,loss_f(init_x)
    curve2 = [y]
    while err > 1e-4 and it < maxIter:
        step = 1.0
        step = backtracking_Armijo(alpha, beta, step, loss_f, grad_f, desc_f, x)

        x = x - step * desc_f(x)
        next_y = loss_f(x)
        err = abs(y - next_y)
        y = next_y

        curve2.append(y)
        it += 1

    figure()
    plot(curve1, 'r*-')
    plot(curve2, 'bo-')
    xlabel("iteration")
    ylabel("loss")

    legend(['normal sgd', 'backtracking line search'])
    show()