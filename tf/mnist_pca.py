# !/usr/bin/python
# coding:utf8

import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from PIL import Image


def array_to_img(array):
    array = array * 255
    new_img = Image.fromarray(array.astype(np.uint8))
    return new_img


def comb_imgs(origin_imgs, col, row, each_width, each_height, new_type):
    new_img = Image.new(new_type, (col * each_width, row * each_height))
    for i in range(len(origin_imgs)):
        each_img = array_to_img(np.array(origin_imgs[i]).reshape(each_width, each_width))
        # 第二个参数为每次粘贴起始点的横纵坐标。在本例中，分别为（0，0）（28，0）（28*2，0）依次类推，第二行是（0，28）（28，28），（28*2，28）类推
        new_img.paste(each_img, ((i % col) * each_width, (i / col) * each_width))
    return new_img


def pca(data_mat, top_n_feat=99999999):
    """
    主成分分析：
    输入：矩阵data_mat ，其中该矩阵中存储训练数据，每一行为一条训练数据
         保留前n个特征top_n_feat，默认全保留
    返回：降维后的数据集和原始数据被重构后的矩阵（即降维后反变换回矩阵）
    """

    # 获取数据条数和每条的维数
    num_data, dim = data_mat.shape
    print(num_data)  # 100
    print(dim)  # 784

    # 数据中心化，即指变量减去它的均值
    mean_vals = data_mat.mean(axis=0)  # shape:(784,)
    mean_removed = data_mat - mean_vals  # shape:(100, 784)

    # 计算协方差矩阵（Find covariance matrix）
    cov_mat = np.cov(mean_removed, rowvar=False)  # shape：(784, 784)

    # 计算特征值(Find eigenvalues and eigenvectors)
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))  # 计算特征值和特征向量，shape分别为（784，）和(784, 784)

    eig_val_index = np.argsort(eig_vals)  # 对特征值进行从小到大排序，argsort返回的是索引，即下标

    eig_val_index = eig_val_index[:-(top_n_feat + 1): -1]  # 最大的前top_n_feat个特征的索引
    # 取前top_n_feat个特征后重构的特征向量矩阵reorganize eig vects,
    # shape为(784, top_n_feat)，top_n_feat最大为特征总数
    reg_eig_vects = eig_vects[:, eig_val_index]

    # 将数据转到新空间
    low_d_data_mat = mean_removed * reg_eig_vects  # shape: (100, top_n_feat), top_n_feat最大为特征总数
    recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_vals  # 根据前几个特征向量重构回去的矩阵，shape:(100, 784)

    return low_d_data_mat, recon_mat


mnist = input_data.read_data_sets("/Users/giggle/Work/data/mnist/", one_hot=False)
imgs = mnist.train.images
labels = mnist.train.labels

# 选择100张7的图片，并显示原始特征的图片
origin_7_imgs = []
for i in range(1000):
    # if labels[i] == 9 and len(origin_7_imgs) < 100:
    if len(origin_7_imgs) < 100:
        origin_7_imgs.append(imgs[i])

ten_origin_7_imgs = comb_imgs(origin_7_imgs, 10, 10, 28, 28, 'L')
ten_origin_7_imgs.show()

# 显示pca降维后特征图片
low_d_feat_for_7_imgs, recon_mat_for_7_imgs = pca(np.array(origin_7_imgs), 1)  # 只取最重要的1个特征
print(type(recon_mat_for_7_imgs))
print low_d_feat_for_7_imgs[0]
print low_d_feat_for_7_imgs[1]
print low_d_feat_for_7_imgs[2]
low_d_img = comb_imgs(recon_mat_for_7_imgs, 10, 10, 28, 28, 'L')
low_d_img.show()
