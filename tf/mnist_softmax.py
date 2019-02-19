#!/usr/bin/python
# coding:utf8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("/Users/giggle/Work/data/mnist/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])  # 构建占位符，代表输入的图像，None表示样本的数量可以是任意的
W = tf.Variable(tf.zeros([784, 10]))  # 构建一个变量，代表训练目标W，初始化为0
b = tf.Variable(tf.zeros([10]))  # 构建一个变量，代表训练目标b，初始化为0

y = tf.nn.softmax(tf.matmul(x, W) + b)  # 构建了一个softmax的模型：y = softmax(Wx + b)，y指样本标签的预测值
y_ = tf.placeholder("float", [None, 10])  # 构建占位符，代表样本标签的真实值

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 交叉熵代价函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # 使用梯度下降法（0.01的学习率）来最小化这个交叉熵代价函数

init = tf.initialize_all_variables()
sess = tf.Session()  # 构建会话
sess.run(init)  # 初始化所有变量
for i in range(1000):  # 迭代次数为1000
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 使用minibatch的训练数据，一个batch的大小为100
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # 用训练数据替代占位符来执行训练
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 用平均值来统计测试准确率
    if i % 50 == 0:
        print i, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})  # 打印测试信息

sess.close()
