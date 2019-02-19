# !/usr/bin/python
# coding:utf8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("/Users/giggle/Work/data/mnist/", one_hot=True)
# x = tf.placeholder(tf.float32, [None, 784])
# w = tf.Variable()


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn():
    # type: () -> object
    # 输入向量
    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层卷积 28*28*1 -> 28*28*32 -> 14*14*32
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积  14*14*32 -> 14*14*64 -> 7*7*64
    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接
    w_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    with tf.name_scope('full_conn'):
        with tf.name_scope('weights'):
            variable_summaries(w_fc1)
        with tf.name_scope('biases'):
            variable_summaries(b_fc1)
        # with tf.name_scope('wx_plus_b'):
        #     tf.summary.histogram(h_fc1)

    # drop
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    with tf.name_scope('dropout'):
        tf.summary.scalar('dropout_keep_prob', keep_prob)

    # output
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    # ********* 训练 **********
    mnist = input_data.read_data_sets("/Users/giggle/Work/data/mnist/", one_hot=True)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_pre = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

    # tensor board create
    tf.summary.scalar("loss", cross_entropy)
    tf.summary.scalar("acc", acc)
    merged_summary_op = tf.summary.merge_all()

    # session init
    init = tf.initialize_all_variables()
    sess = tf.Session()
    test_writer = tf.summary.FileWriter('/Users/giggle/Work/PycharmProjects/learning/logs/mnist_nn_test', sess.graph)
    train_writer = tf.summary.FileWriter('/Users/giggle/Work/PycharmProjects/learning/logs/mnist_nn_train', sess.graph)
    sess.run(init)
    for i in range(500):
        batch_x, batch_y = mnist.train.next_batch(100)
        if i % 50 == 0:
            test_acc, summary_str = sess.run([acc, merged_summary_op], feed_dict={x: mnist.validation.images,
                                                                                  y_: mnist.validation.labels,
                                                                                  keep_prob: 1.0})
            test_writer.add_summary(summary_str, i)
            print "step %d, training accuracy %g" % (i, test_acc)
        _, summary_str = sess.run([train_step, merged_summary_op], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
        train_writer.add_summary(summary_str, i)

    print "test accuracy %g" % sess.run(acc, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


if __name__ == '__main__':
    nn()
