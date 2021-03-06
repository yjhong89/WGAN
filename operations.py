import numpy as np
import tensorflow as tf

def batch_wrapper(x, is_training, name='batch_norm', decay=0.99):
    with tf.variable_scope(name):
        # Population variables are declared as tainable False, since they will be updated by ourselves
        pop_mean = tf.get_variable('pop_mean', [x.get_shape()[-1]], initializer=tf.constant_initializer(0), trainable=False)
        pop_variance = tf.get_variable('pop_variance', [x.get_shape()[-1]], initializer=tf.constant_initializer(1), trainable=False)
        scale = tf.get_variable('scale', [x.get_shape()[-1]], initializer=tf.constant_initializer(1))
        beta = tf.get_variable('beta', [x.get_shape()[-1]], initializer=tf.constant_initializer(0))

        if is_training:
            batch_mean, batch_variance = tf.nn.moments(x, [0,1,2])
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean *  (1-decay))
            train_variance = tf.assign(pop_variance, pop_variance * decay + batch_variance*(1-decay))
            with tf.control_dependencies([train_mean, train_variance]):
                return tf.nn.batch_normalization(x, batch_mean, batch_variance, offset=beta, scale=scale, variance_epsilon=1e-5)
        else:
            return tf.nn.batch_normalization(x, pop_mean, pop_variance, beta, scale, 1e-5)

def conv2d(x, output_dim, filter_height=5, filter_width=5, stride_hor=2, stride_ver=2, name='conv2d'):
    with tf.variable_scope(name):
        filter = tf.get_variable('filter', [filter_height, filter_width, x.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        convolution = tf.nn.conv2d(x, filter, strides=[1,stride_hor, stride_ver, 1], padding='SAME')
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0))
        weighted_sum = convolution + bias
        return weighted_sum

def deconv2d(x, output_shape, filter_height=5, filter_width=5, stride_hor=2, stride_ver=2, name='deconv2d'):
    with tf.variable_scope(name):
        filter = tf.get_variable('filter', [filter_height, filter_width, output_shape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
        deconvolution = tf.nn.conv2d_transpose(x, filter, output_shape=output_shape, strides=[1, stride_hor, stride_ver, 1])
        bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0))
        weighted_sum = deconvolution + bias
        return weighted_sum


def linear(x, hidden, name='linear'): # x : [batch, hi]
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [x. get_shape()[-1], hidden], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', [hidden], initializer=tf.constant_initializer(0))
        weighted_sum = tf.matmul(x, weight) + bias
        return weighted_sum


def leaky_relu_with_batch(x, is_training, leak=0.2, name='lrelu_batch'):
    with tf.variable_scope(name):
        before_nonlinear = batch_wrapper(x, is_training)
        lrelu = tf.maximum(before_nonlinear, before_nonlinear*leak)
        return lrelu

def leaky_relu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        lrelu = tf.maximum(x, x*leak)
        return lrelu

def relu_with_batch(x, is_training, name='relu_batch'):
    with tf.variable_scope(name):
        before_nonlinear = batch_wrapper(x, is_training)
        relu = tf.maximum(before_nonlinear, 0)
        return relu

def relu(x, name='relu'):
    with tf.variable_scope(name):
        relu = tf.maximum(x, 0)
        return relu


if __name__ == "__main__":
    a = tf.get_variable('a', [1, 8, 8, 256])
    b = conv2d(a, 128)
    print(b.get_shape())
