"""
Related papers:
High-Resolution Representations for Labeling Pixels and Regions.higharXiv:1904.04514v1 [cs.CV] 9 Apr 2019
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.training import moving_averages

import tensorflow as tf


def conv(x, kernel_size, out_planes, name, stride = 1):
    """convolution with padding"""
    x_shape = x.get_shape().as_list()
    w = tf.get_variable(name=name+'b', shape=[kernel_size, kernel_size, x_shape[3], out_planes])

    return tf.nn.conv2d(input=x, filter=w, padding='SAME', strides=[1, stride, stride, 1], name=name)


def batch_norm(x, training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))
    beta = tf.get_variable(name='beta', shape=params_shape, initializer=tf.zeros_initializer)
    gamma = tf.get_variable(name='gamma', shape=params_shape, initializer=tf.ones_initializer)

    moving_mean = tf.get_variable(name='moving_mean',
                                  shape=params_shape,
                                  initializer=tf.zeros_initializer,
                                  trainable=False)

    moving_variance = tf.get_variable(name='moving_variance',
                                      shape=params_shape,
                                      initializer=tf.ones_initializer,
                                      trainable=False)

    tf.add_to_collection('BN_MEAN_VARIANCE', moving_mean)
    tf.add_to_collection('BN_MEAN_VARIANCE', moving_variance)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean,
                                                               0.99,
                                                               name='MovingAvgMean')
    update_moving_variance = moving_averages.assign_moving_average(moving_variance,
                                                                   variance,
                                                                   0.99,
                                                                   name='MovingAvgVariance')

    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

    mean, variance = tf.cond(
        pred = training,
        true_fn = lambda: (mean, variance),
        false_fn = lambda: (moving_mean, moving_variance)
    )
    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-3)

    return x

def relu(x):
    return tf.nn.ReLU(x)
    

class Bottleneck():
    expansion = 4

    def __init__(self, x, is_training, block_name, outplanes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.block_name = block_name
        self.is_training = is_training
        self.conv1 = conv(x, 1, outplanes, name=self.block_name + 'conv1', stride=stride)
        self.bn1 = batch_norm(outplanes, is_training)
        self.conv2 = conv(outplanes, 3, outplanes, name=self.block_name + 'conv2', stride=stride)
        self.bn2 = batch_norm(outplanes, is_training)
        self.conv3 = conv(outplanes, 1, outplanes * self.expansion, name=self.block_name + 'conv3', stride=stride)
        self.bn3 = batch_norm(outplanes * self.expansion, is_training)
        self.relu = tf.nn.relu(outplanes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        with tf.variable_scope(self.block_name + '11_1'):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

        with tf.variable_scope(self.block_name + '33_2'):
            out = self.conv2(x)
            out = self.bn2(out)
            out = self.relu(out)

        with tf.variable_scope(self.block_name + '11_3'):
            out = self.conv3(x)
            out = self.bn3(out)

        if self.downsample is not None:
            with tf.variable_scope(self.block_name + 'downsample'):
                residual = self.downsample(x)
                
        out = out + residual
        out = tf.nn.relu(out)
        
        return out


class BasicBlock():
    expansion = 1
     
    def __init__(self, x, is_training, block_name, outplanes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.block_name = block_name
        self.is_training = is_training
        self.conv1 = conv(x, 3, outplanes, name=self.block_name + 'conv1', stride=stride)
        self.bn1 = batch_norm(outplanes, is_training)
        self.conv2 = conv(outplanes, 3, outplanes, name=self.block_name + 'conv2', stride=stride)
        self.bn2 = batch_norm(outplanes, is_training)
        self.relu = tf.nn.relu(outplanes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        with tf.variable_scope(self.block_name + '33_1'):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

        with tf.variable_scope(self.block_name + '33_2'):
            out = self.conv2(x)
            out = self.bn2(out)

        if self.downsample is not None:
            with tf.variable_scope(self.block_name + 'downsample'):
                residual = self.downsample(x)
                residual = batch_norm(residual, self.is_training)

        out += residual
        out = self.relu(out)
        
        return out







