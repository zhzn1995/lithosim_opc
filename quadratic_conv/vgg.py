
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def conv2d_q(inputs, num_outputs, kernel_size, stride=1, rate=1, padding='SAME', scope=None):
    qw1 = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                        rate=rate, padding=padding,
                        weights_initializer=trunc_normal(0.01),
                        biases_initializer=None,
                        activation_fn=None,
                        normalizer_fn=None,
                        normalizer_params=None,
                        scope='qw1'
                        )
    qw2 = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                        rate=rate, padding=padding,
                        weights_initializer=trunc_normal(0.01),
                        biases_initializer=None,
                        activation_fn=None,
                        normalizer_fn=None,
                        normalizer_params=None,
                        scope='qw2'
                        )
    qw3 = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                        rate=rate, padding=padding,
                        weights_initializer=trunc_normal(0.01),
                        biases_initializer=None,
                        activation_fn=None,
                        normalizer_fn=None,
                        normalizer_params=None,
                        scope='qw3'
                        )
    qw4 = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                        rate=rate, padding=padding,
                        weights_initializer=trunc_normal(0.01),
                        biases_initializer=None,
                        activation_fn=None,
                        normalizer_fn=None,
                        normalizer_params=None,
                        scope='qw4'
                        )
    conv = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       rate=rate, padding=padding, scope=scope,
                       activation_fn=None,
                       normalizer_fn=None,
                       normalizer_params=None,
                       biases_initializer=None,
                       )
    #tmp = tf.add(tf.multiply(qw2,qw4), tf.multiply(qw1,qw3))
    # conv = tf.multiply(qw1, qw2) + conv
    conv = tf.multiply(qw1, qw2) + tf.multiply(qw3, qw4) + conv
    #conv = tf.multiply(qw3,conv)
    #bn = slim.batch_norm(conv, scope=scope+'/BatchNorm')
    relu = tf.nn.relu(conv)
    return relu

def vgg_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID'):
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, conv2d_q, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, conv2d_q, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, conv2d_q, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, conv2d_q, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, conv2d_q, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = conv2d_q(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = conv2d_q(net, 4096, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      net = conv2d_q(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_16.default_image_size = 224

# Alias
vgg_d = vgg_16
