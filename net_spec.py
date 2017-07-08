# Copyright (c) 2017 Artsiom Sanakoyeu
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tflearn.layers.conv import upscore_layer

from config import imagenet_snapshots_map
from data_utils import LionClasses
from data_utils import CLASS_NAMES
from net_config import (regression_l2_loss, segmentation_loss,
                        regression_sum_heatmap_loss, l2_heatmap_loss)

from nets import vgg
from nets import resnet_v2
from nets import nets_factory
import preprocessing.vgg_preprocessing as vgg_preprocessing


def vgg_16(inputs,
           num_classes=6,
           is_training=True,
           dropout_keep_prob=0.5,
           scope='vgg_16',
           fc_conv_padding='VALID',
           use_segmentation=True,
           num_layers_to_fix=4):
    """Oxford Net VGG 16-Layers version D Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
      fc_conv_padding: the type of padding to use for the fully connected layer
        that is implemented as a convolutional layer. Use 'SAME' padding if you
        are applying the network in a fully convolutional manner and want to
        get a prediction map downsampled by a factor of 32 as an output. Otherwise,
        the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.

    Returns:
      the last op containing the log predictions and end_points dict.
    """
    print 'Fixing {} layers'.format(num_layers_to_fix)
    assert num_layers_to_fix in [0, 2, 4, 6, 9, 12, 13, 14]
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3],
                              trainable=num_layers_to_fix < 2, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                              trainable=num_layers_to_fix < 4, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3],
                              trainable=num_layers_to_fix < 6, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                              trainable=num_layers_to_fix < 9, scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                              trainable=num_layers_to_fix < 12, scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding,
                              trainable=num_layers_to_fix < 13, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], trainable=num_layers_to_fix < 14, scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout7')
            if use_segmentation:
                net = slim.conv2d(net, num_classes, [1, 1], scope='heatmaps_small')

                # input_shape = tf.shape(inputs)
                # out_shape = tf.stack([input_shape[0], input_shape[1] / 4, input_shape[2] / 4])
                # net = upscore_layer(net, num_classes,
                #                    shape=out_shape,
                #                    kernel_size=16,
                #                    strides=[1, 8, 8, 1],
                #                    trainable=False,
                #                    name='upscore4')
                assert_ops = [tf.verify_tensor_all_finite(net, msg='heatmaps nan')]
                with tf.control_dependencies(assert_ops):
                    net = upscore_layer(net, num_classes,
                                        shape=tf.shape(inputs), kernel_size=64,
                                        strides=[1, 32, 32, 1],
                                        trainable=True,
                                        name='upscore32')

            slim.utils.collect_named_outputs(end_points_collection,
                                             sc.original_name_scope, net)
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points


class Vgg16:
    def __init__(self, input_images, input_heatmaps,
                 relu_heatmaps=False,
                 heatmap_in_loss_multiplier=1.0,
                 class_weights=None,
                 use_regression_loss=False,
                 regression_loss_weight=0.1,
                 reuse=False,
                 is_segmentation=False):
        assert relu_heatmaps is False
        self.is_training_pl = tf.placeholder(tf.bool)
        self.input_images = input_images
        self.input_heatmaps = input_heatmaps
        self.heatmap_in_loss_multiplier = heatmap_in_loss_multiplier
        tf.summary.scalar('heatmap_in_loss_mult', heatmap_in_loss_multiplier)
        if use_regression_loss:
            tf.summary.scalar('other/regression_loss_weight', regression_loss_weight)

        print 'Create Net. Use ReLU after heatmaps={}'.format(relu_heatmaps)

        net_name = 'vgg_16'
        weight_decay = 0.0001
        num_classes = 5 if not is_segmentation else 6
        with tf.device('/gpu:0'):
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                self.global_iter_counter = tf.get_variable('global_iter_counter', shape=[],
                                                           initializer=tf.constant_initializer(
                                                               0),
                                                           trainable=False)
                self.streams_outputs = list()
                self.streams_end_points = list()

                arg_scope = vgg.vgg_arg_scope(weight_decay=weight_decay)
                with slim.arg_scope(arg_scope):
                    self.heatmaps, self.end_points = vgg_16(vgg_preproc(self.input_images),
                                                            num_classes=num_classes,
                                                            is_training=self.is_training_pl,
                                                            dropout_keep_prob=0.5,
                                                            scope='vgg_16',
                                                            fc_conv_padding='SAME')

                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    weights_regularizer=slim.l2_regularizer(weight_decay),
                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    biases_initializer=tf.zeros_initializer(),
                                    padding='SAME',
                                    stride=1):

                    assert_ops = [
                        tf.verify_tensor_all_finite(self.heatmaps, msg='heatmaps big')]
                    with tf.control_dependencies(assert_ops):
                        if is_segmentation:
                            self.loss = segmentation_loss(self.heatmaps, self.input_heatmaps)
                        else:
                            self.loss = l2_heatmap_loss(self.heatmaps, self.input_heatmaps,
                                                        heatmap_multiplier=self.heatmap_in_loss_multiplier,
                                                        class_weights=class_weights)
                    if use_regression_loss:
                        self.regression_loss = regression_loss_weight * regression_sum_heatmap_loss(
                            self.heatmaps, self.input_heatmaps,
                            heatmap_multiplier=self.heatmap_in_loss_multiplier)
                        tf.losses.add_loss(self.regression_loss)
                    self.total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

        tf.summary.scalar('{}/sum/heatmaps'.format(net_name), tf.reduce_sum(self.heatmaps))
        tf.summary.scalar('{}/sparsity/heatmaps'.format(net_name),
                          tf.nn.zero_fraction(self.heatmaps))
        if not is_segmentation:
            heatmaps_per_class = tf.split(self.heatmaps, LionClasses.NUM_CLASSES, axis=3)
            for class_id, heatmap in enumerate(heatmaps_per_class):
                tf.summary.scalar('{}/sum/heatmaps/{}'.format(net_name, CLASS_NAMES[class_id]),
                                  tf.reduce_sum(heatmap))

        tf.summary.scalar('{}/{}'.format(net_name, self.loss.op.name.split('/')[-1]),
                          self.loss)
        if use_regression_loss:
            tf.summary.scalar(
                '{}/{}'.format(net_name, self.regression_loss.op.name.split('/')[-1]),
                self.regression_loss)
        tf.summary.scalar('{}/{}'.format(net_name, self.total_loss.op.name.split('/')[-1]),
                          self.total_loss)
        self.graph = tf.get_default_graph()
        self.sess = None


class SegPlusRegNet:
    def __init__(self, input_images, input_heatmaps,
                 input_gt_counts,
                 images_preproc_fn=None,
                 input_image_ids=None,
                 is_training_pl=None,
                 relu_heatmaps=False,
                 class_weights=None,
                 regression_loss_weight=1.0,
                 segmentation_loss_weight=1.0,
                 use_segmentation_loss=True,
                 use_regression_loss=True,
                 weight_decay=0.0001,
                 num_layers_to_fix=4,
                 extra_fc_size=0,
                 net_name='vgg_16',
                 should_create_summaries=True,
                 global_pool=False,
                 ):
        # TODO implement switch between train/val inputs
        assert relu_heatmaps is False
        assert regression_loss_weight == 1.0, 'Not implemented yet'
        self.use_segmentation_loss = use_segmentation_loss
        self.use_regression_loss = use_regression_loss
        self.class_weights = class_weights
        self.extra_fc_size = extra_fc_size
        self.global_pool = global_pool
        if is_training_pl is None:
            self.is_training_pl = tf.placeholder(tf.bool)
        else:
            self.is_training_pl = is_training_pl
        self.input_images_pl = input_images
        if images_preproc_fn is not None:
            input_images_preprocessed = images_preproc_fn(self.input_images_pl)
        else:
            input_images_preprocessed = self.input_images_pl
        self.input_heatmaps = input_heatmaps
        self.input_gt_counts = input_gt_counts
        self.input_image_ids = input_image_ids
        if use_regression_loss:
            tf.summary.scalar('other/regression_loss_weight', regression_loss_weight)
        if use_segmentation_loss:
            tf.summary.scalar('other/segmentation_loss_weight', segmentation_loss_weight)
        if use_segmentation_loss:
            tf.summary.scalar('other/num_layers_to_fix', num_layers_to_fix)
        tf.summary.scalar('other/extra_fc_size', extra_fc_size)

        print 'Create Net. Use ReLU after heatmaps={}'.format(relu_heatmaps)

        self.net_name = net_name
        num_classes = 6
        with tf.device('/gpu:0'):
            with tf.variable_scope(tf.get_variable_scope()):
                self.global_iter_counter = tf.get_variable('global_iter_counter', shape=[],
                                                           dtype=tf.int32,
                                                           initializer=tf.constant_initializer(
                                                               0),
                                                           trainable=False)
                self.streams_outputs = list()
                self.streams_end_points = list()

                arg_scope = nets_factory.arg_scopes_map[net_name](weight_decay=weight_decay)
                with slim.arg_scope(arg_scope):
                    self.heatmaps = None
                    if net_name == 'vgg_16':
                        self.heatmaps, self.end_points = vgg_16(input_images_preprocessed,
                                                                num_classes=num_classes,
                                                                is_training=self.is_training_pl,
                                                                dropout_keep_prob=0.5,
                                                                scope='vgg_16',
                                                                fc_conv_padding='SAME'
                                                                if use_segmentation_loss
                                                                else 'VALID',
                                                                use_segmentation=
                                                                use_segmentation_loss,
                                                                num_layers_to_fix=num_layers_to_fix)
                    elif net_name == 'vgg_19':
                        _, self.end_points = nets_factory.networks_map['vgg_19'](
                            input_images_preprocessed, num_classes=None,
                            is_training=self.is_training_pl,
                            spatial_squeeze=False,
                            fc_conv_padding='VALID')
                    elif net_name.startswith('resnet_v2'):
                        # TODO: imlement segmentation
                        postnorm, self.end_points = nets_factory.networks_map[net_name](
                            input_images_preprocessed, num_classes=None,
                            is_training=self.is_training_pl,
                            # is_training=False,
                            global_pool=False,
                            spatial_squeeze=False,
                            output_stride=None)
                        self.end_points[self.net_name + '/postnorm'] = postnorm
                    elif net_name == 'inception_resnet_v2':
                        _, self.end_points = nets_factory.networks_map[net_name](
                            input_images_preprocessed, num_classes=None,
                            is_training=self.is_training_pl,
                            dropout_keep_prob=0.8)
                    else:
                        raise NotImplementedError(net_name)
                # pprint([(key, tensor.shape.as_list()) for key, tensor in self.end_points.items()])
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    weights_regularizer=slim.l2_regularizer(weight_decay),
                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    biases_initializer=tf.zeros_initializer()
                                    ):
                    self.define_regression()
                    if use_segmentation_loss:
                        self.loss = tf.multiply(segmentation_loss_weight,
                                                segmentation_loss(self.heatmaps,
                                                                  self.input_heatmaps),
                                                name='segmentation_xe_loss')
                    if use_regression_loss:
                        if self.class_weights:
                            regression_class_weights = self.class_weights[1:]
                        else:
                            regression_class_weights = self.class_weights
                        self.regression_loss = tf.multiply(regression_loss_weight,
                                                           regression_l2_loss(self.obj_counts,
                                                                              self.input_gt_counts,
                                                                              regression_class_weights),
                                                           name='regression_l2_loss')
                        tf.losses.add_loss(self.regression_loss)

                    self.total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
        if should_create_summaries:
            self.create_summaries()
        self.graph = tf.get_default_graph()
        self.sess = None

    def define_regression(self, dropout_keep_prob=0.5):
        if self.net_name.startswith('vgg_1'):
            net = self.end_points[self.net_name + '/fc7']
            net = slim.dropout(net, dropout_keep_prob, is_training=self.is_training_pl,
                               scope='dropout7_1')
            kernel_size = 7 if self.use_segmentation_loss else 1
            net = slim.conv2d(net, self.extra_fc_size, [kernel_size, kernel_size],
                              padding='VALID',
                              activation_fn=tf.nn.relu, scope='fc8_small')
            self.end_points['fc8_small'] = net
            net = slim.dropout(net, dropout_keep_prob, is_training=self.is_training_pl,
                               scope='dropout8_small')
            self.obj_counts = slim.conv2d(net, 5, [1, 1], padding='VALID',
                                          activation_fn=None, scope='fc_count')
        elif self.net_name.startswith('resnet_v2') or self.net_name == 'inception_resnet_v2':
            if self.net_name.startswith('resnet_v2'):
                net = self.end_points[self.net_name + '/postnorm']
            else:
                net = self.end_points['Conv2d_7b_1x1']
                dropout_keep_prob = 0.8
            if self.global_pool == 'sum':
                net = tf.reduce_sum(net, [1, 2], name='pool5', keep_dims=True)
                kernel_size = 1
            elif self.global_pool == 'avg':
                net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                      scope='AvgPool')
                kernel_size = 1
            else:
                kernel_size = net.get_shape().as_list()[1]
            if self.extra_fc_size:
                net = slim.conv2d(net, self.extra_fc_size, [kernel_size, kernel_size],
                                  padding='VALID',
                                  activation_fn=tf.nn.relu, scope='extra_fc')
                self.end_points['extra_fc'] = net
                net = slim.dropout(net, dropout_keep_prob, is_training=self.is_training_pl,
                                   scope='extra_fc_dropout')
                kernel_size = 1
            self.obj_counts = slim.conv2d(net, 5, [kernel_size, kernel_size], padding='VALID',
                                          activation_fn=None, scope='fc_count')
        else:
            raise NotImplementedError()
        self.obj_counts = tf.squeeze(self.obj_counts, axis=[1, 2], name='fc_count_squeeze')
        self.end_points['fc_count'] = self.obj_counts
        return self.obj_counts

    def create_summaries(self):
        if self.use_segmentation_loss:
            tf.summary.scalar('net/sum/heatmaps', tf.reduce_sum(self.heatmaps))
            tf.summary.scalar('net/sparsity/heatmaps',
                              tf.nn.zero_fraction(self.heatmaps))
            tf.summary.scalar('net/{}'.format(self.loss.op.name.split('/', 1)[0]),
                              self.loss)
        if self.use_regression_loss:
            tf.summary.scalar(
                'net/{}'.format(self.regression_loss.op.name.split('/')[-1]),
                self.regression_loss)
        tf.summary.scalar('net/total_loss', self.total_loss)


def inception_preproc(images):
    """
    Transform images
    Args:
        images: tensor with images
    """
    images = tf.cast(images, tf.float32)
    images /= 255
    images -= 0.5
    images *= 2.0
    # images = tf.Print(images, [tf.reduce_min(images), tf.reduce_max(images)], 'min,max valeus of images:')
    return images


def vgg_preproc(images):
    """Subtracts the given means from each image channel.

      For example:
        means = [123.68, 116.779, 103.939]
        image = _mean_image_subtraction(image, means)

      Note that the rank of `image` must be known.

      Args:
        images: a tensor of size [N, height, width, C].
        means: a C-vector of values to subtract from each channel.

      Returns:
        the centered images.

      Raises:
        ValueError: If the rank of `image` is unknown, if `image` has a rank other
          than three or if the number of channels in `image` doesn't match the
          number of values in `means`.
      """
    if images.get_shape().ndims != 4:
        raise ValueError('Input must be of size [N, height, width, C>0]')
    num_channels = images.get_shape().as_list()[-1]
    means = [vgg_preprocessing._R_MEAN,
             vgg_preprocessing._G_MEAN,
             vgg_preprocessing._B_MEAN]
    if num_channels is not None and len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels', num_channels)
    num_channels = 3
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


preprocessing_fn_map = {
    'inception': inception_preproc,
    'resnet_v2_50': inception_preproc,
    'resnet_v2_101': inception_preproc,
    'resnet_v2_152': inception_preproc,
    'inception_resnet_v2': inception_preproc,
    'vgg': vgg_preproc,
    'vgg_a': vgg_preproc,
    'vgg_16': vgg_preproc,
    'vgg_19': vgg_preproc,
}





def get_fixed_vars(model):
    fixed_vars = list()
    lower_vars = list()
    if model == 'vgg_16':
        # print 'Warning! We leave the first conv layer trainable!'
        fixed_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       'vgg_16/conv1') + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       'vgg_16/conv2/conv2_1')
    elif model == 'vgg_19':
        # print 'Warning! We leave the first conv layer trainable!'
        fixed_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       'vgg_19/conv1') + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       'vgg_19/conv2/conv2_1')
        pass
    if model == 'resnet_v2_50':
        fixed_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       'resnet_v2_50/conv1') + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'resnet_v2_50/block1')
        # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'resnet_v2_50/block2')
    elif model == 'resnet_v2_101':
        fixed_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       'resnet_v2_101/conv1') + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       'resnet_v2_101/block1')
    elif model == 'resnet_v2_152':
        fixed_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       'resnet_v2_152/conv1') + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       'resnet_v2_152/block1') #+ \
                     # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                     #                   'resnet_v2_152/block2')
        # lower_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        #                                'resnet_v2_152/block2') + \
        #              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        #                                'resnet_v2_152/block3') + \
        #              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        #                                'resnet_v2_152/block4')
    elif model == 'inception_resnet_v2':
        fixed_names = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
                       'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'Mixed_5b']
        fixed_vars = list()
        for name in fixed_names:
            fixed_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       'InceptionResnetV2/' + name))

    return fixed_vars, lower_vars