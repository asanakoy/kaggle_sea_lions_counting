# Copyright (c) 2017 Artsiom Sanakoyeu
import os
from os.path import join
import time
import numpy as np
from pprint import pformat
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.core.framework import summary_pb2
from tqdm import tqdm
import deepdish as dd

from data_utils import CLASS_NAMES
from data_utils import LionClasses
from helper import visualize_model_predictions
from helper import visualize_segmentation


CONV_PARAMS = {
    'small': {
        'conv1': dict(n_out=24, kernel_size=5),
        'conv2': dict(n_out=48, kernel_size=3),
        'conv3': dict(n_out=24, kernel_size=3),
        'conv4': dict(n_out=12, kernel_size=3),
    },

    'large': {
        'conv1': dict(n_out=16, kernel_size=9),
        'conv2': dict(n_out=32, kernel_size=7),
        'conv3': dict(n_out=16, kernel_size=7),
        'conv4': dict(n_out=8, kernel_size=7),
    }
}


# CLASS_WEIGHT = [10., 10., 1., 10., 10.]
_DEFAULT_CLASS_WEIGHT = [1., 1., 1., 1., 1.]
# DEFAULT_CLASS_WEIGHT = [0., 0., 1., 0., 0.]


def l2_heatmap_loss(predicted_heatmaps, gt_heatmaps, heatmap_multiplier=1.0, class_weights=None,
                    name='heatmap_l2_loss'):
    """
    Calculate l2 Loss on 5 heatmaps separately
    Returns:
      the heatmaps loss as a float tensor.
    """
    if class_weights is None:
        class_weights = _DEFAULT_CLASS_WEIGHT
    with tf.variable_scope(name):
        NUM_HEATMAPS = 5
        # tf.assert_equal(NUM_HEATMAPS, predicted_heatmaps.get_shape()[-1])
        heatmap_sq_diff = tf.square(tf.subtract(predicted_heatmaps, gt_heatmaps * heatmap_multiplier))
        # heatmap_sq_diff += tf.multiply(heatmap_sq_diff,
        #                                tf.cast(tf.greater(gt_heatmaps, 0.0), tf.float32)) \
        #                                * LOSS_MULT

        heatmap_sqdiff_list = tf.split(heatmap_sq_diff, NUM_HEATMAPS, axis=3)

        losses_list = list()
        for class_id, sq_diff in enumerate(heatmap_sqdiff_list):
            class_loss = tf.sqrt(tf.reduce_sum(sq_diff, axis=[1, 2])) * class_weights[class_id] # shape = N x 1
            losses_list.append(class_loss)
            # tf.summary.scalar(name + '/' + CLASS_NAMES[class_id], class_loss)
        loss = tf.reduce_mean(tf.concat(losses_list, 1), name=name)
        tf.losses.add_loss(loss)
    return loss


def segmentation_loss(predicted_heatmaps, gt_labels,
                      name='segmentation_xe_loss'):
    """
    Calculate l2 Loss on 5 heatmaps separately
    Returns:
      the heatmaps loss as a float tensor.
    """
    NUM_CLASSES = 6
    with tf.name_scope(name) as sc:
        maps_shape = tf.shape(predicted_heatmaps)
        tf.assert_equal(NUM_CLASSES, maps_shape[-1])
        new_shape = tf.stack([maps_shape[0], -1, maps_shape[-1]])
        maps_reshaped = tf.reshape(predicted_heatmaps, shape=new_shape)

        gt_labels_reshaped = tf.reshape(gt_labels,
                                        shape=tf.stack([tf.shape(gt_labels)[0], -1]))

        loss = tf.losses.sparse_softmax_cross_entropy(labels=gt_labels_reshaped,
                                                      logits=maps_reshaped,
                                                      scope=sc)

    return loss


def regression_l2_loss(predicted_counts, gt_counts, class_weights=None,
                    name='regression_l2_loss'):

    if class_weights is None:
        class_weights = _DEFAULT_CLASS_WEIGHT
    NUM_CLASSES = 5

    assert len(class_weights) == 5
    with tf.variable_scope(name) as sc:
        tf.assert_equal(NUM_CLASSES, predicted_counts.get_shape()[-1])
        tf.assert_equal(NUM_CLASSES, gt_counts.get_shape()[-1])
        tf.assert_equal(predicted_counts.get_shape()[1:], gt_counts.get_shape()[1:])

        counts_sq_diff = tf.square(predicted_counts - gt_counts)
        diff_per_class = tf.reduce_mean(counts_sq_diff, axis=0) * class_weights
        loss = tf.reduce_sum(diff_per_class, axis=0, name=sc.name)
    return loss


def regression_sum_heatmap_loss(predicted_heatmaps, gt_heatmaps, heatmap_multiplier=1.0,
                    name='regression_sum_hmap_loss'):

    with tf.variable_scope(name):
        # NUM_HEATMAPS = 5
        # tf.assert_equal(NUM_HEATMAPS, predicted_heatmaps.get_shape()[-1])

        counts = tf.reduce_sum(predicted_heatmaps, axis=[1, 2])
        gt_counts = tf.reduce_sum(gt_heatmaps, axis=[1, 2]) * heatmap_multiplier
        counts_sq_diff = tf.square(counts - gt_counts)
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(counts_sq_diff, axis=1)), axis=0)
    return loss


def l1_heatmap_loss(predicted_heatmaps, gt_heatmaps, heatmap_multiplier=1.0,
                    name='heatmap_l1_loss'):
    """
    Calculate l2 Loss on 5 heatmaps separately
    Returns:
      the heatmaps loss as a float tensor.
    """
    with tf.variable_scope(name):
        NUM_HEATMAPS = 5
        tf.assert_equal(NUM_HEATMAPS, predicted_heatmaps.get_shape()[-1])
        abs_diff = tf.abs(
            tf.subtract(predicted_heatmaps, gt_heatmaps * heatmap_multiplier))
        # heatmap_sq_diff += tf.multiply(heatmap_sq_diff,
        #                                tf.cast(tf.greater(gt_heatmaps, 0.0), tf.float32)) \
        #                                * LOSS_MULT

        heatmap_sqdiff_list = tf.split(abs_diff, NUM_HEATMAPS, axis=3)

        losses_list = list()
        for class_id, diff in enumerate(heatmap_sqdiff_list):
            class_loss = tf.reduce_sum(diff, axis=[1, 2])  # shape = N x 1
            losses_list.append(class_loss)
            # tf.summary.scalar(name + '/' + CLASS_NAMES[class_id], class_loss)
        loss = tf.reduce_mean(tf.concat(losses_list, 1), name=name)
    return loss


def my_arg_scope(use_batch_norm=True,
                batch_norm_decay=0.9997,
                batch_norm_epsilon=0.001,
                is_training=True):
    """Defines the default arg scope for inception models.

    Args:
      weight_decay: The weight decay to use for regularizing the model.
      use_batch_norm: "If `True`, batch_norm is applied after each convolution.
      batch_norm_decay: Decay for batch norm moving average.
      batch_norm_epsilon: Small float added to variance to avoid dividing by zero
        in batch norm.

    Returns:
      An `arg_scope` to use for the inception models.
    """
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': batch_norm_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': batch_norm_epsilon,
        # collection containing update_ops.
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    # Set weight_decay for weights in Conv and FC layers.

    with slim.arg_scope(
            [slim.conv2d],
            activation_fn=tf.nn.relu,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training) as sc:

            return sc


class NetStream:
    """
    4-layer FCN network
    """

    def __init__(self, stream_name, conv_params):

        if len(conv_params) != 4:
            raise ValueError('conv_params must be a list of 4 elements')
        self.conv_params = conv_params
        self.stream_name = stream_name
        print 'Define Stream "{}" with the following number of units:\n{}'.format(stream_name,
                                                                                  pformat(self.conv_params))

    def __call__(self, inputs, weight_decay=0.000001,
                 use_batch_norm=False,
                 reuse=False,
                 is_training=True):
        conv_params = self.conv_params
        end_points = {}
        with tf.variable_scope(self.stream_name, reuse=reuse) as sc:
            with slim.arg_scope(my_arg_scope(use_batch_norm=use_batch_norm, is_training=is_training)):
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_regularizer=slim.l2_regularizer(weight_decay),
                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    biases_initializer=tf.zeros_initializer(),
                                    padding='SAME'):
                    end_point = 'conv1'
                    n_out = conv_params[end_point]['n_out']
                    kernel_size = conv_params[end_point]['kernel_size']
                    net = slim.conv2d(inputs, n_out, [kernel_size, kernel_size],
                                      stride=1,
                                      scope=end_point)

                    end_points[end_point] = net

                    end_point = 'conv2'
                    n_out = conv_params[end_point]['n_out']
                    kernel_size = conv_params[end_point]['kernel_size']
                    net = slim.conv2d(inputs, n_out, [kernel_size, kernel_size],
                                      stride=1,
                                      scope=end_point)
                    end_points[end_point] = net
                    end_point = 'maxpool2'
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
                    end_points[end_point] = net

                    end_point = 'conv3'
                    n_out = conv_params[end_point]['n_out']
                    kernel_size = conv_params[end_point]['kernel_size']
                    net = slim.conv2d(net, n_out, [kernel_size, kernel_size],
                                      stride=1,
                                      scope=end_point)
                    end_points[end_point] = net
                    end_point = 'maxpool3'
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
                    end_points[end_point] = net

                    end_point = 'conv4'
                    n_out = conv_params[end_point]['n_out']
                    kernel_size = conv_params[end_point]['kernel_size']
                    net = slim.conv2d(net, n_out, [kernel_size, kernel_size],
                                      stride=1,
                                      scope=end_point)
                    end_points[end_point] = net

                    tf.summary.scalar('activation_norm_' + end_point + '_' + self.stream_name,
                                      tf.nn.l2_loss(net))

                    return net, end_points


class Net:
    def __init__(self, input_images, input_heatmaps, streams=None,
                 relu_heatmaps=True,
                 heatmap_in_loss_multiplier=1.0,
                 use_batch_norm=False,
                 class_weights=None,
                 use_regression_loss=False,
                 regression_loss_weight=0.1,
                 is_training=True,
                 reuse=False):
        self.is_training_pl = None
        self.input_images = input_images
        self.input_heatmaps = input_heatmaps
        self.heatmap_in_loss_multiplier = heatmap_in_loss_multiplier
        tf.summary.scalar('heatmap_in_loss_mult', heatmap_in_loss_multiplier)
        if use_regression_loss:
            tf.summary.scalar('other/regression_loss_weight', regression_loss_weight)

        print 'Create Net. Use ReLU after heatmaps={}'.format(relu_heatmaps)

        with tf.device('/gpu:0'):
            with tf.variable_scope('net', reuse=reuse):
                self.global_iter_counter = tf.get_variable('global_iter_counter', shape=[],
                                                           initializer=tf.constant_initializer(0),
                                                           trainable=False)
                self.streams_outputs = list()
                self.streams_end_points = list()
                for stream_fn in streams:
                    out, end_points = stream_fn(self.input_images,
                                                use_batch_norm=use_batch_norm,
                                                is_training=is_training,
                                                reuse=reuse)
                    self.streams_outputs.append(out)
                    self.streams_end_points.append(end_points)
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    weights_regularizer=slim.l2_regularizer(0.00001),
                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    biases_initializer=tf.zeros_initializer(),
                                    padding='SAME',
                                    stride=1):
                    feats_concat = tf.concat(self.streams_outputs, 3, name='concat')
                    self.heatmaps = slim.conv2d(feats_concat, 5, [1, 1],
                                      activation_fn=tf.nn.relu if relu_heatmaps else None,
                                      padding='SAME',
                                      scope='conv_heatmap')

                    heatmaps_per_class = tf.split(self.heatmaps, LionClasses.NUM_CLASSES, axis=3)
                    self.loss = l2_heatmap_loss(self.heatmaps, self.input_heatmaps,
                                                heatmap_multiplier=self.heatmap_in_loss_multiplier,
                                                class_weights=class_weights)
                    tf.losses.add_loss(self.loss)
                    if use_regression_loss:
                        self.regression_loss = regression_loss_weight * regression_sum_heatmap_loss(self.heatmaps, self.input_heatmaps,
                                                               heatmap_multiplier=self.heatmap_in_loss_multiplier)
                        tf.losses.add_loss(self.regression_loss)
                    self.total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

        net_name = 'net' if is_training else 'net_test'
        tf.summary.scalar('{}/sum/heatmaps'.format(net_name), tf.reduce_sum(self.heatmaps))
        tf.summary.scalar('{}/sparsity/heatmaps'.format(net_name), tf.nn.zero_fraction(self.heatmaps))
        for class_id, heatmap in enumerate(heatmaps_per_class):
            tf.summary.scalar('{}/sum/heatmaps/{}'.format(net_name, CLASS_NAMES[class_id]),
                              tf.reduce_sum(heatmap))

        tf.summary.scalar('{}/{}'.format(net_name, self.loss.op.name.split('/')[-1]), self.loss)
        if use_regression_loss:
            tf.summary.scalar('{}/{}'.format(net_name, self.regression_loss.op.name.split('/')[-1]),
                              self.regression_loss)
        tf.summary.scalar('{}/{}'.format(net_name, self.total_loss.op.name.split('/')[-1]), self.total_loss)
        self.graph = tf.get_default_graph()
        self.sess = None


def training_convnet(net, loss_op, lr, optimizer_type='adam',
                     fixed_vars=None, lower_vars=None,
                     trace_gradients=False):
    if fixed_vars is None:
        fixed_vars = list()
    if lower_vars is None:
        lower_vars = list()
    with net.graph.as_default():
        lower_lr = lr * 0.1
        lrs = [lr, lower_lr]
        print('Creating optimizer {}'.format(optimizer_type))
        tf.summary.scalar('lr', lr)
        optimizer = [None, None]
        for i in xrange(2):
            if optimizer_type == 'adagrad':
                optimizer[i] = tf.train.AdagradOptimizer(lrs[i], initial_accumulator_value=0.00001)
            elif optimizer_type == 'sgd':
                optimizer[i] = tf.train.GradientDescentOptimizer(lrs[i])
            elif optimizer_type == 'momentum':
                optimizer[i] = tf.train.MomentumOptimizer(lrs[i], momentum=0.9)
            elif optimizer_type == 'adam':
                optimizer[i] = tf.train.AdamOptimizer(lrs[i])
            else:
                raise ValueError('Unknown optimizer type {}'.format(optimizer_type))

        print('LR: {}'.format(lr))

        all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        trainable_vars = list(set(all_trainable_vars).
            difference(set(fixed_vars + lower_vars)))
        print 'all trainable vars:', len(all_trainable_vars)
        print 'fixed vars:', len(fixed_vars)
        print 'lower vars:', len(lower_vars)
        print 'to train vars:', len(trainable_vars)
        assert len(all_trainable_vars) == len(fixed_vars) + len(lower_vars) + len(trainable_vars)

        grads = tf.gradients(loss_op, trainable_vars + lower_vars)
        grads_full = grads[:len(trainable_vars)]
        gads_lower = grads[len(trainable_vars):]
        with tf.name_scope('grad_norms'):
            for v, grad in zip(trainable_vars, grads):
                if grad is not None:
                    grad_norm_op = tf.nn.l2_loss(grad, name=format(v.name[:-2]))
                    tf.add_to_collection('grads', grad_norm_op)
                    if trace_gradients:
                        # mean, var = tf.nn.moments(grad, axes=[1])
                        tf.summary.scalar(grad_norm_op.name, grad_norm_op)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer[0].apply_gradients(zip(grads_full, trainable_vars),
                                                      global_step=net.global_iter_counter,
                                                      name='full_train_op')
            if len(lower_vars):
                lower_tran_op = optimizer[1].apply_gradients(zip(gads_lower, lower_vars),
                                                          name='lower_train_op')
                train_op = tf.group(train_op, lower_tran_op)
        return train_op


def create_sumamry(tag, value):
    """
    Create a summary for logging via tf.train.SummaryWriter
    """
    x = summary_pb2.Summary.Value(tag=tag, simple_value=value)
    return summary_pb2.Summary(value=[x])


def obtain_predictions_and_gt(net, batch_generator, global_step=None, batch_size=2, visualize=False,
                              predictions_output_dir=None):
    if global_step is None:
        global_step = net.sess.run(net.global_iter_counter)

    total_loss = 0.0
    assert batch_generator.cur_pos == 0
    num_iterations = int(np.ceil(len(batch_generator.images) / float(batch_size)))
    pred_counts_tiles = list()
    gt_counts_tiles = list()

    if num_iterations == 0:
        print 'No val images found!'
        return

    for batch_i in tqdm(xrange(num_iterations)):
        feed_dict = batch_generator.next_feed_dict(net, batch_size=batch_size, phase='test')
        loss_value, heatmaps = net.sess.run(
            [net.total_loss, net.heatmaps],
            feed_dict=feed_dict)

        if visualize:
            visualize_model_predictions(feed_dict[net.input_images],
                                        feed_dict[net.input_heatmaps],
                                        heatmaps,
                                        visualize=True,
                                        visualize_each_class=True)
        if predictions_output_dir is not None:
            dd.io.save(join(predictions_output_dir, 'pred_heatmaps_batch-{:05d}.hdf5'.format(batch_i)), heatmaps)

        total_loss += loss_value

        for tile_i in xrange(len(heatmaps)):
            pred = heatmaps[tile_i]
            gt = feed_dict[net.input_heatmaps][tile_i]
            preds_per_class = pred.sum(axis=(0, 1))
            assert preds_per_class.shape == (5,), preds_per_class.shape
            gt_per_class = gt.sum(axis=(0, 1))
            assert gt_per_class.shape == (5,), gt_per_class.shape
            pred_counts_tiles.append(preds_per_class)
            gt_counts_tiles.append(gt_per_class)

    pred_counts_tiles = np.asarray(pred_counts_tiles)
    gt_counts_tiles = np.asarray(gt_counts_tiles)
    total_loss /= num_iterations
    return global_step, pred_counts_tiles, gt_counts_tiles, total_loss


def obtain_predictions_and_gt_segm(net, batch_generator, global_step=None, batch_size=2,
                              visualize=False,
                              predictions_output_dir=None):
    if global_step is None:
        global_step = net.sess.run(net.global_iter_counter)

    total_loss = 0.0
    assert batch_generator.cur_pos == 0
    num_iterations = int(np.ceil(len(batch_generator.images) / float(batch_size)))
    pred_tiles = list()
    gt_tiles = list()

    if num_iterations == 0:
        print 'No val images found!'
        return

    for batch_i in tqdm(xrange(num_iterations)):
        feed_dict = batch_generator.next_feed_dict(net, batch_size=batch_size, phase='test')
        heatmap_probs = tf.nn.softmax(net.heatmaps)
        loss_value, heatmaps, heatmap_probs = net.sess.run(
            [net.total_loss, net.heatmaps, heatmap_probs],
            feed_dict=feed_dict)

        if visualize:
            visualize_segmentation(feed_dict[net.input_images],
                                   feed_dict[net.input_heatmaps],
                                   heatmap_probs,
                                    visualize=True,
                                    visualize_each_class=True)
        if predictions_output_dir is not None:
            dd.io.save(
                join(predictions_output_dir, 'pred_heatmaps_batch-{:05d}.hdf5'.format(batch_i)),
                heatmaps)

        total_loss += loss_value

        for tile_i in xrange(len(heatmaps)):
            pred = heatmaps[tile_i]
            gt = feed_dict[net.input_heatmaps][tile_i]
            pred_tiles.append(pred)
            gt_tiles.append(gt)

    pred_tiles = np.asarray(pred_tiles)
    gt_tiles = np.asarray(gt_tiles)
    total_loss /= num_iterations
    print 'total_loss=', total_loss
    return global_step, pred_tiles, gt_tiles, total_loss


def calc_test_metrics(global_step, pred_counts_tiles,
                      gt_counts_tiles, total_loss, cnt_tiles_per_image,
                      heatmap_in_loss_multiplier=1.0, summary_writer=None, min_threshold=0,
                      calc_per_tile=True, split_name='val'):
    """

    :param global_step:
    :param pred_counts_tiles: array N x 5, where N is the number of tiles
    :param gt_counts_tiles: array N x 5
    :param total_loss:
    :param num_tiles_per_image:
    :param heatmap_in_loss_multiplier:
    :param summary_writer:
    :param min_threshold: clip predictions per tile to have this min number of objects
    :return:
    """
    assert len(pred_counts_tiles) == len(gt_counts_tiles)
    start_time = time.time()

    if cnt_tiles_per_image is not None:
        assert len(pred_counts_tiles) == np.sum(cnt_tiles_per_image), \
            'Wrong number of tiles {}'.format(np.sum(cnt_tiles_per_image))
        pred_counts_full_imgs = list()
        gt_counts_full_imgs = list()
        cur_pos = 0
        for cur_tiles_per_image in cnt_tiles_per_image:
            assert cur_pos + cur_tiles_per_image <= len(pred_counts_tiles)
            slice = np.s_[cur_pos:cur_pos + cur_tiles_per_image]
            pred_counts_full_imgs.append(pred_counts_tiles[slice].sum(axis=0))
            gt_counts_full_imgs.append(gt_counts_tiles[slice].sum(axis=0))
            cur_pos += cur_tiles_per_image
        pred_counts_full_imgs = np.asarray(pred_counts_full_imgs)
        gt_counts_full_imgs = np.asarray(gt_counts_full_imgs)

        pred_counts_full_imgs = np.round(pred_counts_full_imgs / heatmap_in_loss_multiplier).astype(int)
        pred_counts_full_imgs = pred_counts_full_imgs.clip(min=min_threshold)
        gt_counts_full_imgs = np.round(gt_counts_full_imgs).astype(int)

        pred_counts_tiles = pred_counts_tiles / heatmap_in_loss_multiplier
        pred_counts_tiles = pred_counts_tiles.clip(min=min_threshold)
        gt_counts_tiles = gt_counts_tiles

        assert pred_counts_tiles.shape[1] == LionClasses.NUM_CLASSES, pred_counts_tiles.shape
    else:
        # tiles are full images
        pred_counts_full_imgs = np.round(pred_counts_tiles / heatmap_in_loss_multiplier).astype(int)
        pred_counts_full_imgs = pred_counts_full_imgs.clip(min=min_threshold)
        gt_counts_full_imgs = np.round(gt_counts_tiles).astype(int)
        pred_counts_tiles = None
        gt_counts_tiles = None

    assert pred_counts_full_imgs.shape[1] == LionClasses.NUM_CLASSES, pred_counts_full_imgs.shape

    rmse_dict = dict()

    for pred_counts, gt_counts, name in \
            [(pred_counts_tiles, gt_counts_tiles, 'tiles'),
             (pred_counts_full_imgs, gt_counts_full_imgs, 'full_imgs')]:
        if pred_counts is None or gt_counts is None:
            continue
        if name == 'tiles' and not calc_per_tile:
            continue
        diff = gt_counts - pred_counts
        assert diff.shape[1] == LionClasses.NUM_CLASSES
        rmse = np.sqrt(np.mean(diff**2, axis=0))
        rmse_dict[name] = np.mean(rmse)
        assert len(rmse) == LionClasses.NUM_CLASSES
        mean_absloute_error = np.mean(np.abs(diff), axis=0)
        if summary_writer:
            summary_writer.add_summary(
                create_sumamry('RMSE_{}/{}'.format(split_name, name), np.mean(rmse)),
                global_step=global_step)
            # summary_writer.add_summary(
            #     create_sumamry('MAE_{}/{}'.format(split_name, name), np.mean(mean_absloute_error)),
            #     global_step=global_step)
        print '{} {}/RMSE: {}'.format(split_name, name, np.mean(rmse))
        print '{} {}/RMSE per class:'.format(split_name, name)
        for class_id in xrange(LionClasses.NUM_CLASSES):
            class_name = CLASS_NAMES[class_id]
            print '    {}:\t{}'.format(class_name, rmse[class_id])
            if summary_writer:
                summary_writer.add_summary(
                    create_sumamry('RMSE_{}/{}/{}'.format(split_name, name, class_name), rmse[class_id]),
                    global_step=global_step)
        print '{} {}/MAE: {}'.format(split_name, name, np.mean(mean_absloute_error))
    if summary_writer:
        summary_writer.add_summary(
            create_sumamry('{}/total_loss'.format(split_name), total_loss),
            global_step=global_step)
    print '{} loss: {}'.format(split_name, total_loss)
    print('Test completed in {:.3f} s'.format(time.time() - start_time))
    return pred_counts_full_imgs, gt_counts_full_imgs, rmse_dict


def calculate_reg_predictions(net, batch_generator,
                              global_step, batch_size):
    if global_step is None:
        global_step = net.sess.run(net.global_iter_counter)

    total_loss = 0.0
    assert batch_generator.cur_pos == 0
    if batch_generator.val_epoch_size is None:
        num_iterations = int(np.ceil(len(batch_generator.images) / float(batch_size)))
    else:
        num_iterations = batch_generator.val_epoch_size
    pred_tiles = list()
    gt_tiles = list()

    if num_iterations == 0:
        print 'No val images found!'
        return

    for _ in tqdm(xrange(num_iterations)):
        feed_dict = batch_generator.next_feed_dict(net, batch_size=batch_size, phase='test')
        # print [net.total_loss, net.obj_counts, net.input_gt_counts]
        loss_value, obj_counts, gt_counts = net.sess.run(
            [net.total_loss, net.obj_counts, net.input_gt_counts],
            feed_dict=feed_dict)

        total_loss += loss_value
        assert len(obj_counts.shape) == 2
        for tile_i in xrange(len(obj_counts)):
            pred_cnt = obj_counts[tile_i]
            gt_cnt = gt_counts[tile_i]
            pred_tiles.append(pred_cnt)
            gt_tiles.append(gt_cnt)

    pred_tiles = np.asarray(pred_tiles)
    gt_tiles = np.asarray(gt_tiles)
    total_loss /= num_iterations
    print 'total_loss=', total_loss
    return global_step, pred_tiles, gt_tiles, total_loss


def run_training(net, batch_generator, train_op, loss_op, saver,
                 test_net=None,
                 test_fn=None,
                 val_batch_generator=None,
                 batch_size=2, max_iter=0, snapshot_iter=1800,
                 output_dir='/tmp/tf_output_dir', test_step=1e9,
                 summary_step=200, log_step=1, do_not_summary=2000, visualize_iter=None,
                 val_cnt_tiles_per_image=None, moving_average_loss=True):

    with net.graph.as_default():
        summary_writer = tf.summary.FileWriter(output_dir, net.sess.graph)
        summary_op = tf.summary.merge_all()

    global_step = None
    start_training_time = time.time()
    if moving_average_loss:
        loss_ema = None
    for step in xrange(max_iter + 1):
        if step % snapshot_iter == 0 and step > 0:
            checkpoint_prefix = os.path.join(output_dir, 'checkpoint')
            saver.save(net.sess, checkpoint_prefix, global_step=global_step)
        if step == max_iter:
            break

        if step % test_step == 0 and (step or test_fn):
            print 'Test network'
            if test_fn is None:
                global_step, pred_counts_tiles, gt_counts_tiles, total_loss \
                    = obtain_predictions_and_gt(test_net, val_batch_generator,
                                                global_step=global_step, batch_size=batch_size)
                calc_test_metrics(global_step, pred_counts_tiles,
                                  gt_counts_tiles, total_loss, val_cnt_tiles_per_image,
                                  test_net.heatmap_in_loss_multiplier, summary_writer)
            else:
                test_fn(test_net, val_batch_generator, global_step=global_step, batch_size=batch_size,
                        cnt_tiles_per_image=val_cnt_tiles_per_image, summary_writer=summary_writer)

        # training
        start_time = time.time()
        # TODO: fix if batch_generator. If we use tfrecords
        feed_dict = batch_generator.next_feed_dict(net, batch_size=batch_size, phase='train')

        if step % summary_step == 0 or step + 1 == max_iter or step % test_step == 0:
            if net.heatmaps is not None:
                heatmaps, global_step, summary_str, _, loss_value = net.sess.run(
                    [net.heatmaps,
                    net.global_iter_counter,
                     summary_op,
                     train_op,
                     loss_op],
                    feed_dict=feed_dict)
                print 'heatmaps min-max:', heatmaps.min(), heatmaps.max()
            else:
                global_step, summary_str, _, loss_value = net.sess.run(
                    [net.global_iter_counter,
                     summary_op,
                     train_op,
                     loss_op],
                    feed_dict=feed_dict)

            if global_step >= do_not_summary:
                summary_writer.add_summary(summary_str, global_step=global_step)
            if visualize_iter is not None and global_step >= visualize_iter:
                visualize_model_predictions(feed_dict[net.input_images],
                                            feed_dict[net.input_heatmaps],
                                            heatmaps,
                                            visualize=True)
        else:
            global_step, _, loss_value = net.sess.run(
                [net.global_iter_counter, train_op, loss_op],
                feed_dict=feed_dict)

        if moving_average_loss:
            if step == 0:
                loss_ema = loss_value
            else:
                decay = 0.9997
                loss_ema = loss_ema * decay + (1 - decay) * loss_value
            if step % summary_step == 0:
                summary_writer.add_summary(
                    create_sumamry('net/ema_loss', loss_ema), global_step=global_step)

        duration = time.time() - start_time

        if step % log_step == 0 or step + 1 == max_iter:
            print('Step %d: loss = %.2f (%.3f s, %.2f im/s)'
                  % (global_step, loss_value, duration, batch_size / duration))

    training_duration = (time.time() - start_training_time)
    print('Training completed in {:.1f} min ({:.2f} h)'.format(training_duration / 60.0,
                                                               training_duration / 3600.0))

