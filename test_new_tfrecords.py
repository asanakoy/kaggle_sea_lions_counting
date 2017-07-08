# Copyright (c) 2017 Artsiom Sanakoyeu
"""
Run prediction on train or validation splits and calculate RMSE
"""
import argparse
import numpy as np
import os
from os.path import join
from pprint import pformat
import sys
import tensorflow as tf
import tflearn
from tqdm import tqdm

from records.records import read_train_batch
from data_utils import ROOT_DIR
import net_spec
from net_spec import SegPlusRegNet
from net_config import calc_test_metrics


def parse_bool(value):
    return bool(int(value))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suf', help='Checkpoints dir suffix.')
    parser.add_argument('--model', default='small', help='Network model type.',
                        choices=['vgg_16', 'vgg_19', 'resnet_v2_50',
                                 'resnet_v2_101', 'resnet_v2_152', 'inception_resnet_v2'])
    parser.add_argument('--snapshot_iter', type=int, default=1000, help='Do snapsot every')
    parser.add_argument('--max_iter', type=int, default=7200 * 4000, help='max iterations')
    parser.add_argument('--val_part', type=float, default=0.1,
                        help='part of the validation split')
    parser.add_argument('--tile_size', type=int, default=224,
                        help='part of the validation split')

    parser.add_argument('--extra_fc', type=int, default=0,
                        help='size of the extra fc layer. If 0 do not add extra layer.')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')

    parser.add_argument('--pool', choices=['avg', 'sum'], default=None,
                        help='sum/avg pool of last base conv(pool) layer')

    parser.add_argument('--num_img', type=int, default=None,
                        help='number of val images to load')

    parser.add_argument('--preds_out', default=None,
                        help='predicted maps output dir')

    parser.add_argument('--full_scale_aug', action='store_true',
                        help='Do scale aug on full images?')

    parser.add_argument('--min_scale_aug', type=float, default=None,
                        help='Min scale for scale augmentation')
    parser.add_argument('--max_scale_aug', type=float, default=None,
                        help='Min scale for scale augmentation')

    parser.add_argument('--coords_v', type=int, default=0,
                        help='Version of the coordinates of the lions to use')

    parser.add_argument('--split', default='val', choices=['train', 'val'],
                        help='which split to use for evaluation')

    parser.add_argument('--dbg', action='store_true',
                        help='Should debug (load only few images)?')

    parser.add_argument('--aug_seed', default=2017,
                        help='seed for augmentations')

    parser.add_argument('--rescaled', action='store_true',
                        help='use tfrecords with adaptively rescaled images?')

    args = parser.parse_args(sys.argv[1:])
    return args


def calculate_reg_predictions_tfrecords(net):
    total_loss = 0.0
    pred_tiles = list()
    gt_tiles = list()
    image_ids = list()  # contains an image id from which the tile was extracted

    num_examples = 0
    num_iterations = 0
    print 'Run until epoch ends'
    with tqdm(desc='predict') as pbar:
        while True:
            feed_dict = {net.is_training_pl: False}
            try:
                loss_value, obj_counts, gt_counts, batch_image_ids = net.sess.run(
                    [net.total_loss, net.obj_counts, net.input_gt_counts, net.input_image_ids],
                    feed_dict=feed_dict)
                num_iterations += 1
                pbar.update()
            except tf.errors.OutOfRangeError:
                print 'Epoch end.'
                break
            batch_image_ids = batch_image_ids[:, 0].tolist()
            image_ids.extend(batch_image_ids)
            total_loss += loss_value
            assert len(obj_counts.shape) == 2
            for tile_i in xrange(len(obj_counts)):
                pred_tiles.append(obj_counts[tile_i])
                gt_tiles.append(gt_counts[tile_i])
            num_examples += len(obj_counts)
    print 'Total num examples', num_examples
    print 'Total num iterations:', num_iterations
    cnt_tiles_per_image = list()
    checked_ids = set()
    cur_image_id = image_ids[0]
    cur_tiles_cnt = 0
    end_separator = None
    for id_i in image_ids + [end_separator]:
        if cur_image_id == id_i:
            cur_tiles_cnt += 1
        else:
            cnt_tiles_per_image.append(cur_tiles_cnt)
            checked_ids.add(cur_image_id)
            # sometimes first image can occur in the first and last batch
            assert id_i not in checked_ids, \
                'Wrong order of samples in test! {} was already checked'.format(id_i)
            cur_image_id = id_i
            cur_tiles_cnt = 1

    pred_tiles = np.asarray(pred_tiles)
    gt_tiles = np.asarray(gt_tiles)
    total_loss /= num_iterations
    print 'total_loss=', total_loss
    return pred_tiles, gt_tiles, total_loss, cnt_tiles_per_image


def run_test_tfrecords_adaptive_epoch(net, summary_writer=None):
    pred_counts_tiles, gt_counts_tiles, total_loss, cnt_tiles_per_image = \
        calculate_reg_predictions_tfrecords(net)

    _, _, rmse = calc_test_metrics(0, pred_counts_tiles,
                      gt_counts_tiles, total_loss, cnt_tiles_per_image,
                      heatmap_in_loss_multiplier=1.0, summary_writer=summary_writer,
                      min_threshold=0)
    return rmse


if __name__ == '__main__':
    args = parse_args()
    print args
    num_images_to_load = args.num_img
    ###########################################################################
    # Prepare data
    #=============================
    np.random.seed(41)
    rescaled_suffix = '_rescaled-1'
    tfrecords_file = {'train': join(ROOT_DIR,
                                    'records/train_black_sc1.0_seed1993_vp{}_v{}{}.tfrecords'.format(
                                        args.val_part, args.coords_v,
                                        rescaled_suffix if args.rescaled else '')),
                      'val': join(ROOT_DIR,
                                  'records/val_black_sc1.0_seed1993_vp{}_v{}{}.tfrecords'.format(
                                      args.val_part, args.coords_v,
                                      rescaled_suffix if args.rescaled else ''))}
    # TODO: add val_test

    if args.dbg:
        print '====DEBUG MODE===='
        for key in tfrecords_file:
            tfrecords_file[key] = os.path.dirname(tfrecords_file[key]) + '/' + 'dbg_' + \
                                  os.path.basename(tfrecords_file[key])
    print tfrecords_file

    ###########################################################################
    # Train
    checkpoints_dir = join(ROOT_DIR, 'checkpoints', args.model + args.suf)
    with tf.Graph().as_default():
        with tf.variable_scope('input'):
            input_images, input_gt_counts, input_image_ids = read_train_batch(
                tfrecords_file[args.split],
                batch_size=args.batch_size,
                tile_size=args.tile_size,
                num_threads=1,
                preprocessing_fn=net_spec.preprocessing_fn_map[args.model],
                as_float=True,
                shuffle=False,
                augmentations=False,
                full_image_scale_aug=args.full_scale_aug,
                min_scale_augmentations=args.min_scale_aug,
                max_scale_augmentations=args.max_scale_aug,
                num_epochs=1,
                seed=args.aug_seed,
                name='batch_val'
            )
            input_heatmaps = None
            is_training_pl = tf.placeholder(tf.bool)

        net = SegPlusRegNet(input_images, input_heatmaps, input_gt_counts,
                             is_training_pl=is_training_pl,
                             input_image_ids=input_image_ids,
                             use_regression_loss=True,
                             use_segmentation_loss=False,
                             extra_fc_size=args.extra_fc,
                             net_name=args.model,
                             should_create_summaries=False,
                             global_pool=args.pool)

        config = tflearn.init_graph(num_cores=0, gpu_memory_fraction=None,
                                    log_device=False, seed=42)
        net.sess = tf.Session(config=config)

        net.sess.run(tf.local_variables_initializer())
        net.sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=net.sess, coord=coord)

        if args.snapshot_iter is not None:
            snapshot_path = join(checkpoints_dir, 'checkpoint-{}'.format(args.snapshot_iter))
        else:
            snapshot_path = tf.train.latest_checkpoint(checkpoints_dir)
        assert snapshot_path is not None
        print 'Evaluate on split="{}" the snapshot: {}'.format(args.split, snapshot_path)
        saver = tf.train.Saver()
        saver.restore(net.sess, snapshot_path)

        # TODO: use Tester to evaluate on val_test
        rmse = run_test_tfrecords_adaptive_epoch(net)
        print 'RMSE', rmse
        print args

        if not args.dbg:
            out_dir = join(ROOT_DIR, 'out_preds/val_seed1993_vp{}_coordsv{}'.format(args.val_part,
                                                                                     args.coords_v))
            if args.min_scale_aug is not None and args.max_scale_aug is not None:
                if args.full_scale_aug:
                    aug_str = '_fullscaleaug_{}-{}'.format(args.min_scale_aug,
                                                          args.max_scale_aug)
                else:
                    aug_str = '_scaleaug_{}-{}'.format(args.min_scale_aug,
                                                      args.max_scale_aug)
            else:
                aug_str = ''

            filepath = join(out_dir, args.model + args.suf + aug_str)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(filepath, 'a') as f:
                f.write('===========\n\n')
                f.write('{}\n'.format(pformat(args, width=160)))
                f.write('---\n')
                if args.rescaled:
                    f.write('tfrecords suffix: {}'.format(rescaled_suffix))
                f.write('{}\n'.format(pformat(rmse, width=100)))

        coord.request_stop()
        coord.join(threads)
