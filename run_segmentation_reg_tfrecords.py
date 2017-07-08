# Copyright (c) 2017 Artsiom Sanakoyeu
import argparse
import itertools
import numpy as np
import os
from os.path import join
import pandas as pd
from pprint import pprint
import json
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn
from tqdm import tqdm

from batch_generator import BatchGenerator
from records.records import read_train_batch
from data_utils import ROOT_DIR
from data_utils import CLASS_NAMES
from data_utils import df_from_script_data
from data_utils import get_train_images_info_df
from data_utils import get_test_images_info_df
from data_utils import calc_number_of_tiles
from data_utils import evaluate_df
from data_utils import evaluate_total_cnt_rmse_df
from net_config import run_training
from net_config import training_convnet
from net_config import calculate_reg_predictions
from net_config import calc_test_metrics
from net_config import create_sumamry
import net_spec
from net_spec import SegPlusRegNet
from net_spec import get_fixed_vars
from net_spec import imagenet_snapshots_map


def parse_class_weights(value):
    splits = value.split(',')
    if len(splits) != 6:
        raise ValueError('Must be 6 class weights')
    class_weights = map(lambda x: float(x.strip()), splits)
    return class_weights


def parse_bool(value):
    return bool(int(value))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suf', help='Checkpoints dir suffix.', default='')
    parser.add_argument('--model', default='small', help='Network model type.',
                        choices=['vgg_16', 'vgg_19', 'resnet_v2_50',
                                 'resnet_v2_101', 'resnet_v2_152', 'inception_resnet_v2'])
    parser.add_argument('--optimizer', default='adam', help='Optmizer type.')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate.')
    parser.add_argument('--init_snapshot', default=None, help='Initial snapshot.')
    parser.add_argument('--snapshot_iter', type=int, default=1000, help='Do snapsot every')
    parser.add_argument('--resume', action='store_true', help='Should restore evey variable from the snapshot (resume training)?')
    parser.add_argument('--test_iter', type=int, default=1000, help='Do test every')
    parser.add_argument('--max_iter', type=int, default=150000, help='max iterations')
    parser.add_argument('--val_part', type=float, default=0.1,
                        help='part of the validation split')
    parser.add_argument('--tile_size', type=int, default=299,
                        help='part of the validation split')

    parser.add_argument('--extra_fc', type=int, default=0,
                        help='size of the extra fc layer. If 0 do not add extra layer.')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('-segm', '--segmentation_loss_weight', type=float, default=0,
                        help='segmentation loss weight?')
    parser.add_argument('-w_dec', '--weight_decay', type=float, default=0.0,
                        help='weights regularization')
    parser.add_argument('--num_layers_to_fix', type=int, default=0,
                        help='number of layers to fix')

    parser.add_argument('--pool', choices=['avg', 'sum'], default=None,
                        help='sum/avg pool of last base conv(pool) layer')
    parser.add_argument('--aug', action='store_true',
                        help='Do augmentations on train?')
    parser.add_argument('--full_scale_aug', action='store_true',
                        help='Do scale aug on full images?')
    parser.add_argument('--min_scale_aug', type=float, default=None,
                        help='Min scale for scale augmentation (tile size multiplier if full_scale_aug)')
    parser.add_argument('--max_scale_aug', type=float, default=1.3,
                        help='Max scale for scale augmentation (tile size multiplier if full_scale_aug)')
    parser.add_argument('-rt', '--repeat_tiles', action='store_true',
                        help='Repeat tiles with lions?')

    parser.add_argument('--coords_v', type=int, default=0,
                        help='Version of the coordinates of the lions to use')

    parser.add_argument('--class_weights', type=parse_class_weights, default='1,1,1,1,1,1',
                        help='class weights')
    parser.add_argument('--dbg', action='store_true', help='Should debug (load only few images)?')
    args = parser.parse_args(sys.argv[1:])
    return args


def calculate_reg_predictions_tfrecords(net, batch_generator,
                                        global_step, batch_size):
    if global_step is None:
        global_step = net.sess.run(net.global_iter_counter)

    total_loss = 0.0
    assert batch_generator.cur_pos == 0
    if batch_generator.val_epoch_size is None:
        raise ValueError('batch_generator must know an epoch size!')
    pred_tiles = list()
    gt_tiles = list()
    image_ids = list()  # contains an image id from which the tile was extracted

    num_iterations = int(np.ceil(batch_generator.val_epoch_size / float(batch_size)))
    num_examples = 0
    is_last_batch = False
    for _ in tqdm(xrange(num_iterations)):
        feed_dict = batch_generator.next_feed_dict(net, batch_size=batch_size, phase='test')
        loss_value, obj_counts, gt_counts, batch_image_ids = net.sess.run(
            [net.total_loss, net.obj_counts, net.input_gt_counts, net.input_image_ids],
            feed_dict=feed_dict)
        batch_image_ids = batch_image_ids[:, 0].tolist()
        # we made one epoch nd came to the same image tiles
        cur_batch_size = batch_size
        if num_examples + batch_size >= batch_generator.val_epoch_size:
            cur_batch_size = batch_generator.val_epoch_size - num_examples
            print 'Found epoch end!'
            is_last_batch = True
        assert cur_batch_size, cur_batch_size
        assert is_last_batch or cur_batch_size == batch_size
        if cur_batch_size < batch_size:
            obj_counts = obj_counts[:cur_batch_size]
            gt_counts = gt_counts[:cur_batch_size]
            batch_image_ids = batch_image_ids[:cur_batch_size]

        image_ids.extend(batch_image_ids)
        total_loss += loss_value
        assert len(obj_counts.shape) == 2
        for tile_i in xrange(len(obj_counts)):
            pred_cnt = obj_counts[tile_i]
            gt_cnt = gt_counts[tile_i]
            pred_tiles.append(pred_cnt)
            gt_tiles.append(gt_cnt)
        num_examples += cur_batch_size
    print 'num_examples', num_examples
    assert num_examples == batch_generator.val_epoch_size
    if not is_last_batch:
        raise ValueError('Wrong max_iterations estimation or wrong test input order. Epoch was not finished!')
    check_tiles_order_in_epoch(image_ids)

    cnt_tiles_per_image = list()
    cur_image_id = image_ids[0]
    cur_tiles_cnt = 0
    end_separator = None
    for id_i in image_ids + [end_separator]:
        if cur_image_id == id_i:
            cur_tiles_cnt += 1
        else:
            cnt_tiles_per_image.append(cur_tiles_cnt)
            cur_image_id = id_i
            cur_tiles_cnt = 1
    # TODO: merge image at the beginning and at the end of the list

    pred_tiles = np.asarray(pred_tiles)
    gt_tiles = np.asarray(gt_tiles)
    total_loss /= num_iterations
    print 'total_loss=', total_loss

    return global_step, pred_tiles, gt_tiles, total_loss, cnt_tiles_per_image


def check_tiles_order_in_epoch(image_ids):
    """
    Check that every image occured once in the epoch.
    All tiles of the images should lie sequentially.
    Args:
        image_ids:

    Returns:

    """
    if not len(image_ids):
        raise ValueError('no tiles in epoch! Empty list of image ids!')
    image_ids = np.array(image_ids)
    # allow at the beginning and at the end to have the same image ids
    # (when image tile are split on 2 parts)
    pos = len(image_ids) - 1
    while pos >= 0 and image_ids[pos] == image_ids[0]:
        pos -= 1
    image_ids = image_ids[:pos + 1]
    if len(image_ids):
        checked_ids = set()
        cur_image_id = image_ids[0]
        prev_id = image_ids[0]
        for id_i in image_ids:
            if id_i != prev_id:
                if cur_image_id in checked_ids:
                    raise ValueError('{}\nWrong order of samples in test! {} was already checked.'.format(
                        image_ids.tolist()[:-5], id_i))
                checked_ids.add(id_i)
            prev_id = id_i


def get_predictions_for_images_from_tfrecords(net, batch_generator,
                                              global_step, batch_size):
    """
    Predict counts for every image in the tfrecords (1 epoch).
    Do not compute loss per tile.
    Args:
        net:
        batch_generator:
        global_step:
        batch_size:

    Returns:
        dataframe with predicted counts for every class
    """
    if global_step is None:
        global_step = net.sess.run(net.global_iter_counter)

    assert batch_generator.cur_pos == 0
    if batch_generator.val_epoch_size is None:
        raise ValueError('batch_generator must know an epoch size!')
    pred_tiles = list()
    image_ids = list()  # contains an image id from which the tile was extracted

    num_iterations = int(np.ceil(batch_generator.val_epoch_size / float(batch_size)))
    num_examples = 0
    is_last_batch = False
    for _ in tqdm(xrange(num_iterations)):
        feed_dict = batch_generator.next_feed_dict(net, batch_size=batch_size, phase='test')
        obj_counts, batch_image_ids = net.sess.run(
            [net.obj_counts, net.input_image_ids],
            feed_dict=feed_dict)
        batch_image_ids = batch_image_ids[:, 0].tolist()
        # we made one epoch nd came to the same image tiles
        cur_batch_size = batch_size
        if num_examples + batch_size >= batch_generator.val_epoch_size:
            cur_batch_size = batch_generator.val_epoch_size - num_examples
            print 'Found epoch end!'
            is_last_batch = True
        assert cur_batch_size, cur_batch_size
        assert is_last_batch or cur_batch_size == batch_size
        if cur_batch_size < batch_size:
            obj_counts = obj_counts[:cur_batch_size]
            batch_image_ids = batch_image_ids[:cur_batch_size]

        image_ids.extend(batch_image_ids)
        assert len(obj_counts.shape) == 2
        for tile_i in xrange(len(obj_counts)):
            pred_cnt = obj_counts[tile_i]
            pred_tiles.append(pred_cnt)
        num_examples += cur_batch_size
    print 'num_examples', num_examples
    assert num_examples == batch_generator.val_epoch_size
    if not is_last_batch:
        raise ValueError(
            'Wrong max_iterations estimation or wrong test input order. Epoch was not finished!')

    check_tiles_order_in_epoch(image_ids)

    preds_df = pd.DataFrame(index=np.unique(image_ids), data=0, columns=CLASS_NAMES, dtype=float)
    preds_df.index.name = 'image_id'
    assert len(image_ids) == len(pred_tiles)
    for id_i, cur_tile_preds in itertools.izip(image_ids, pred_tiles):
        preds_df.loc[id_i, CLASS_NAMES] += cur_tile_preds.clip(min=0)

    preds_df = np.round(preds_df).astype(int)
    preds_df.sort_index(inplace=True)
    return global_step, preds_df


class Tester:
    """
    Class for evalutiong model during training
    """

    def __call__(self, test_nets,
                   batch_generators,
                   global_step, batch_size,
                   cnt_tiles_per_image,
                   summary_writer=None):
        assert cnt_tiles_per_image is None
        for key, net in test_nets.iteritems():
            print 'Validation: {}'.format(key)
            if key in ['val', 'val_0.5']:
                global_step, pred_counts_tiles, gt_counts_tiles, total_loss, cnt_tiles_per_image = \
                    calculate_reg_predictions_tfrecords(net, batch_generators[key],
                                                        global_step=global_step, batch_size=batch_size)

                calc_test_metrics(global_step, pred_counts_tiles,
                                  gt_counts_tiles, total_loss, cnt_tiles_per_image,
                                  heatmap_in_loss_multiplier=1.0, summary_writer=summary_writer,
                                  min_threshold=0, calc_per_tile=False, split_name=key)


def get_epoch_size(val_part, tile_size, coords_v, dbg=False, seed=1993):
    train_img_ids = df_from_script_data(coords_v=coords_v).index
    if dbg:
        train_img_ids = train_img_ids[:20]
    assert train_img_ids.dtype == np.int64
    rs = np.random.RandomState(seed)
    random_perm = rs.permutation(len(train_img_ids))
    num_val = int(len(random_perm) * val_part)
    print 'Number validation images:', num_val
    image_ids = dict()
    image_ids['val'] = train_img_ids[random_perm[:num_val]]
    image_ids['train'] = train_img_ids[random_perm[num_val:]]

    images_info = get_train_images_info_df(scale=1.0)
    images_info['num_tiles'] = calc_number_of_tiles(images_info, tile_size)
    epoch_size = dict()
    for key in ['train', 'val']:
        epoch_size[key] = images_info.loc[image_ids[key]]['num_tiles'].sum()

    return epoch_size


if __name__ == '__main__':
    args = parse_args()
    print 'Args:', args
    use_segmentation_loss = args.segmentation_loss_weight > 0
    use_scale_aug = args.min_scale_aug is not None and args.max_scale_aug is not None
    assert not use_scale_aug or args.min_scale_aug <= args.max_scale_aug
    print 'use_segmentation_loss=', use_segmentation_loss
    print('Reserve GPU memory!')
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95))
    tmp_session = tf.Session(config=config)
    ###########################################################################
    # Prepare data
    np.random.seed(41)

    tfrecords_file = {'train': join(ROOT_DIR, 'records/train_black_sc1.0_seed1993_vp{}_v{}.tfrecords'.format(args.val_part, args.coords_v)),
                      'val': join(ROOT_DIR, 'records/val_black_sc1.0_seed1993_vp{}_v{}.tfrecords'.format(args.val_part, args.coords_v)),
                      'val_0.5': join(ROOT_DIR, 'records/val_black_sc1.0_seed1993_vp{}_v{}.tfrecords'.format(args.val_part, args.coords_v))}

    val_splits = ['val']
    if args.dbg:
        print '====DEBUG MODE===='
        for key in tfrecords_file:
            tfrecords_file[key] = os.path.dirname(tfrecords_file[key]) + '/' + 'dbg_' + \
                                  os.path.basename(tfrecords_file[key])
    print tfrecords_file

    ###########################################################################
    # Train
    checkpoints_dir = join(ROOT_DIR, 'checkpoints', '{}_tile{}{pool:}{extrafc:}{aug:}{scale_aug:}{repeat_tiles:}{suf:}_coordsv{coords_v:}'.format(args.model, args.tile_size,
                                                                           pool=('_{}pool'.format(args.pool) if args.pool else ''),
                                                                           extrafc=('_fc' + str(args.extra_fc)) if args.extra_fc else '',
                                                                           aug=('_aug2' if args.aug else ''),
                                                                           scale_aug=('_{}scaleaug-{}-{}'.format('full' if args.full_scale_aug else '',
                                                                                             args.min_scale_aug,
                                                                                             args.max_scale_aug) if use_scale_aug else ''),
                                                                           repeat_tiles=('' if args.repeat_tiles else ''),
                                                                           suf=(args.suf if args.suf else ''),
                                                                           coords_v=args.coords_v))
    print checkpoints_dir
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    seed = 1993
    with open(join(checkpoints_dir, 'seed.txt'), mode='a') as f:
        f.write('{}\n'.format(seed))
    with open(join(checkpoints_dir, 'args.json'), mode='w') as f:
        json.dump(vars(args), f, indent=1)

    print('Close tmp session')
    tmp_session.close()
    with tf.Graph().as_default():
        tf.summary.scalar('tile_size', args.tile_size)

        val_batch_generators = dict()
        val_tile_scales = {'val_0.5': 2.0, 'val': 1.0}
        for key in val_splits:
            val_batch_generators[key] = BatchGenerator(None, None)

            # do validation using 2 times smaller tiles (equivalent to having 2 times bigger images)
            effective_val_tile_size = int(args.tile_size * val_tile_scales[key])
            # if val_tile_scales[key] != 1.0:
            #     effective_val_tile_size -= effective_val_tile_size % 4
            # effective_val_tile_size = args.tile_size

            epoch_size_key = key
            if epoch_size_key == 'val_0.5':
                # val_0.5 essentially is val but with different tile_size
                epoch_size_key = 'val'
            val_batch_generators[key].val_epoch_size = get_epoch_size(args.val_part,
                                                                      effective_val_tile_size,
                                                                      args.coords_v,
                                                                      dbg=args.dbg)[epoch_size_key] # num tiles in epoch
        assert args.val_part == 0.1
        # just a quick hack for 0.1 val part
        tester = Tester()

        with tf.variable_scope('input'):
            input_images = dict()
            input_gt_counts = dict()
            input_image_ids = dict()
            for key in ['train'] + val_splits:
                input_images[key], input_gt_counts[key], input_image_ids[key] = read_train_batch(
                    tfrecords_file[key],
                    batch_size=args.batch_size,
                    tile_size=args.tile_size,
                    num_threads=4 if key == 'train' else 1,
                    preprocessing_fn=net_spec.preprocessing_fn_map[args.model],
                    as_float=True,
                    shuffle=(key == 'train'),
                    augmentations=args.aug if key == 'train' else False,
                    full_image_scale_aug=args.full_scale_aug if key == 'train' else True,
                    min_scale_augmentations=args.min_scale_aug if key == 'train' else val_tile_scales[key],
                    max_scale_augmentations=args.max_scale_aug if key == 'train' else val_tile_scales[key],
                    should_repeat_tiles=args.repeat_tiles if key == 'train' else False,
                    density_map_downsample_factor=4,
                    name='batch_' + key,
                    seed=seed
                )
            tf.summary.scalar('other/repeat_tiles', int(args.repeat_tiles))
            input_heatmaps = None
            is_training_pl = tf.placeholder(tf.bool)

        net = SegPlusRegNet(input_images['train'], input_heatmaps, input_gt_counts['train'],
                            input_image_ids=input_image_ids['train'],
                            is_training_pl=is_training_pl,
                            class_weights=args.class_weights,
                            use_regression_loss=True,
                            use_segmentation_loss=use_segmentation_loss,
                            segmentation_loss_weight=args.segmentation_loss_weight,
                            weight_decay=args.weight_decay,
                            num_layers_to_fix=args.num_layers_to_fix,
                            extra_fc_size=args.extra_fc,
                            net_name=args.model,
                            global_pool=args.pool)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            nets_test = dict()
            for key in val_splits:
                nets_test[key] = SegPlusRegNet(input_images[key], input_heatmaps, input_gt_counts[key],
                                     input_image_ids=input_image_ids[key],
                                     is_training_pl=is_training_pl,
                                     class_weights=args.class_weights,
                                     use_regression_loss=(key == 'val'),
                                     use_segmentation_loss=use_segmentation_loss,
                                     segmentation_loss_weight=args.segmentation_loss_weight,
                                     weight_decay=args.weight_decay,
                                     num_layers_to_fix=args.num_layers_to_fix,
                                     extra_fc_size=args.extra_fc,
                                     net_name=args.model,
                                     should_create_summaries=False,
                                     global_pool=args.pool)

        pprint(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        config = tflearn.init_graph(num_cores=0, gpu_memory_fraction=None,
                                    log_device=False, seed=seed)  # seed None before
        net.sess = tf.Session(config=config)
        for key in val_splits:
            nets_test[key].sess = net.sess

        fixed_vars, lower_vars = get_fixed_vars(args.model)
        lr = tf.constant(args.lr, dtype=tf.float32)
        train_op = training_convnet(net, net.total_loss, lr, optimizer_type=args.optimizer,
                                    fixed_vars=fixed_vars,
                                    lower_vars=lower_vars,
                                    trace_gradients=False)

        net.sess.run(tf.local_variables_initializer())
        net.sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=net.sess, coord=coord)

        snapshot_path = tf.train.latest_checkpoint(checkpoints_dir)
        if snapshot_path is not None:
            print 'Continue training from the snapshot: {}'.format(snapshot_path)
            saver = tf.train.Saver()
            saver.restore(net.sess, snapshot_path)
        elif args.init_snapshot:
            snapshot_path = args.init_snapshot
            print 'Init weights from the snapshot: {}'.format(snapshot_path)
            if not args.resume:
                init_fn = slim.assign_from_checkpoint_fn(
                        snapshot_path,
                        slim.get_model_variables(),
                        ignore_missing_vars=True)
                init_fn(net.sess)
            else:
                print 'Restore every variable from the snapshot (resume from the snapshot)'
                saver = tf.train.Saver()
                saver.restore(net.sess, snapshot_path)
        else:
            snapshot_path = imagenet_snapshots_map[args.model]
            print 'Init with Imagenet pretrained: {}'.format(snapshot_path)
            init_fn = slim.assign_from_checkpoint_fn(
                    snapshot_path,
                    slim.get_model_variables(),
                    ignore_missing_vars=True)
            init_fn(net.sess)

        # TODO: we need to fix exact epoch in val.tfrecord or reset reader every test ?
        run_training(net, BatchGenerator(None, None), train_op, net.total_loss,
                     tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None),
                     test_net=nets_test,
                     test_fn=tester,
                     val_batch_generator=val_batch_generators,
                     batch_size=args.batch_size, max_iter=args.max_iter,
                     snapshot_iter=args.snapshot_iter,
                     output_dir=checkpoints_dir,
                     test_step=args.test_iter,
                     do_not_summary=0,
                     summary_step=50,
                     val_cnt_tiles_per_image=None)
        coord.request_stop()
        coord.join(threads)
