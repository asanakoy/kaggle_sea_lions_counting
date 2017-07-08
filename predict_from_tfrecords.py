# Copyright (c) 2017 Artsiom Sanakoyeu
import argparse
import tflearn
import sys

from data_utils import *
from net_config import *
from net_spec import SegPlusRegNet
from records.records import read_test_batch

from net_spec import preprocessing_fn_map


def parse_dir(value):
    if value:
        return join(ROOT_DIR, 'out_preds', value)
    else:
        return None


def parse_scale(value):
    try:
        value = float(value)
    except:
        pass
    return value


parser = argparse.ArgumentParser()
parser.add_argument('--suf', help='Dir suffix')
parser.add_argument('--model', default='small', help='Network model type.',
                    choices=['vgg_16', 'vgg_19', 'resnet_v2_50', 'resnet_v2_101',
                             'resnet_v2_152', 'inception_resnet_v2'])
parser.add_argument('--snapshot_iter', type=int, default='None',
                    help='Snapshot iteration to use for prediction. Last snapshot if None')

parser.add_argument('--tile_size', type=int, default=224,
                    help='part of the validation split')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')

parser.add_argument('--scale', type=parse_scale, default=1, choices=[1, 0.5, 'a', 'b'],
                    help='images scale to use for prediction')

parser.add_argument('--preds_out', default=None, type=parse_dir,
                    help='predicted maps output dir')

parser.add_argument('--ssd', action='store_true',
                    help='read from ssd?')

parser.add_argument('--preproc', default=None, choices=['vgg', 'inception'],
                    help='preprocessing type')

parser.add_argument('--extra_fc', type=int, default=0,
                    help='size of the extra fc layer. If 0 do not add extra layer.')

parser.add_argument('--pool', choices=['avg', 'sum'], default=None,
                    help='sum/avg pool of last base conv(pool) layer')

parser.add_argument('--img_pad_offset', type=int, default=None,
                    help='Number of rows/columns of zeros to add on top/left of the image. Pad to center if None.')


FLAGS = parser.parse_args(sys.argv[1:])


def assemble_predictions_per_image(class_counts, images_info):
    predict_df = pd.DataFrame(index=images_info.index, columns=CLASS_NAMES, dtype=np.float32)
    prev_pos = 0
    for image_id, num_tiles in images_info['num_tiles'].iteritems():
        predict_df.loc[image_id, CLASS_NAMES] = \
            class_counts[prev_pos:prev_pos + num_tiles, :].sum(axis=0)
        prev_pos += num_tiles
    assert prev_pos == images_info.num_tiles.sum()
    return predict_df


def run_prediction(net, images_info, batch_size):
    assert 'num_tiles' in images_info.columns
    assert 'image_path' in images_info.columns
    
    number_tiles_total = images_info['num_tiles'].sum()
    print 'Number of tiles in this block:', number_tiles_total
    class_counts_per_tile = np.zeros((number_tiles_total, LionClasses.NUM_CLASSES), dtype=np.float32)

    for batch_start in tqdm(range(0, number_tiles_total, batch_size), desc='feed-fwd'):
        feed_dict = {net.is_training_pl: False}
        try:
            class_counts_per_tile[batch_start:batch_start + batch_size, :] = \
                np.clip(net.sess.run(net.obj_counts, feed_dict=feed_dict), 0, 1e6)
        except tf.errors.OutOfRangeError:
            print 'Finished.'
            break

    predict_df = assemble_predictions_per_image(class_counts_per_tile, images_info)
    return predict_df, class_counts_per_tile


def predict(net, images_info,
            tile_size=224, batch_size=12,
            predictions_output_dir=None,
            image_pad_offset=None):
    images_info['num_tiles'] = calc_number_of_tiles(images_info, tile_size,
                                                    image_pad_offset=image_pad_offset)

    predict_df, tile_predictions = run_prediction(net, images_info, batch_size)
    predict_df.index.name = 'test_id'
    
    if predictions_output_dir is not None:
        print 'Saving predicitons for all tiles'
        np.save(join(predictions_output_dir, 'all_tile_predictions.npy'), tile_predictions)
        images_info.to_hdf(join(predictions_output_dir, 'images_info.hdf5'), 'df', mode='w')
    assert np.all(predict_df.index == images_info.index)
    return predict_df


BLACK_OUT_TRAIN_DIR = join(ROOT_DIR, 'Train/black')


def get_splits(train_from_script_df, test_dir=join(ROOT_DIR, 'Test')):
    train_img_ids = train_from_script_df.index
    # train_img_ids = train_img_ids[:100]

    rs = np.random.RandomState(1993)
    val_part = 0.2
    random_perm = rs.permutation(len(train_img_ids))
    num_val = int(len(random_perm) * val_part)
    val_image_ids = train_img_ids[random_perm[:num_val]]
    train_img_ids = train_img_ids[random_perm[num_val:]]

    test_img_paths = glob.glob(join(test_dir, '*jpg'))
    test_img_paths.sort()
    test_img_ids = map(lambda x: int(os.path.basename(x)[:-4]), test_img_paths)
    test_img_ids.sort()
    return train_img_ids, val_image_ids, test_img_ids


def main():
    print FLAGS
    PREDICTIONS_DIR = join(ROOT_DIR, 'predictions')
    if not os.path.exists(PREDICTIONS_DIR):
        os.makedirs(PREDICTIONS_DIR)
    preds_name = FLAGS.model + FLAGS.suf + '_scale{scale:}{img_pad_offset:}it{it:}'.format(
        scale=FLAGS.scale,
        img_pad_offset='_pad{}'.format(FLAGS.img_pad_offset) if FLAGS.img_pad_offset is not None else '',
        it=str(FLAGS.snapshot_iter / 1000) + 'k' if FLAGS.snapshot_iter % 1000 == 0 else FLAGS.snapshot_iter)

    if FLAGS.preds_out is not None:
        preds_out_dir = FLAGS.preds_out
    else:
        preds_out_dir = join(ROOT_DIR, 'out_preds', 'test_' + preds_name)
    df_output_path = join(preds_out_dir, 'test_preds_df_' + preds_name + '.csv')

    if FLAGS.ssd:
        raise NotImplemented()
    else:
        data_dir = ROOT_DIR
    print 'Test images dir:', data_dir

    ###########################################################################
    # Prepare data
    np.random.seed(41)
    train_from_script_df = df_from_script_data()
    train_ids, val_ids, test_ids = get_splits(train_from_script_df)
    
    images_info = get_test_images_info_df(scale=FLAGS.scale)
    image_ids = test_ids
    assert np.all(images_info.index == image_ids)
    should_calc_scores = False
    # images_info = images_info[:10]

    ###########################################################################
    # Predict
    with tf.Graph().as_default():
        if FLAGS.preproc is not None:
            preproc_fn = preprocessing_fn_map[FLAGS.preproc]
        else:
            preproc_fn = preprocessing_fn_map[FLAGS.model]
        images_tensor, img_ids_tensor = read_test_batch(join(data_dir,
                                             'records/test_black_{}.tfrecords'.format(FLAGS.scale)),
                                            num_threads=1,
                                            batch_size=FLAGS.batch_size, tile_size=FLAGS.tile_size,
                                            image_pad_offset=FLAGS.img_pad_offset,
                                            preprocessing_fn=preproc_fn)
        net = SegPlusRegNet(images_tensor, None, None,
                            use_regression_loss=False,
                            use_segmentation_loss=False,
                            net_name=FLAGS.model,
                            should_create_summaries=False,
                            extra_fc_size=FLAGS.extra_fc,
                            global_pool=FLAGS.pool)

        config = tflearn.init_graph(num_cores=0, gpu_memory_fraction=None,
                                    log_device=False, seed=41)
        net.sess = tf.Session(config=config)
        net.sess.run(tf.local_variables_initializer())
        net.sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=net.sess, coord=coord)

        checkpoints_dir = join(ROOT_DIR, 'checkpoints', FLAGS.model + FLAGS.suf)
        if not os.path.exists(checkpoints_dir):
            raise ValueError('Dir {} doesn\'t exist!'.format(checkpoints_dir))
        if FLAGS.snapshot_iter is not None:
            snapshot_path = join(checkpoints_dir, 'checkpoint-{}'.format(FLAGS.snapshot_iter))
        else:
            snapshot_path = tf.train.latest_checkpoint(checkpoints_dir)
        saver = tf.train.Saver()
        saver.restore(net.sess, snapshot_path)
        print 'Restored from {}'.format(snapshot_path)
        if not os.path.exists(preds_out_dir):
            os.makedirs(preds_out_dir)
        start_time = time.time()
        predict_df = predict(net, images_info,
                             tile_size=FLAGS.tile_size,
                             batch_size=FLAGS.batch_size,
                             predictions_output_dir=preds_out_dir,
                             image_pad_offset=FLAGS.img_pad_offset)
        coord.request_stop()
        coord.join(threads)
        duration = time.time() - start_time
        print 'Elapsed time: {}h {}m'.format(int(duration) / 60**2, int(duration) / 60)
        print predict_df.head()
        print predict_df.describe()
        predict_df[CLASS_NAMES].to_csv(df_output_path)
        print 'Saved preds df to {}'.format(df_output_path)

        # TODO: create submit file
        submission_df = np.round(predict_df[CLASS_NAMES]).astype(int)
        submission_df = np.clip(submission_df, 0, 1e6)
        submission_df.index.name = 'test_id'
        submission_df.to_csv(join(ROOT_DIR, 'predictions/submission_{}.csv'.format(preds_name)))

        if should_calc_scores:
            evaluate_df(train_from_script_df, predict_df[CLASS_NAMES],
                        check_indices=False)


if __name__ == '__main__':
    main()
