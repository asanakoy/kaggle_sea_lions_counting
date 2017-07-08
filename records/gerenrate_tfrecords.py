import glob
import numpy as np
import os
from os.path import join
import pandas as pd
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
from PIL import Image

from data_utils import ROOT_DIR
from data_utils import df_from_script_data
from data_utils import get_test_images_info_df
from data_utils import test_images_dir_map


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def dbg_images_info():
    """Just for debugging"""
    df = pd.DataFrame(index=range(4))
    df['image_path'] = ['/export/home/asanakoy/tmp/GMII_BAM_0276_1915.jpg',
                        '/export/home/asanakoy/tmp/mycat.jpg',
                        '/export/home/asanakoy/tmp/NeTIxUZgzXJL_pmPBBgRetKo7mT_ASkaozxyRhjBDH9_sRd62YiEiso7uhXgeg',
                        '/export/home/asanakoy/tmp/wallpaper.jpg']
    return df


def for_test(scale=0.5):
    assert scale in [1.0, 0.5, 'a', 'b']
    images_dir = test_images_dir_map[scale]
    print 'images dir:', test_images_dir_map

    out_dir = join(ROOT_DIR, 'records')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tfrecords_filename = join(out_dir, 'test_black_{}.tfrecords'.format(scale))
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    images_info = get_test_images_info_df(scale=scale)
    assert images_info.index.dtype == np.int64
    images_info.sort_index(inplace=True)

    for image_id in tqdm(images_info.index, total=len(images_info)):
        img_path = join(images_dir, '{}.jpg'.format(image_id))
        with open(img_path, 'r') as f:
            img_encoded = f.read()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_encoded': _bytes_feature(img_encoded),
            'image_id': _int64_feature(image_id)}))
        writer.write(example.SerializeToString())
    writer.close()


def for_train(val_part, coords_v=0, seed=1993, dbg=False,
              splits=('train', 'val'),
              scale_str='1.0',
              images_dir=join(ROOT_DIR, 'Train/black'),
              density_maps_dir=None,
              suf=''):
    if density_maps_dir is None:
        density_maps_dir = join(ROOT_DIR, 'Train/density_maps_v{}'.format(coords_v))
    out_dir = join(ROOT_DIR, 'records')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_img_ids = df_from_script_data(coords_v=coords_v).index
    if dbg:
        train_img_ids = train_img_ids[:20]
        print train_img_ids
    assert train_img_ids.dtype == np.int64
    rs = np.random.RandomState(seed)
    random_perm = rs.permutation(len(train_img_ids))
    num_val = int(len(random_perm) * val_part)
    print 'Number validation images:', num_val
    image_ids = dict()
    image_ids['val'] = train_img_ids[random_perm[:num_val]]
    image_ids['train'] = train_img_ids[random_perm[num_val:]]
    # print image_ids

    for key in splits:
        tfrecords_filename = join(out_dir, '{}{}_black_sc{}_seed{}_vp{}_v{}{}.tfrecords'.
                                  format('dbg_' if dbg else '',
                                         key, scale_str,
                                         seed, val_part, coords_v, suf))
        print tfrecords_filename
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)
        for image_id in tqdm(image_ids[key], total=len(image_ids[key]), desc=key):
            img_path = join(images_dir, '{}.jpg'.format(image_id))
            maps_path = join(density_maps_dir, '{}.npy'.format(image_id))
            maps = np.load(maps_path)
            maps = np.asarray(maps.transpose(1, 2, 0), dtype=np.float32) # make H x W x NUM_MAPS

            with open(img_path, 'r') as f:
                img_encoded = f.read()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_encoded': _bytes_feature(img_encoded),
                'density_map': _bytes_feature(maps.tostring()),
                'image_id': _int64_feature(image_id)}))
            writer.write(example.SerializeToString())
        writer.close()


def create_dbg_records():
    out_dir = join(ROOT_DIR, 'records')
    tfrecords_filename = join(out_dir, 'dbg_1_600x600.tfrecords')
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    maps = np.zeros((150, 150, 5), dtype=np.float32)
    img_path = '/export/home/asanakoy/tmp/mycat_600x600.jpg'
    with open(img_path, 'r') as f:
        img_encoded = f.read()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_encoded': _bytes_feature(img_encoded),
        'density_map': _bytes_feature(maps.tostring()),
        'image_id': _int64_feature(0)}))
    writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    # for_train(val_part=0.1, coords_v=0, dbg=False, splits=('val',),
    #           images_dir=join(ROOT_DIR, 'val/val_avg_size_seed1993_vp0.1_coordsv0'),
    #           density_maps_dir=join(ROOT_DIR, 'val/val_avg_size_seed1993_vp0.1_coordsv0/density_maps'),
    #           suf='_rescaled-1')
    # for_train(val_part=0.1, coords_v=0, dbg=False,
    #           images_dir=join(ROOT_DIR, 'Train/black_0.5'),
    #           scale_str='0.5')
    create_dbg_records()
    for_train(val_part=0.1, coords_v=0, dbg=False, splits=('val', 'train'))
    for_train(val_part=0.1, coords_v=0, dbg=True, splits=('val', 'train'))
    for_test(scale=0.5)

