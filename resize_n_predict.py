# Copyright (c) 2017 Artsiom Sanakoyeu
import os
from os.path import join

from data_utils import ROOT_DIR
from data_utils import get_test_images_info_df
from resize_images import downscale
from records import gerenrate_tfrecords


def resize(scale_name):
    test_dir = join(ROOT_DIR, 'Test/')
    test_resized_dir = join(ROOT_DIR, 'Test/black_{}'.format(scale_name))

    if not os.path.exists(test_resized_dir):
        os.makedirs(test_resized_dir)
    print 'Resizing {} -> {}'.format(test_dir, test_resized_dir)

    info_df = get_test_images_info_df(scale=1)
    info_df['size'] = info_df['height'].astype(int).apply(str) + 'x' + info_df['width'].astype(int).apply(str)
    print info_df['size'].value_counts()

    path_list = info_df[info_df['size'].isin(['3456x4608', '3456x5184'])].image_path
    downscale(path_list, test_resized_dir, factor=0.5)

    path_list = info_df[~info_df['size'].isin(['3456x4608', '3456x5184'])].image_path
    downscale(path_list, test_resized_dir, factor=0.7)


if __name__ == '__main__':
    scale_name = 'b'
    resize(scale_name)
    gerenrate_tfrecords.for_test(scale=scale_name)
