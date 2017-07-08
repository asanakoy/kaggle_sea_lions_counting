# Copyright (c) 2017 Artsiom Sanakoyeu
from collections import defaultdict
import os
from os.path import join
import gc
import glob
import json
from math import floor
from math import ceil
import image_slicer
import pandas as pd
from pandas.core.base import FrozenList
from PIL import Image
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

from config import ROOT_DIR


class LionClasses:
    ADULT_MALES = 0
    SUBADULT_MALES = 1
    ADULT_FEMALES = 2
    JUVENILES = 3
    PUPS = 4
    NUM_CLASSES = 5


CLASS_NAMES = FrozenList(['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups'])


def coords_df_to_dict(coords_df_):
    images_info = defaultdict(list)
    for row in coords_df_.itertuples():
        item = dict(cls=row.cls, row=row.row, col=row.col)
        images_info[row.tid].append(item)
    return images_info


def hardcode_coords(coords_df_):
    coords_df_ = coords_df_.copy()
    coords_df_.loc[(coords_df_['tid'] == 691) & (coords_df_['cls'] == 0), 'row'] = 105
    coords_df_.loc[(coords_df_['tid'] == 691) & (coords_df_['cls'] == 0), 'col'] = 1550
    return coords_df_


def create_coords_df(coords_v=0):
    if coords_v not in [0, 1]:
        raise ValueError('Wrong coords version')
    filenames = {0: 'data/coords.csv',
                 1: 'data/coords_v1.csv'}
    coords_df = pd.read_csv(join(ROOT_DIR, filenames[coords_v]))
    coords_df = hardcode_coords(coords_df)
    images_info = coords_df_to_dict(coords_df)
    return images_info, coords_df


def create_train_images_info_df(images_info):
    train_imgs_df = get_train_images_info_df(scale=1)
    #################################################################

    images_with_coords = list(
        set(images_info.keys()).intersection(set(train_imgs_df.index.values)))
    print 'Num images with obj coords:', len(images_with_coords)
    images_with_coords.sort()

    for image_id in images_with_coords:
        objects = images_info[image_id]
        if len(objects):
            coords = np.zeros((len(objects), 2), dtype=int)
            for obj_idx, obj in enumerate(objects):
                coords[obj_idx, :] = [obj['col'], obj['row']]
            dist_matrix = pairwise_distances(coords, metric='euclidean')
            if dist_matrix.shape[0] > 1:
                dist_vals = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]
            else:
                dist_vals = np.array([np.nan])
            train_imgs_df.at[image_id, 'max_nn_dist'] = dist_vals.max()
            train_imgs_df.at[image_id, 'min_nn_dist'] = dist_vals.min()
            train_imgs_df.at[image_id, 'mean_nn_dist'] = dist_vals.mean()
            train_imgs_df.at[image_id, 'std_nn_dist'] = dist_vals.std()
            train_imgs_df.at[image_id, 'num_obj'] = len(objects)
    return train_imgs_df


test_images_dir_map = {
    0.5: join(ROOT_DIR, 'Test/black_0.5'),
    1.0: join(ROOT_DIR, 'Test'),
    'a': join(ROOT_DIR, 'Test/black_a'),
    'b': join(ROOT_DIR, 'Test/black_b'),
}


def get_test_images_info_df(scale=1):
    if scale == 1:
        file_path = join(ROOT_DIR, 'data/test_images_info.hdf5')
    elif scale == 0.5:
        file_path = join(ROOT_DIR, 'data/test_images_0.5_info.hdf5')
    elif scale in ['a', 'b']:
        file_path = join(ROOT_DIR, 'data/test_images_{}_info.hdf5'.format(scale))
    else:
        raise ValueError('Unknown scale: {}'.format(scale))
    images_dir = test_images_dir_map[scale]
    if os.path.exists(file_path):
        imgs_df = pd.read_hdf(file_path)
    else:
        train_img_paths = glob.glob(join(images_dir, '*jpg'))
        print len(train_img_paths)
        imgs_df = pd.DataFrame(data=train_img_paths, columns=['image_path'])
        imgs_df.index = imgs_df['image_path'].apply(
            lambda x: int(os.path.basename(x)[:-4]))
        imgs_df.index.name = 'image_id'
        print imgs_df[:5]

        for path in tqdm(train_img_paths, desc='Reading img sizes'):
            image_id = int(os.path.basename(path)[:-4])
            img = Image.open(path)
            imgs_df.loc[image_id, 'height'] = img.size[1]
            imgs_df.loc[image_id, 'width'] = img.size[0]
        assert len(imgs_df), 'Empty df!'
        imgs_df.sort_index(inplace=True)
        imgs_df.to_hdf(file_path, 'df', mode='w')
    return imgs_df


def get_train_images_info_df(scale=1):
    if scale == 1:
        file_path = join(ROOT_DIR, 'data/train_images_1.0_info.hdf5')
        images_dir = join(ROOT_DIR, 'Train/black_out')
    elif scale == 0.5:
        file_path = join(ROOT_DIR, 'data/train_images_0.5_info.hdf5')
        raise NotImplementedError()
    else:
        raise ValueError('Unknown scale: {}'.format(scale))
    if os.path.exists(file_path):
        imgs_df = pd.read_hdf(file_path)
    else:
        train_img_paths = glob.glob(join(images_dir, '*jpg'))
        print len(train_img_paths)
        imgs_df = pd.DataFrame(data=train_img_paths, columns=['image_path'])
        imgs_df.index = imgs_df['image_path'].apply(
            lambda x: int(os.path.basename(x)[:-4]))
        imgs_df.index.name = 'image_id'
        print imgs_df[:5]

        for path in tqdm(train_img_paths, desc='Reading img sizes'):
            image_id = int(os.path.basename(path)[:-4])
            img = Image.open(path)
            imgs_df.loc[image_id, 'height'] = img.size[1]
            imgs_df.loc[image_id, 'width'] = img.size[0]
        imgs_df.sort_index(inplace=True)
        imgs_df.to_hdf(file_path, 'df', mode='w')
    return imgs_df


def df_from_script_data(train_df=None, coords_df=None, coords_v=0):
    """
    :param train_df:
    :param coords_df:
    :return: df of train images which have coords (6 of them don't have coords) with obj counts
    """
    if coords_v not in [0, 1]:
        raise ValueError('Wrong coords version {}'.format(coords_v))
    print '::get df from script data v{}'.format(coords_v)
    if coords_v == 0:
        filepath = join(ROOT_DIR, 'data/train_from_coords.csv')
    else:
        filepath = join(ROOT_DIR, 'data/train_from_coords_v{}.csv'.format(coords_v))

    if os.path.exists(filepath):
        train_from_script = pd.read_csv(filepath, index_col='train_id')
    else:
        if train_df is None:
            train_df = pd.read_csv(join(ROOT_DIR, 'data/train.csv'), index_col='train_id')
        if coords_df is None:
            images_info, coords_df = create_coords_df(coords_v=coords_v)

        train_from_script = pd.DataFrame(data=0, columns=train_df.columns,
                                         index=coords_df['tid'].unique(),
                                         dtype=int)

        for row in tqdm(coords_df.iterrows(), total=len(coords_df)):
            image_id = row[1]['tid']
            class_id = row[1]['cls']
            train_from_script.at[image_id, CLASS_NAMES[class_id]] += 1

        train_from_script.index.name = 'train_id'
        # train_from_script['number_of_animals'] = train_from_script.sum(axis=1)
        # train_img_paths = glob.glob(join(ROOT_DIR, 'Train/*jpg'))
        # image_ids_good = map(lambda x: int(os.path.basename(x)[:-4]), train_img_paths)
        mismatched_ids = pd.read_csv(join(ROOT_DIR, 'data/MismatchedTrainImages.txt'), index_col='train_id').index
        image_ids_good = train_df.index.difference(mismatched_ids)

        image_ids_withcoords = list(set(image_ids_good).intersection(set(train_from_script.index.values)))
        train_from_script = train_from_script.loc[image_ids_withcoords]
        train_from_script.sort_index(inplace=True)
        train_from_script.to_csv(filepath)
    return train_from_script


def transform_images(images, inplace=False):
    """
    Transform an image or ndarray of images before feeding to CNN
    Args:
        images: 3d ndarray with one image or 4d ndarray with batch of images
    """
    assert images.dtype == np.float32
    if not inplace:
        images = images.copy()
    images /= 255
    images -= 0.5
    images *= 2.0
    return images


def get_data(train_images_dir, density_maps_dir, train_img_ids,
             tile_size=None,
             heatmap_tile_size=None,
             heatmaps_downsample_after_load_factor=1,
             min_count_threshold=-1.0,
             use_heatmaps=True,
             mock_heatmaps=True,
             should_transform_images=True,
             is_segmentation=False,
             return_tile_ids=False,
             pad_value=128):
    """Get tiles and heatmaps

    Return:
        images: (ndarray) tiles
        density_maps: (ndarray) gt density maps for tiles
        cnt_tiles_per_image: (list of ints) number of tiles used per image
    """
    print 'use_heatmaps={}, mosck_heatmaps={}'.format(use_heatmaps, mock_heatmaps)
    assert heatmaps_downsample_after_load_factor >= 1
    images = list()
    density_maps = list()
    cnt_tiles_per_image = list()
    # ids of the tiles selected from each image
    tile_ids_per_image = list()
    for image_id in tqdm(train_img_ids, desc='Load images'):
        img = imread(join(train_images_dir, '{}.jpg'.format(image_id)))
        if use_heatmaps:
            maps = np.load(join(density_maps_dir, '{}.npy'.format(image_id)))
            maps = maps.transpose(1, 2, 0)  # make H x W x NUM_MAPS
            if heatmaps_downsample_after_load_factor > 1:
                factor = 1. / heatmaps_downsample_after_load_factor
                maps = np.concatenate([imresize(maps[:, :, :3], factor, interp='nearest'),
                                       imresize(maps[:, :, 2:], factor,
                                       interp='nearest')[:, :, 1:]], axis=2)
            if is_segmentation:
                maps_sum = np.sum(maps, axis=2)
                assert np.all(maps_sum <= 1)
                labels = np.zeros(maps.shape[:2], dtype=np.int32)
                for channel in xrange(5):
                    labels += maps[:, :, channel] * (channel + 1)
                assert np.min(labels) == 0 and np.max(labels) <= 5
                maps = labels
                assert len(maps.shape) == 2

        if tile_size:
            img_tiles = slice_image_fixed_tiles(img, tile_size, pad_value=pad_value)
            if use_heatmaps:
                maps_tiles = slice_image_fixed_tiles(maps, heatmap_tile_size,
                                                     pad_value=0.0)
                assert len(maps_tiles) == len(img_tiles), \
                    '{] != {}'.format(len(maps_tiles), len(img_tiles))
            cnt_tiles = 0
            cur_tile_ids = list()
            if use_heatmaps:
                for tile_i in xrange(len(maps_tiles)):
                    if is_segmentation:
                        tile_heatmap_sum = (maps_tiles[tile_i] > 0).sum()
                    else:
                        tile_heatmap_sum = maps_tiles[tile_i].sum()
                    assert tile_heatmap_sum >= 0.0, 'tile_heatmap_sum={}'.format(tile_heatmap_sum)
                    if tile_heatmap_sum >= min_count_threshold:
                        images.append(img_tiles[tile_i])
                        density_maps.append(maps_tiles[tile_i])
                        cnt_tiles += 1
                        cur_tile_ids.append(tile_i)
            else:
                images.extend(img_tiles)
                cnt_tiles = len(img_tiles)
                cur_tile_ids = range(len(img_tiles))
            cnt_tiles_per_image.append(cnt_tiles)
            tile_ids_per_image.append(cur_tile_ids)
        else:
            images.append(img)
            if use_heatmaps:
                density_maps.append(maps)
            cnt_tiles_per_image.append(1)
            tile_ids_per_image.append([0])

    images = np.asarray(images, dtype=np.float32)
    if use_heatmaps:
        density_maps = np.asarray(density_maps)
    else:
        if mock_heatmaps:
            if is_segmentation:
                density_maps = np.zeros((len(images),
                                         heatmap_tile_size,
                                         heatmap_tile_size), dtype=np.int32)
            else:
                density_maps = np.zeros((len(images),
                                         heatmap_tile_size,
                                         heatmap_tile_size, 5), dtype=np.float16)
        else:
            density_maps = None

    if is_segmentation and use_heatmaps:
        assert density_maps.shape == images.shape[:-1], \
            'density_maps.shape={}'.format(density_maps.shape)
    if should_transform_images:
        transform_images(images, inplace=True)
    assert len(images) == np.sum(cnt_tiles_per_image), \
        '{} != {}'.format(len(images), np.sum(cnt_tiles_per_image))
    print 'Total number of tiles:', len(images)
    gc.collect()
    if return_tile_ids:
        return images, density_maps, cnt_tiles_per_image, tile_ids_per_image
    else:
        return images, density_maps, cnt_tiles_per_image


def get_gt_counts(density_maps_dir, tile_size, image_ids, tile_ids_per_image):
    gt_counts = list()
    for image_id, cur_img_tile_ids in \
            tqdm(zip(image_ids, tile_ids_per_image), desc='Counting objects'):
        maps = np.load(join(density_maps_dir, '{}.npy'.format(image_id)))
        maps = maps.transpose(1, 2, 0)  # make H x W x NUM_MAPS
        maps_tiles = slice_image_fixed_tiles(maps, tile_size, pad_value=0.0)
        for tile_i in xrange(len(maps_tiles)):
            if tile_i not in cur_img_tile_ids:
                continue
            obj_counts = maps_tiles[tile_i].sum(axis=(0, 1))
            assert obj_counts.shape[0] == 5
            assert np.all(obj_counts >= 0.0), 'obj_counts={}'.format(obj_counts)
            gt_counts.append(obj_counts)
    gt_counts = np.asarray(gt_counts, dtype=np.float32)
    assert gt_counts.shape[1] == 5, gt_counts.shape
    return gt_counts


def slice_image(full_image, number_tiles):
    """
    Split an image into a specified number of tiles.

    Args:
       filename (str):  The filename of the image to split.
       number_tiles (int):  The number of tiles required.

    Returns:
        Tuple of :class:`Tile` instances.
    """
    im_w, im_h = full_image.shape[1], full_image.shape[0]
    columns, rows = image_slicer.calc_columns_rows(number_tiles)
    tile_w, tile_h = int(floor(im_w / columns)), int(floor(im_h / rows))
    # print 'Tile w, h =', tile_w, tile_h

    total_slices_h = range(0, im_h - rows, tile_h)[-1] + tile_h
    total_slices_w = range(0, im_w - columns, tile_w)[-1] + tile_w
    # assert total_slices_h == full_image.shape[0], '{} != {}'.format(total_slices_h,
    #                                                                 full_image.shape[0])
    # assert total_slices_w == full_image.shape[1], '{} != {}'.format(total_slices_w,
    #                                                                 full_image.shape[1])

    tiles = []
    for pos_y in range(0, im_h - rows, tile_h):  # -rows for rounding error.
        for pos_x in range(0, im_w - columns, tile_w):  # as above.
            area = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            tile = full_image[pos_y:pos_y + tile_h, pos_x: pos_x + tile_w]
            tiles.append(tile)
    return tiles


def calc_number_of_tiles(images_info, tile_size, image_pad_offset=None):
    num_tiles = np.zeros(len(images_info), dtype=int)
    for i, row in enumerate(images_info.itertuples()):
        num_tiles[i] = get_number_of_tiles(row.height, row.width, tile_size,
                                           image_pad_offset=image_pad_offset)
    return num_tiles


def get_number_of_tiles(im_height, im_width, tile_size, image_pad_offset=None):
    if image_pad_offset is None:
        image_pad_offset = 0
    columns = int(ceil(float(im_width + image_pad_offset) / tile_size))
    rows = int(ceil(float(im_height + image_pad_offset) / tile_size))
    return rows * columns


def slice_image_fixed_tiles(full_image, tile_size, pad_value=128):
    """
    Split an image into a number of square tiles of specific size.

    Args:
       full_image (ndarray):  The filename of the image to split.
       tile_size (int):  The size of the side of the tile
       pad_value
    Returns:
        List of tiles
    """
    if not isinstance(tile_size, int):
        raise ValueError('tile_size must be int')
    columns = int(ceil(float(full_image.shape[1]) / tile_size))
    rows = int(ceil(float(full_image.shape[0]) / tile_size))

    tile_w, tile_h = tile_size, tile_size

    padded_image_size = (tile_h * rows, tile_w * columns)

    padded_image = np.ones(padded_image_size + full_image.shape[2:], dtype=full_image.dtype) * pad_value
    padded_image[:full_image.shape[0], :full_image.shape[1], ...] = full_image

    # print 'Tile w, h =', tile_w, tile_h
    assert padded_image.shape[:2] == padded_image_size
    im_h = padded_image.shape[0]
    im_w = padded_image.shape[1]

    num_tiles = columns * rows
    tile_shape = list(padded_image.shape)
    tile_shape[0] = tile_h
    tile_shape[1] = tile_w
    tile_shape = tuple(tile_shape)

    tiles = np.zeros((num_tiles,) + tile_shape, dtype=full_image.dtype)
    cur_tile = 0
    for pos_y in range(0, im_h, tile_h):  # -rows for rounding error.
        for pos_x in range(0, im_w, tile_w):  # as above.
            tiles[cur_tile, ...] = padded_image[pos_y:pos_y + tile_h, pos_x: pos_x + tile_w]
            cur_tile += 1
    assert len(tiles) == columns * rows, 'wrong number of tiles'
    return tiles


def evaluate_total_cnt_rmse_df(gt_df, preds_df, check_indices=True, name=''):
    preds_df.sort_index(inplace=True)
    gt_df.sort_index(inplace=True)

    ordered_class_names = CLASS_NAMES
    assert preds_df.columns.tolist() == ordered_class_names
    assert preds_df.columns.tolist() == gt_df.columns.tolist()

    if check_indices and preds_df.index.values.tolist() != gt_df.index.values.tolist():
        raise ValueError('Indices are different')

    gt_df = gt_df.loc[preds_df.index].copy()
    assert not pd.isnull(gt_df).values.sum()
    assert not pd.isnull(preds_df).values.sum()

    diff = preds_df.values.sum(axis=1) - gt_df.values.sum(axis=1)

    rmse = np.sqrt(np.mean(diff ** 2, axis=0))
    print '{}::total count RMSE: {}'.format(name, rmse.mean())
    return rmse


def evaluate_df(gt_df, preds_df, check_indices=True, ae=False, name='', is_3_classes=False):
    """
    Calculate RMSE between 2 datasframes
    Args:
        gt_df: ground truth counts
        preds_df: predicted counts
        check_indices:
        ae: calculate absolute error?
        name: name to print
        is_3_classes: use only 3 classes: MALES, FEMALES and PUPS instead of 5

    Returns: rmse

    """
    preds_df = preds_df.copy()
    gt_df = gt_df.copy()
    preds_df.sort_index(inplace=True)
    gt_df.sort_index(inplace=True)

    if is_3_classes:
        # as we easily confuse males and subadults, females and juveniles -> make 3 classes only
        for df in [gt_df, preds_df]:
            df['adult_males'] += df['subadult_males']
            df['subadult_males'] = 0
            df['adult_females'] += df['juveniles']
            df['juveniles'] = 0

    ordered_class_names = CLASS_NAMES
    assert preds_df.columns.tolist() == ordered_class_names
    assert preds_df.columns.tolist() == gt_df.columns.tolist()

    if check_indices and preds_df.index.values.tolist() != gt_df.index.values.tolist():
        raise ValueError('Indices are different')

    gt_df = gt_df.loc[preds_df.index].copy()
    assert not pd.isnull(gt_df).values.sum()
    assert not pd.isnull(preds_df).values.sum()

    diff = preds_df.values - gt_df.values

    if ae:
        absolute_error = np.mean(np.abs(diff), axis=0)
        print 'average AE:', absolute_error.mean()
        print 'AE per class:'
        for class_name, error in zip(ordered_class_names, absolute_error):
            print '  {}: {}'.format(class_name, error)
    else:
        absolute_error = None

    rmse = np.sqrt(np.mean(diff ** 2, axis=0))
    print '{}::average RMSE: {}'.format(name, rmse.mean())
    print '{}::RMSE per class:'.format(name)
    for class_name, error in zip(ordered_class_names, rmse):
        print '  {}: {}'.format(class_name, error)

    if ae:
        return rmse, absolute_error
    else:
        return rmse
