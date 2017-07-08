# Copyright (c) 2017 Artsiom Sanakoyeu
import numpy as np
import os
from os.path import join
import pandas as pd
import tensorflow as tf

from data_utils import ROOT_DIR
from preprocessing.inception_preprocessing import apply_with_random_selector


def produce_tiles(image, tile_size, pad_offset=None):
    """
    Slice image into slice in tensorflow graph.
    Necessary padding on X and Y dimensions will be made to make them be multiple of the tile_size.
    Args:
        image:
        tile_size:
        pad_offset: offset of teh padding
        (automatic by default: place the original image in the center of the new padded image)

    Returns:

    """
    patch_height = patch_width = tile_size
    float_patch_height = float_patch_width = tf.to_float(tile_size)
    # resize image so that it's dimensions are dividable by patch_height and patch_width
    image_height = tf.cast(tf.shape(image)[0], dtype=tf.float32)
    image_width = tf.cast(tf.shape(image)[1], dtype=tf.float32)
    height = tf.to_int32(tf.ceil(image_height / float_patch_height) * float_patch_height)
    width = tf.to_int32(tf.ceil(image_width / float_patch_width) * float_patch_width)

    # make zero-padding
    if pad_offset is not None:
        height = tf.cond(tf.to_int32(image_height) + pad_offset <= height,
                         lambda: height,
                         lambda: tf.to_int32(tf.ceil((image_height + pad_offset) / float_patch_height) * float_patch_height))
        width = tf.cond(tf.to_int32(image_width) + pad_offset <= width,
                         lambda: width,
                         lambda: tf.to_int32(tf.ceil((image_width + pad_offset) / float_patch_width) * float_patch_width))

        image = tf.image.pad_to_bounding_box(
            image,
            pad_offset,
            pad_offset,
            height,
            width)
    else:
        image = tf.squeeze(tf.image.resize_image_with_crop_or_pad(image, height, width))

    num_rows = height // tf.to_int32(patch_height)
    num_cols = width // tf.to_int32(patch_width)

    # get slices along the 0-th axis
    image = tf.reshape(image, [num_rows, patch_height, width, -1])
    # h/patch_h, w, patch_h, c
    image = tf.transpose(image, [0, 2, 1, 3])
    # get slices along the 1-st axis
    # h/patch_h, w/patch_w, patch_w,patch_h, c
    image = tf.reshape(image, [num_rows, num_cols, patch_width, patch_height, -1])
    # num_patches, patch_w, patch_h, c
    image = tf.reshape(image, [num_rows * num_cols, patch_width, patch_height, -1])
    # num_patches, patch_h, patch_w, c
    return tf.transpose(image, [0, 2, 1, 3])


def read_and_decode_single_example(filename, num_epochs=None):
    filename_queue = tf.train.string_input_producer([filename],
                                                    shuffle=False,
                                                    num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_encoded': tf.VarLenFeature(tf.string),
            'image_id': tf.FixedLenFeature([], tf.int64)
        })
    image_id = tf.cast(features['image_id'], tf.int32)
    image_encoded = tf.sparse_tensor_to_dense(features['image_encoded'], default_value='\0')
    image_encoded = tf.squeeze(image_encoded)
    # image_encoded = tf.Print(image_encoded, [tf.shape(image_encoded)], 'encoded image shape:')
    image = tf.image.decode_image(image_encoded)
    # image = tf.Print(image, [tf.shape(image), image], 'decoded image shape:')

    return image, image_id


def read_test_batch(filename, batch_size=64, tile_size=224,
                    image_pad_offset=None, preprocessing_fn=None,
                    num_threads=2, as_float=True):
    image, image_id = read_and_decode_single_example(filename,
                                                     num_epochs=1)
    tiles = produce_tiles(image, tile_size, pad_offset=image_pad_offset)
    if as_float:
        tiles = tf.cast(tiles, dtype=tf.float32)
    if preprocessing_fn is not None:
        tiles = preprocessing_fn(tiles)

    num_tiles = tf.shape(tiles)[0]
    image_id_list = tf.reshape(tf.tile([image_id], [num_tiles]), [num_tiles, 1])

    images, ids = tf.train.batch([tiles, image_id_list], num_threads=num_threads,
                            enqueue_many=True,
                            shapes=[[tile_size, tile_size, 3], [1]],
                            batch_size=batch_size,
                            capacity=20 * batch_size,
                            allow_smaller_final_batch=True,
                            name='test_batch')
    return images, ids


def read_and_decode_single_train_example(filename, num_epochs=None,
                                         density_map_downsample_factor=4):
    print '::Create reader for {}'.format(filename)
    filename_queue = tf.train.string_input_producer([filename],
                                                    shuffle=False,
                                                    num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_encoded': tf.VarLenFeature(tf.string),
            'density_map': tf.VarLenFeature(tf.string),
            'image_id': tf.FixedLenFeature([], tf.int64)
        })
    image_id = tf.cast(features['image_id'], tf.int32)

    image_encoded = tf.sparse_tensor_to_dense(features['image_encoded'], default_value='\0')
    image_encoded = tf.squeeze(image_encoded)
    image = tf.image.decode_image(image_encoded)

    density_map = tf.sparse_tensor_to_dense(features['density_map'], default_value='\0')
    density_map = tf.squeeze(density_map)
    density_map = tf.decode_raw(density_map, tf.float32)
    map_channels = 5
    map_height = tf.shape(image)[0] // density_map_downsample_factor
    map_width = tf.shape(image)[1] // density_map_downsample_factor
    density_map = tf.reshape(density_map, shape=[map_height, map_width, map_channels])

    return image, density_map, image_id


def density_tiles_to_counts(density_map_tiles):
    counts = tf.reduce_sum(density_map_tiles, axis=[1, 2])
    tf.assert_equal(tf.shape(counts)[1], 5)
    return counts


def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = random_rotation(image)
    image = tf.image.flip_left_right(image)
    return image


def random_rotation(image):
    image = apply_with_random_selector(
            image,
            lambda x, k: tf.image.rot90(x, k),
            num_cases=4)
    return image


def random_crop(combined_image_map,
                tile_size,
                scale_range=(0.5, 1.0),
                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
      image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
      density_map: 3-D float Tensor of density maps of the image
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding box
        supplied.
      area_range: An optional list of `floats`. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
      scope: Optional scope for name_scope.
    Returns:
      A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'random_crop', [combined_image_map]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        # bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
        #                    dtype=tf.float32,
        #                    shape=[1, 1, 4])
        # sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        #     tf.shape(combined_image_map),
        #     bounding_boxes=bbox,
        #     min_object_covered=min_object_covered,
        #     aspect_ratio_range=(1.0, 1.0),
        #     area_range=area_range,
        #     max_attempts=max_attempts,
        #     use_image_if_no_bounding_boxes=True)
        # bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        # bbox_size = tf.Print(bbox_size, [bbox_size], 'bbox_size')
        # cropped_combined_image_map = tf.slice(combined_image_map, bbox_begin, bbox_size)
        img_size = tf.to_float(tf.shape(combined_image_map)[0])
        num_channels = tf.shape(combined_image_map)[-1]
        crop_size = tf.random_uniform([],
                                      minval=tf.to_int32(scale_range[0] * img_size),
                                      maxval=tf.to_int32(scale_range[1] * img_size),
                                      dtype=tf.int32)
        cropped_combined_image_map = tf.random_crop(combined_image_map,
                                                    [crop_size, crop_size, num_channels])

        image, density_map = tf.split(cropped_combined_image_map, [3, 5], axis=2)
        orig_counts = tf.reduce_sum(density_map, axis=[0, 1])
        with tf.control_dependencies([tf.assert_equal(tf.shape(orig_counts)[0], 5)]):
            orig_counts = tf.reshape(orig_counts, [1, 1, 5])
        resized_combined_image_map = tf.image.resize_images(cropped_combined_image_map, [tile_size, tile_size])
        resized_image, resized_density_map = tf.split(resized_combined_image_map, [3, 5], axis=2)
        cur_count = tf.reshape(tf.reduce_sum(resized_density_map, axis=[0, 1]), [1, 1, 5])

        corrected_resized_density_map = resized_density_map / tf.maximum(cur_count, 1e-6) * orig_counts
        corrected_combined_image_map = tf.concat([resized_image, corrected_resized_density_map], axis=2)
        return corrected_combined_image_map


def produce_scale_augmented_tiles(image, density_map, tile_size,
                                  min_scale_augmentations=0.8, max_scale_augmentations=1.2,
                                  preprocessing_fn=None,
                                  density_map_downsample_factor=4):

    effective_tile_size = tf.to_int32(tile_size * max_scale_augmentations)
    print 'effective_tile_size', effective_tile_size

    tiles = produce_tiles(image, effective_tile_size)
    density_map_tiles = produce_tiles(density_map,
                                      effective_tile_size // density_map_downsample_factor)
    tf.assert_equal(tf.shape(tiles)[0], tf.shape(density_map_tiles)[0])
    gt_counts_per_tile = density_tiles_to_counts(density_map_tiles)

    tiles = tf.cast(tiles, dtype=tf.float32)
    if preprocessing_fn is not None:
        tiles = preprocessing_fn(tiles)

    print 'Scale aug!'
    big_density_map_tiles = tf.image.resize_images(density_map_tiles, (effective_tile_size, effective_tile_size))
    cur_counts = tf.reshape(tf.reduce_sum(big_density_map_tiles, axis=[1, 2]),
                            [-1, 1, 1, 5])
    orig_counts = tf.reshape(gt_counts_per_tile, [-1, 1, 1, 5])
    big_density_map_tiles = big_density_map_tiles / tf.maximum(cur_counts,
                                                               1e-6) * orig_counts
    combined_img_map = tf.concat([tiles, big_density_map_tiles], axis=3)
    scale = float(min_scale_augmentations) / max_scale_augmentations
    combined_img_map_cropped = tf.map_fn(lambda x: random_crop(x, tile_size,
                                                               scale_range=(scale, 1.0)),
                                         combined_img_map)
    tiles, density_map_tiles = tf.split(combined_img_map_cropped, [3, 5], axis=3)
    # tiles = tf.Print(tiles, [tf.shape(tiles)], 'tiles shape', first_n=3)
    return tiles, density_map_tiles


def random_resize_image(image, density_map, min_scale_augmentations, max_scale_augmentations):
    scale = tf.random_uniform([],
                              minval=min_scale_augmentations,
                              maxval=max_scale_augmentations,
                              dtype=tf.float32)

    new_img_size = tf.to_int32(tf.to_float(tf.shape(image)[:2]) * scale)
    print image.get_shape().as_list()
    image = tf.image.resize_images(image, new_img_size)

    orig_counts = tf.reduce_sum(density_map, axis=[0, 1])
    with tf.control_dependencies([tf.assert_equal(tf.shape(orig_counts)[0], 5)]):
        orig_counts = tf.reshape(orig_counts, [1, 1, 5])

    new_density_mape_size = tf.to_int32(tf.shape(density_map)[:2] * scale)
    density_map = tf.image.resize_images(density_map, new_density_mape_size)

    new_count = tf.reshape(tf.reduce_sum(density_map, axis=[0, 1]), [1, 1, 5])
    density_map = density_map / tf.maximum(new_count, 1e-6) * orig_counts
    return image, density_map


def resize_density_map(density_map, new_size):
    orig_counts = tf.reduce_sum(density_map, axis=[0, 1])
    orig_counts = tf.reshape(orig_counts, [1, 1, 5])
    density_map = tf.image.resize_images(density_map, new_size)

    new_count = tf.reshape(tf.reduce_sum(density_map, axis=[0, 1]), [1, 1, 5])
    density_map = density_map / tf.maximum(new_count, 1e-6) * orig_counts
    return density_map


def repeat_tiles(tiles, gt_counts_per_tile):
    """
    Repeat tiles according to number objects on them
    Args:
        tiles:
        gt_counts_per_tile:

    Returns:

    """
    indices = list()
    total_counts = gt_counts_per_tile.sum(axis=1)
    assert len(tiles) == len(total_counts)
    for i in xrange(len(tiles)):
        num_repeats = 1
        if total_counts[i] > 0.8:
            num_repeats = int(np.ceil(np.log(total_counts[i]) / np.log(2)))
            num_repeats = max(3, num_repeats)
        indices.extend([i] * num_repeats)
    tiles = tiles[indices, ...]
    gt_counts_per_tile = gt_counts_per_tile[indices, ...]
    return tiles, gt_counts_per_tile


def read_train_batch(filename, batch_size=64, tile_size=224, preprocessing_fn=None,
                     num_threads=2, as_float=True, shuffle=False,
                     augmentations=False,
                     full_image_scale_aug=False,
                     min_scale_augmentations=None,
                     max_scale_augmentations=1.3,
                     density_map_downsample_factor=4,
                     should_repeat_tiles=False,
                     name=None, num_epochs=None,
                     seed=None):
    assert os.path.exists(filename), filename
    use_scale_aug = min_scale_augmentations is not None and max_scale_augmentations is not None
    image, density_map, image_id = read_and_decode_single_train_example(filename,
                                                                        num_epochs=num_epochs,
                                                                        density_map_downsample_factor=density_map_downsample_factor)

    if use_scale_aug and not full_image_scale_aug:
        if not as_float:
            raise ValueError('To do scale augmentations per tiel use_scale_aug must be True!')
        # TODO: use seed
        tiles, density_map_tiles = produce_scale_augmented_tiles(image, density_map, tile_size,
                                      min_scale_augmentations=min_scale_augmentations,
                                      max_scale_augmentations=max_scale_augmentations,
                                      preprocessing_fn=preprocessing_fn,
                                      density_map_downsample_factor=density_map_downsample_factor)
        print tiles, density_map_tiles
    else:
        if use_scale_aug and full_image_scale_aug:
            print 'Aug full images'
            scale = tf.random_uniform([],
                                      minval=min_scale_augmentations,
                                      maxval=max_scale_augmentations,
                                      dtype=tf.float32,
                                      seed=seed)
            new_tile_size = tf.to_int32(tile_size * scale)
            tiles = produce_tiles(image, new_tile_size)
            tiles = tf.Print(tiles, [scale, tf.shape(tiles)],
                                     'scale, tiles shape:', first_n=10)
            tiles = tf.image.resize_images(tiles, (tile_size, tile_size))
            density_map = resize_density_map(density_map, tf.shape(image)[:2])
            density_map_tiles = produce_tiles(density_map,
                                              new_tile_size)
        else:
            tiles = produce_tiles(image, tile_size)
            density_map_tiles = produce_tiles(density_map,
                                              tile_size // density_map_downsample_factor)
        if as_float:
            tiles = tf.cast(tiles, dtype=tf.float32)
        if preprocessing_fn is not None:
            tiles = preprocessing_fn(tiles)
    with tf.control_dependencies([tf.assert_equal(tf.shape(tiles)[0], tf.shape(density_map_tiles)[0],
                                                  message='the same nmber of tiles and map-tiles')]):
        gt_counts_per_tile = density_tiles_to_counts(density_map_tiles)

    if should_repeat_tiles:
        tiles, gt_counts_per_tile = tf.py_func(repeat_tiles, [tiles, gt_counts_per_tile], [tf.float32, tf.float32])

    if augmentations:
        tiles = tf.map_fn(augment_image, tiles)

    num_tiles = tf.shape(tiles)[0]
    image_id_list = tf.reshape(tf.tile([image_id], [num_tiles]), [num_tiles, 1])

    if shuffle:
        images, gt_counts, ids = tf.train.shuffle_batch([tiles, gt_counts_per_tile, image_id_list],
                                                   capacity=400 * 16,
                                                   min_after_dequeue=400 * 8,
                                                   num_threads=num_threads,
                                                   enqueue_many=True,
                                                   shapes=[[tile_size, tile_size, 3], [5], [1]],
                                                   batch_size=batch_size,
                                                   allow_smaller_final_batch=True,
                                                   name=name)
    else:
        images, gt_counts, ids = tf.train.batch([tiles, gt_counts_per_tile, image_id_list],
                                               capacity=400 * 3,
                                               num_threads=num_threads,
                                               enqueue_many=True,
                                               shapes=[[tile_size, tile_size, 3], [5], [1]],
                                               batch_size=batch_size,
                                               allow_smaller_final_batch=True,
                                               name=name)
    return images, gt_counts, ids


def test_test_batches():
    images, image_ids = read_test_batch(join(ROOT_DIR, 'records/test_gray_0.5.tfrecords'),
                                        tile_size=299,
                                        image_pad_offset=30,
                                        num_threads=1,
                                        as_float=False)

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    import matplotlib.pyplot as plt

    cnt = 0
    for i in xrange(200):
        try:
            out, ids = sess.run([images, image_ids])
        except tf.errors.OutOfRangeError:
            print 'Data end.'
            break
        print out.shape
        cnt += len(out)
        print 'Tiles num:', cnt
        plt.figure(figsize=(8, 8))
        for k, tile in enumerate(out):
            plt.subplot(8, 8, 1 + k)
            plt.imshow(tile)
            plt.title('{}({})'.format(k, ids[k, 0]))
            plt.axis('off')
        plt.show(True)

    plt.show()
    coord.request_stop()
    coord.join(threads)


def test_train_batches():
    images, gt_counts, image_id_list = read_train_batch(
        # join(ROOT_DIR, 'data/dbg_train_black_sc1.0_seed1993_vp0.1_v0.tfrecords',
        join(ROOT_DIR, 'records/dbg_1_600x600.tfrecords'),
        tile_size=299,
        shuffle=False,
        augmentations=True,
        full_image_scale_aug=True,
        min_scale_augmentations=1.0,
        max_scale_augmentations=1.5,
        should_repeat_tiles=False,
        as_float=True,
        num_epochs=2)

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    import matplotlib.pyplot as plt

    unique_img_ids = set()
    cnt = 0
    for i in xrange(0, 200):
        try:
            out, out_counts, out_ids = sess.run([images, gt_counts, image_id_list])
            out = np.asarray(out, dtype=np.uint8)
            unique_img_ids.update(out_ids[:, 0].tolist())
        except tf.errors.OutOfRangeError:
            print 'Data end.'
            break
        if i < 0: continue
        print out.shape
        cnt += len(out)
        print 'Tiles num:', cnt
        plt.figure(figsize=(12, 12))
        for k, tile in enumerate(out):
            plt.subplot(8, 8, 1 + k)
            plt.imshow(tile)
            cnt_str = ','.join(map(lambda x: str(int(np.round(x, 1))), out_counts[k].tolist()))
            plt.title('{}:{}'.format(out_ids[k, 0], cnt_str))
            plt.axis('off')
        plt.show(True)

    plt.show()
    # print unique_img_ids
    # df = pd.DataFrame(data=sorted(list(unique_img_ids)))
    # df.to_csv('/export/home/asanakoy/workspace/tmp/dbg_val_gray_sc1.0_seed1993_vp0.1_imgids.csv')
    coord.request_stop()
    coord.join(threads)


def dbg():
    image, density_map, image_id = read_and_decode_single_train_example(
        join(ROOT_DIR, 'records/val_gray_sc1.0_seed1993_vp0.1.tfrecords'),
                                                                        num_epochs=None)
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    import matplotlib.pyplot as plt

    cnt = 0
    for i in xrange(200):
        try:
            img, d_map, im_id = sess.run([image, density_map, image_id])
        except tf.errors.OutOfRangeError:
            print 'Data end.'
            break
        print img.shape
        print d_map.shape
        print 'Tiles num:', cnt
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        cnt_str = ','.join(map(lambda x: str(int(np.round(x, 0))), d_map.sum(axis=(0, 1)).tolist()))
        plt.title('{}:{}'.format(im_id, cnt_str))
        plt.axis('off')
        plt.figure(figsize=(8, 8))
        plt.imshow(d_map.sum(axis=2), cmap='jet')
        plt.show(True)

    plt.show()
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    # dbg()
    test_train_batches()
    # test_test_batches()
