# Copyright (c) 2017 Artsiom Sanakoyeu
import os
from os.path import join
import glob
from tqdm import tqdm
import cv2

from data_utils import ROOT_DIR


def get_images_list(images_dir_path, extension='jpg'):
    image_list = glob.glob(images_dir_path + "/*." + extension)
    image_list.sort()
    return image_list


def downscale(images_list, output_dir, factor=0.5):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for path_to_image in tqdm(images_list):
        image = cv2.imread(path_to_image)
        res = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        file_to_save = os.path.join(output_dir, os.path.basename(path_to_image))
        if not os.path.exists(file_to_save):
            cv2.imwrite(file_to_save, res)


if __name__ == '__main__':
    test_dir = join(ROOT_DIR, 'Test/')
    test_resized_dir = join(ROOT_DIR, 'Test/black_0.5')

    # train_dir = join(ROOT_DIR, 'Train/black_out')
    # train_resized_dir = join(ROOT_DIR, 'Train/black_0.5')

    for src_dir, dst_dir in [(test_dir, test_resized_dir)]:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        print 'Resizing {} -> {}'.format(src_dir, dst_dir)
        image_list = get_images_list(src_dir)
        downscale(image_list, dst_dir, factor=0.5)
