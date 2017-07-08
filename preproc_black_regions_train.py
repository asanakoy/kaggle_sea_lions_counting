# Copyright (c) 2017 Artsiom Sanakoyeu
import glob
import numpy as np
import os
from os.path import join
from scipy.ndimage.morphology import binary_opening
from scipy.ndimage.morphology import generate_binary_structure
from scipy.misc import imsave, imread
from tqdm import tqdm

from data_utils import ROOT_DIR


def black_out_train_images(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_img_paths = glob.glob(join(ROOT_DIR, 'Train/*jpg'))
    train_img_ids = map(lambda x: int(os.path.basename(x)[:-4]), train_img_paths)
    fill_color = [0, 0, 0]

    struct_el = generate_binary_structure(2, 2)
    for image_id in tqdm(train_img_ids):
        img_dotted_path = join(ROOT_DIR, 'TrainDotted', '{}.jpg'.format(image_id))
        img_path = join(ROOT_DIR, 'Train', '{}.jpg'.format(image_id))

        image_dotted = imread(img_dotted_path)
        image = imread(img_path)

        mask = np.all(image_dotted == 0, axis=2)
        mask_dil = binary_opening(mask, structure=struct_el, iterations=6)

        image_black_out = image.copy()
        # set to fil color all regions which are black on dotted images
        image_black_out[mask_dil] = fill_color

        imsave(join(out_dir, '{}.jpg'.format(image_id)), image_black_out)


# mask_image = np.zeros_like(image)
#         mask_image[mask] = [255, 255, 0]
#         mask_image[mask_dil] = [255, 0, 0]
#         plt.figure(figsize=(12, 12))
#         plt.imshow(image)
#         plt.imshow(mask_image, alpha=0.6)
#         plt.title(str(image_id))
#         plt.grid(False)

if __name__ == '__main__':
    black_out_train_images(join(ROOT_DIR, 'Train/black'))
