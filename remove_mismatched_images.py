# Copyright (c) 2017 Artsiom Sanakoyeu
import os
from os.path import join
import pandas as pd

from data_utils import ROOT_DIR


if __name__ == '__main__':
    df = pd.read_csv(join(ROOT_DIR, 'data/MismatchedTrainImages.txt'))
    print df
    dst_dir = join(ROOT_DIR, 'Train')

    for img_id in df['train_id']:
        dst_path = join(dst_dir, '{}.jpg'.format(img_id))
        os.remove(dst_path)
