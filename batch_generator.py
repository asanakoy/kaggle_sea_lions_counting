# Copyright (c) 2017 Artsiom Sanakoyeu
import numpy as np


class BatchGenerator:
    def __init__(self, images, density_maps, gt_counts=None, shuffle_every_epoch=False, seed=42):
        self.cur_pos = 0
        self.images = images
        self.density_maps = density_maps
        self.gt_counts = gt_counts
        self.rs = np.random.RandomState(seed)
        self.shuffle_every_epoch = shuffle_every_epoch
        self.val_epoch_size = None
        if self.images is not None:
            if self.shuffle_every_epoch:
                self.perm = self.rs.permutation(len(self.images))
            else:
                self.perm = np.arange(len(images))

    def __iter__(self):
        return self

    def next_feed_dict(self, net, batch_size=128, phase='test'):
        """Fills the feed_dict for training the next step.

        Returns:
          feed_dict: The feed dictionary mapping from placeholders to values.
        """
        if phase not in ['train', 'test']:
            raise ValueError('phase must be "train" or "test"')
        is_phase_train = phase == 'train'
        if self.images is None:
            assert net.is_training_pl is not None, 'is_training_pl is None!'
            return {net.is_training_pl: is_phase_train}

        batch_indices = self.perm[self.cur_pos:self.cur_pos + batch_size]
        feed_dict = {
            net.input_images_pl: self.images[batch_indices],
        }
        if self.density_maps is not None:
            assert net.input_heatmaps
            feed_dict[net.input_heatmaps] = self.density_maps[batch_indices]

        if self.gt_counts is not None:
            feed_dict[net.input_gt_counts] = self.gt_counts[batch_indices]

        if net.is_training_pl is not None:
            feed_dict[net.is_training_pl] = is_phase_train

        self.cur_pos += batch_size
        if self.cur_pos >= len(self.images):
            self.cur_pos = 0
            if self.shuffle_every_epoch:
                self.perm = self.rs.permutation(len(self.images))

        return feed_dict
