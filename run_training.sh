#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python run_segmentation_reg_tfrecords.py \
--model=inception_resnet_v2 --test_iter=1000 --snapshot_iter=1000 \
--lr=1e-5 --batch_size=32 --tile_size=299 --pool=avg --extra_fc=256 \
--aug --full_scale_aug --min_scale_aug=0.5 --max_scale_aug=1.5
