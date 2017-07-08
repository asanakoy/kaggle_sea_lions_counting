#!/usr/bin/env bash


python predict_from_tfrecords.py --model=inception_resnet_v2 \
    --suf=_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0 \
    --snapshot_iter=95000 \
    --batch_size=32 \
    --pool=avg \
    --tile_size=299 \
    --scale=a \
    --extra_fc=256

for pad_offset in 0 149 74 223
do
    echo "iter ${iter}"

    python predict_from_tfrecords.py --model=inception_resnet_v2 \
    --suf=_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0 \
    --snapshot_iter=95000 \
    --batch_size=32 \
    --pool=avg \
    --tile_size=299 \
    --scale=a \
    --extra_fc=256 \
    --img_pad_offset=${pad_offset}
done
