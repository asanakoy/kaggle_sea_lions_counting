# Copyright (c) 2017 Artsiom Sanakoyeu
"""
Generate final submission.
"""
import numpy as np
import pandas as pd

from data_utils import *


paths = {
    'inception_resnet_v2_tile299_avgpool_fc256_aug_scale0.5_it74k':
        [
            join(ROOT_DIR, 'out_preds/blending/best2/submission_inception_resnet_v2_tile299_avgpool_fc256_aug_scale0.5_it74k.csv'),
            join(ROOT_DIR,              'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug_scale0.5_pad0it74k.csv'),
            join(ROOT_DIR,              'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug_scale0.5_pad149it74k.csv'),
            join(ROOT_DIR,              'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug_scale0.5_pad74it74k.csv'),
            join(ROOT_DIR,              'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug_scale0.5_pad223it74k.csv')
        ],
    'inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0_scale0.5_it95k':
        [
            join(ROOT_DIR, 'out_preds/blending/best2/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0_scale0.5_it95k.csv'),
            join(ROOT_DIR,              'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0_scale0.5_pad0it95k.csv'),
            join(ROOT_DIR,              'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0_scale0.5_pad149it95k.csv'),
            join(ROOT_DIR,              'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0_scale0.5_pad74it95k.csv'),
            join(ROOT_DIR,              'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0_scale0.5_pad223it95k.csv')
        ],
    'inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0_scalea_it95k':
        [
            join(ROOT_DIR, 'out_preds/blending/best2/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0_scalea_it95k.csv'),
            join(ROOT_DIR,              'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0_scalea_pad0it95k.csv'),
            join(ROOT_DIR,              'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0_scalea_pad149it95k.csv'),
            join(ROOT_DIR,              'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0_scalea_pad74it95k.csv'),
            join(ROOT_DIR,              'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0_scalea_pad223it95k.csv')
        ],
    'inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.8-1.2_coordsv0_scale0.5it100k':
        [
            join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.8-1.2_coordsv0_scale0.5it100k.csv'),
            join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.8-1.2_coordsv0_scale0.5_pad0it100k.csv'),
            join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.8-1.2_coordsv0_scale0.5_pad149it100k.csv'),
            join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.8-1.2_coordsv0_scale0.5_pad74it100k.csv'),
            join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.8-1.2_coordsv0_scale0.5_pad223it100k.csv')
        ],
    # 'inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.8-1.2_coordsv0_scalea_it100k': #gavno
    # [
    #     join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.8-1.2_coordsv0_scalea_pad0it100k.csv'),
    #     join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.8-1.2_coordsv0_scalea_pad149it100k.csv'),
    #     join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.8-1.2_coordsv0_scalea_pad74it100k.csv'),
    #     join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.8-1.2_coordsv0_scalea_pad223it100k.csv')
    #
    # ]
    # 'inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-1.0-2.0_coordsv0_scale0.5it52500':
    # [
    #     join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-1.0-2.0_coordsv0_scale0.5it52500.csv') # gavno
    # ]
    'submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.7-1.7_coordsv0_scale0.5_it50k':
    [
        join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.7-1.7_coordsv0_scale0.5_pad0it50k.csv'),  # TODO: try submit this
        join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.7-1.7_coordsv0_scale0.5_pad74it50k.csv'),
        join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.7-1.7_coordsv0_scale0.5_pad149it50k.csv'),
        join(ROOT_DIR, 'predictions/submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.7-1.7_coordsv0_scale0.5_pad223it50k.csv')
    ]
}

subm_df = dict()
for net_name in paths.keys():
    subm_df[net_name] = list()
    for filepath in paths[net_name]:
        df = pd.read_csv(filepath, index_col='test_id')
        df.sort_index(inplace=True)
        subm_df[net_name].append(df)

res = None
number_predictions = 0
for net_name in subm_df.keys():
    print 'net_name:', net_name
    print '====================='
    weight = 1
    if net_name == 'inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.8-1.2_coordsv0_scale0.5it100k':
        weight = 1
    elif net_name == 'inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.8-1.2_coordsv0_scalea_it100k':
        weight = 0
    elif net_name == 'inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-1.0-2.0_coordsv0_scale0.5it52500':
        weight = 0
    elif net_name == 'submission_inception_resnet_v2_tile299_avgpool_fc256_aug2_fullscaleaug-0.7-1.7_coordsv0_scale0.5_it50k':
        weight = 1
    for df in subm_df[net_name]:
        print df.describe()
        number_predictions += weight
        if res is None:
            res = df.copy() * weight
        else:
            res += df * weight

number_predictions += 0
print 'number_predictions', number_predictions
print '======'
print 'RESULT'
print '======'
res = np.round(res / float(number_predictions)).astype(int)
print '==================================\n'
print res.describe()

val_test_df = create_val_test_df()
evaluate_df(val_test_df, res.loc[val_test_df.index], is_3_classes=True)

ensemble_name = ('ens_aug_scale0.5_it74k_pad-None,0,149,74,223++'
                 'aug2_fullscaleaug-0.5-1.5_coordsv0_scale-0.5,a_it95k_pad-None,0,149,74,223++'
                 'aug2_fullscaleaug-0.8-1.2_coordsv0_scale0.5it100k_pad-None,0,149,74,223++'
                 '0.7-1.7_coordsv0_scale0.5_it50k_pad-0,149,74,223.csv')

res.to_csv(
    '../../predictions/' + ensemble_name)


df_ens_prev = pd.read_csv(('../../predictions/'
    'submission_inception_resnet_v2_tile299_avgpool_fc256_aug_scale0.5_it74k+tile299_avgpool_fc256_aug2_fullscaleaug-0.5-1.5_coordsv0_scale0.5_it95k+fullscaleaug-0.5-1.5_coordsv0_scalea_it95k.csv'),
                          index_col='test_id')

print '====\ncmp with prev'
evaluate_df(res, df_ens_prev)
evaluate_df(val_test_df, df_ens_prev.loc[val_test_df.index], is_3_classes=True)