import numpy as np
import pandas as pd

CLASS_NAMES = list(['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups'])

def assemble_predictions_per_image(class_counts, images_info):
    predict_df = pd.DataFrame(index=images_info.index, columns=CLASS_NAMES, dtype=np.float32)
    prev_pos = 0
    for image_id, num_tiles in images_info['num_tiles'].iteritems():
        predict_df.loc[image_id, CLASS_NAMES] = \
            class_counts[prev_pos:prev_pos + num_tiles, :].sum(axis=0)
        prev_pos += num_tiles
    assert prev_pos == images_info.num_tiles.sum()
    return predict_df

all_tiles_preds = np.load('all_tile_predictions.npy')

images_info = pd.read_hdf('images_info.hdf5')
predict_df = assemble_predictions_per_image(all_tiles_preds, images_info)
predict_df.to_hdf('test_predict_df_59k.hdf5', 'df')