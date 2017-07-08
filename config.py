# Set this to point to your working directory where you have cloned the code
# Cloned repository folder `kaggle_sea_lions_counting` must be inside $ROOT_DIR
ROOT_DIR = '/my/path/to/workspace/kaggle/noaa'

# Set here your paths to tensorflow slim models
# The links to download them are in https://github.com/tensorflow/models/blob/master/slim/README.md
imagenet_snapshots_map = {
    # 'vgg_16': 'tf_models/slim/checkpoints/vgg_16.ckpt',
    # 'vgg_19': 'tf_models/slim/checkpoints/vgg_19.ckpt',
    # 'resnet_v2_50': 'tf_models/slim/checkpoints/resnet_v2_50.ckpt',
    # 'resnet_v2_101': 'tf_models/slim/checkpoints/resnet_v2_101.ckpt',
    # 'resnet_v2_152': 'tf_models/slim/checkpoints/resnet_v2_152.ckpt',
    'inception_resnet_v2': 'tf_models/slim/checkpoints/inception_resnet_v2_2016_08_30.ckpt'
}
