# The code for the Kaggle competition "NOAA Fisheries Steller Sea Lion Population Count".
https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count

Our team  me([artem.sanakoev][1]) and [DmitryKotovenko][2]) scored **2-nd place** on the Public leaderboard and **4-th place** on Private.


## Short description of the solution


Preprocessing
--------------

GT count for each tile was generated as a sum over heatmap (to overcome cases with lions on the border of the tile).   On top of each lion we put a Gaussian with a standard deviation heuristically estimated by calculating the smallest distance between lion on the image.
We set the standard deviation to be 50 at least for each Gaussian and adjust it according to size of the animals from different classes (
multiplied by 2 for adult males and by 0.5 for pups).

<a href="https://ibb.co/hhAda5">GT Gaussians for tile</a>
![GT Gaussians for tile][3]


Model and Training
------
Our model incarnates **regression for 5 classes on tiles of the images**. (In similar spirit as the approach of @outrunner)

**Inception Resnet v2** pretrainedon Imagenet.
We substituted the last layer with 256-way FC layer + dropout + 5-way FC layer on top. + RMSE loss.
Then we fine-tuned the model on 299x299 image tiles with Adam optimizer.

**Augmentations:**  random rotation on 80/180/270 grads, random flip left-right, bottom-up.

**Scale augmentations:** one model without them, one model with 0.83-1.25 random scaling, one model with 0.66 - 1.5 random scaling.

<a href="https://ibb.co/kVhwTQ">RMSE on val for 3 best models</a>
![RMSE on val for 3 best models][4]


Testing
-------

**During test** we made predictions up to 5 times for each model using different shifts of the tiles in the image.
Test images were downscaled in 0.4-0.5 times.

The final ensemble was made by averaging all the predictions.
**Private LB RMSE:** *13.18968*
**Public LB RMSE:** *13.29065*

Applying further postprocessing as suggested by @outrunner, could improve results.
Just increasing the number of pups by 20% gives a huge improvement:
**Private LB RMSE:**  *12.58131*
**Public LB RMSE:** *12.75510*


Some negative experiments
-------

We labeled some images from train set according to scale and trained a CNN to regress a scale of the image.   This could unify all the images to have the same approximate size of the lions of corresponding classes and simplify the CNN training to count animals.

But it didn't work out.   I reckon, the reason is the high variation in terrain and inability to estimate scale of objects if you look at them within a small spatial context (even with my own eyes).


# How to run

### Requirements

- python 2.7
- tensorflow v1.*

### Setup

0. Set up the following folder structure:
- ROOT_DIR
    - checkpoints
    - out_preds
    - predictions
    - kaggle_sea_lions_counting  - this is the cloned repository

1. `mv kaggle_sea_lions_counting/data ${ROOT_DIR}/data`

2.
- Set your *ROO_TDIR* path in `config.py`
- Donwload tensorflow slim models (at least inception_resnet_v2) and set the path to it in `config.py`

3. Downsample test images to scale=0.5 (decrease the size on each dimension in 2 times)
`python resize_images.py`

4. Remove mismatched images from train: `python remove_mismatched_images.py`

5. Translate black regions from train_dotted to train : `python preproc_black_regions_train.py`

6. Prepare GT heatmaps for training (using GT coordinates of the lions):
run all cells in `notebooks/generate_gt_heatmaps.ipynb`

7. Generate tfrecords for train/val/test: run `python data/generate_tfrecords.py`
    - check that tfrecords are correctly generated: run `python data/records.py`

7. Run training `sh run_training.sh`

8. Run predict using 5 different offsets (paddings) for test images `sh run_predict_with_5_different_paddings.sh`.
Submission file will be generated and saved to `$ROOT_DIR/predictions`


### Misc

- `data/make_ensemble.py` generates the final submission which gave us the 2/4-th place in public/private test.
But to run it you need to train a whole bunch of different models first:P

- `examples_how_to_run_scripts.txt` contains some examples how to use different parameters of the scripts


  [1]: https://www.kaggle.com/asanakoev
  [2]: https://www.kaggle.com/chelovekparohod
  [3]: https://preview.ibb.co/bJ5Bv5/Screenshot_from_2017_06_28_16_28_57.png
  [4]: https://preview.ibb.co/jx6O8Q/Screenshot_from_2017_06_28_17_02_17.png