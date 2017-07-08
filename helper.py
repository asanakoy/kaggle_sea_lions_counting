# Copyright (c) 2017 Artsiom Sanakoyeu and Dmitry Kotovenko
"""
Helpr function to visualize predicted heatmaps
"""
import numpy as np
import os
from matplotlib import pyplot as plt

import preprocessing.vgg_preprocessing as vgg_preprocessing


def from_scaled_image(img):
    return np.asarray((img + 1.) * 127., dtype=np.uint8)


def visualize_model_predictions(img_patches, ground_truth_patches, predictions_patches,
                                visualize=False,
                                visualize_each_class=False):
    if visualize:
        if visualize_each_class:
            for img, ground_truth, prediction in zip(img_patches, ground_truth_patches,
                                                     predictions_patches):

                num_cols = ground_truth.shape[-1]
                plt.figure(figsize=(3 * (num_cols + 1), 10))
                # plt.figure(figsize=(10, 10))
                plt.subplot(3, 1, 1)
                plt.title("Original Image.")
                plt.imshow(from_scaled_image(img))
                # plt.show(False)

                # Now plot heatmap for an each class separately.
                for cls in range(ground_truth.shape[-1]):
                    num_animals_gt = np.sum(ground_truth[:, :, cls])
                    num_animals_predicted = np.sum(prediction[:, :, cls])

                    plt.subplot(3, num_cols, num_cols + cls + 1)
                    plt.title("GT. Class{}: ({:.2f})".format(cls, num_animals_gt))
                    patch_img = from_scaled_image(img)  # TODO: resize to the size of heatmap
                    plt.imshow(patch_img)
                    plt.imshow(ground_truth[:, :, cls], cmap='jet', alpha=0.5)

                    plt.subplot(3, num_cols, 2 * num_cols + cls + 1)
                    plt.title("Prediction. Class{}: ({:})".format(cls, num_animals_predicted))
                    plt.imshow(patch_img)
                    plt.imshow(prediction[:, :, cls], cmap='jet', alpha=0.5)

                    print("Class %d." % cls)
                    print("L2 loss on class: %f." % np.sqrt(np.sum(
                        np.power(ground_truth[:, :, cls] - prediction[:, :, cls], 2))))
                    print("Gt number of animals in class: %f." % num_animals_gt)
                    print("Predicted number of animals in class: %f." % num_animals_predicted)
                    print('AE:', np.abs(num_animals_predicted - num_animals_gt))
                plt.show()

        else:
            for img, ground_truth, prediction in zip(img_patches, ground_truth_patches,
                                                     predictions_patches):
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.title("Original Image.")
                plt.imshow(from_scaled_image(img))
                plt.subplot(1, 3, 2)
                plt.title("Ground Truth Density.")
                im = plt.imshow(np.sum(ground_truth, axis=-1), cmap='jet')
                plt.colorbar(im, orientation='horizontal')
                plt.subplot(1, 3, 3)
                plt.title("Predicted Density.")
                im = plt.imshow(np.sum(prediction, axis=-1), cmap='jet')
                plt.colorbar(im, orientation='horizontal')

                print("L2 loss: %f." % np.sqrt(np.sum(np.power(ground_truth - prediction, 2))))
                print("Gt number of animals: %f." % np.sum(ground_truth))
                print("Predicted number of animals: %f." % np.sum(prediction))

                plt.show()

        print('\n\n')
    # Now compute RMSE for each class.
    rmse_loss = 0
    for cls in range(ground_truth_patches.shape[-1]):
        predicted_number = np.sum(predictions_patches[:, :, :, cls], axis=(1, 2))
        true_number = np.sum(ground_truth_patches[:, :, :, cls], axis=(1, 2))
        rmse_loss_for_class = np.sqrt(np.mean(np.power(predicted_number - true_number, 2)))
        print("RMSE for class %d = %f." % (cls, rmse_loss_for_class))
        rmse_loss += rmse_loss_for_class

    # And find final RMSE .
    rmse_loss /= ground_truth_patches.shape[-1]
    print("\n\nRMSE = %f " % rmse_loss)


def inverse_vgg16_preproc(img):
    img = img.copy()
    means = [vgg_preprocessing._R_MEAN,
             vgg_preprocessing._G_MEAN,
             vgg_preprocessing._B_MEAN]
    num_channels = img.shape[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    for i in range(num_channels):
        img[:, :, i] += means[i]
    img = np.asarray(img, dtype=np.uint8)
    print img
    assert np.all(0 <= img) and np.all(img <= 255)
    return img


def visualize_segmentation(img_patches,
                           ground_truth_patches,
                           predictions_patches,
                           visualize=False,
                           visualize_each_class=False,
                           save=True):

    if visualize:
        if visualize_each_class:
            for img, ground_truth, prediction in zip(img_patches, ground_truth_patches,
                                                     predictions_patches):

                assert np.all(np.isclose(prediction.sum(axis=2), 1.0))
                img = np.asarray(img, dtype=np.uint8)
                num_cols = prediction.shape[-1]
                plt.figure(figsize=(3 * (num_cols + 1), 10))
                # plt.figure(figsize=(10, 10))
                plt.subplot(3, 1, 1)
                plt.title("Original Image.")
                plt.imshow(img)
                # plt.show(False)

                prediction_labels = np.argmax(prediction, axis=2)

                # Now plot heatmap for an each class separately.
                for cls in range(prediction.shape[-1]):

                    plt.subplot(3, num_cols, num_cols + cls + 1)
                    plt.title("GT. Class{}".format(cls))
                    patch_img = img
                    plt.imshow(patch_img)
                    plt.imshow((ground_truth == cls).astype(float), cmap='jet', alpha=0.5, vmin=0.0, vmax=1.0)

                    plt.subplot(3, num_cols, 2 * num_cols + cls + 1)
                    plt.title("Prd. Class{}: max={:.2f}".format(cls, prediction[:, :, cls].astype(float).max()))
                    plt.imshow(patch_img)
                    plt.imshow(prediction[:, :, cls], cmap='jet', alpha=0.65, vmin=0.0)
                    print("Class %d." % cls)
                    if save:
                        plt.figure(figsize=(20, 15))
                        plt.title("Prd. Class{}: max={:.2f}".format(cls, prediction[:, :, cls].astype(float).max()))
                        plt.imshow(patch_img)
                        plt.imshow(prediction_labels == cls, cmap='jet', alpha=0.65, vmin=0.0)
                        plt.savefig(os.path.expanduser('~/tmp/seg_predictions/class_{}.jpg'.format(cls)))
                        plt.close()

                plt.show()

        else:
            for img, ground_truth, prediction in zip(img_patches, ground_truth_patches,
                                                     predictions_patches):
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.title("Original Image.")
                plt.imshow(inverse_vgg16_preproc(img))
                plt.subplot(1, 3, 2)
                plt.title("GT Segmentation.")
                im = plt.imshow(ground_truth, cmap='jet')
                plt.colorbar(im, orientation='horizontal')
                plt.subplot(1, 3, 3)
                plt.title("Prd Hearmap.")
                im = plt.imshow(prediction.sum(axis=2), cmap='jet', vmin=0, vmax=5)
                plt.colorbar(im, orientation='horizontal')
                plt.show()
