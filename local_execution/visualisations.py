from typing import List
from rasterio.plot import reshape_as_image

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from constants import classes_to_label


def visualize_losses_during_training(train_epoch_losses: List[float], validation_epoch_losses: List[float]) -> None:
    """
    Visualizes losses gathered during training.
    :param train_epoch_losses: List containing training loss per epoch
    :param validation_epoch_losses: List containing validation loss per epoch
    :return: None
    """
    # prepare plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # add grid
    ax.grid(linestyle="dotted")

    # plot the training epochs vs. the epochs' classification error
    ax.plot(np.array(range(1, len(train_epoch_losses) + 1)), train_epoch_losses, label='epoch train. loss (blue)')
    ax.plot(np.array(range(1, len(validation_epoch_losses) + 1)), validation_epoch_losses,
            label='epoch val. loss (blue)')
    # add axis legends
    ax.set_xlabel("[training epoch $e_i$]", fontsize=10)
    ax.set_ylabel("[Classification Error $\\mathcal{L}^{CE}$]", fontsize=10)

    # set plot legend
    plt.legend(loc="upper right", numpoints=1, fancybox=True)

    # add plot title
    plt.title("Training Epochs $e_i$ vs. Classification Error $L^{CE}$", fontsize=10);


def plot_confusion_matrix(y_true: List[int], y_pred: List[int]) -> None:
    """
    Plot confusion matrix for predictions
    :param y_true: List containing true labels
    :param y_pred: List containing predicted labels
    :return: None; show plot
    """
    classes_in_order = [v for k, v in sorted(classes_to_label.items())]
    cm = confusion_matrix(y_true, y_pred)

    df_confusion_matrix = pd.DataFrame(
        cm / np.sum(cm) * 10,
        index=classes_in_order,
        columns=classes_in_order
    )
    plt.figure(figsize=(10, 10))
    sns.heatmap(df_confusion_matrix, annot=True)


def normalize_for_display(band_data):
    """Normalize multi-spectral imagery across bands.
    The input is expected to be in HxWxC format, e.g. 64x64x13.
    To account for outliers (e.g. extremly high values due to
    reflective surfaces), we normalize with the 2- and 98-percentiles
    instead of minimum and maximum of each band.
    """
    band_data = np.array(band_data)
    lower_perc = np.percentile(band_data, 2, axis=(0, 1))
    upper_perc = np.percentile(band_data, 98, axis=(0, 1))

    return (band_data - lower_perc) / (upper_perc - lower_perc)


def visualize_bands(sample) -> None:
    """
    Visualize all bands of an image
    :param sample: image of shape (12xWxH)
    :return: None; show plot
    """
    img = reshape_as_image(sample)
    normalized_img = normalize_for_display(img)
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    b = 0
    for i in range(3):
        for j in range(4):
            idx = (i, j)
            axs[idx].imshow(normalized_img[:, :, b], cmap="gray")
            axs[idx].set_title(f"B{b+1}")
            axs[idx].axis(False)
            b += 1
    plt.show()
