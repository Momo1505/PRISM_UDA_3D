import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from mmseg.datasets import CityscapesDataset

def save_segmentation_map(ema,sam, save_path):
    palette = CityscapesDataset.PALETTE
    cityscapes_cmap = ListedColormap(np.array(palette) / 255.0)

    rows, cols = 1, 2  # Increase cols to 4 for the new plot
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 3 * rows),
        gridspec_kw={
            'hspace': 0.1,
            'wspace': 0.05,
            'top': 0.95,
            'bottom': 0.05,
            'right': 0.95,
            'left': 0.05
        },
    )

    # Plot the images
    axs[0].imshow(ema)
    axs[0].set_title('EMA Source')
    plt.axis('off')

    axs[1].imshow(sam)
    axs[1].set_title('SAM')

    plt.axis('off')
    plt.savefig(save_path)
    plt.close()