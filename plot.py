import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from mmseg.datasets import CityscapesDataset

def save_segmentation_map(seg_map, save_path):
    palette = CityscapesDataset.PALETTE
    cityscapes_cmap = ListedColormap(np.array(palette) / 255.0)

    plt.figure(figsize=(10, 5))
    plt.imshow(seg_map, cmap=cityscapes_cmap)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()