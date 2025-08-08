import os
import glob
from PIL import Image
import torch
from torchvision import transforms
from scipy.ndimage import label
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Helper function to compute surfaces of connected components
def compute_connected_components_area(mask: torch.Tensor):
    mask = mask.numpy()
    areas = []

    # for binary mask
    if mask.max() <=1 :
        labeled_mask, num_features = label(mask)
        for i in range(1,num_features+1):
            areas.append((labeled_mask == i).sum())
    # for multi class mask
    else :
        for cls in np.unique(mask):
            if cls == 0:
                continue # background
            else :
                binary_mask = (mask == cls).astype(np.uint)
                labeled_mask,num_features = label(binary_mask)
                for i in range(1,num_features+1):
                    areas.append((labeled_mask == i).sum())
    return areas

def plot_surface_dist(dataset:list):

    # Transformation for loading masks
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    percentile_5 = []
    percentile_95 = []

    for domain in dataset:
        # Load mask paths
        paths = sorted(glob.glob(f"data/{domain}/labels/*.png"))
        paths = [path for path in paths if not path.endswith("_labelTrainIds.png")]
        domain_areas = []
        # Process domain
        for path in tqdm(paths, desc=f"Processing {domain}"):
            mask = Image.open(path).convert("L")
            mask = mask_transform(mask).long().squeeze(0)
            areas = compute_connected_components_area(mask)
            domain_areas.extend(areas)
        
        percentiles = np.percentile(domain_areas,[5,95])
        percentile_5.append(percentiles[0])
        percentile_95.append(percentiles[1])

        print(f"for {domain} domain, 5% have less than {percentiles[0]} of surface and more than 5% have surface grater than {percentiles[1]}. min area = {min(domain_areas)}, max area = {max(domain_areas)}")

        # Plot distributions
        if domain_areas:
            plt.hist(np.log10(domain_areas), bins=50, alpha=0.5, label=f'{domain}')

    plt.xlabel("log10(Area in pixels)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Connected Component Surface Distribution Across Domains")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    min_area = min(percentile_5)
    max_area = max(percentile_95)

    min_zoom = np.sqrt(256*256 / max_area)
    max_zoom = np.sqrt(256*256 / min_area)

    print(f"Suggested zoom interval: [{min_zoom:.2f}, {max_zoom:.2f}]")