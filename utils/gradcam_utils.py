import numpy as np
from PIL import Image

def generate_simple_heatmap(input_tensor):
    """
    Generates a simple attention-like heatmap for UI visualization.
    """

    img = input_tensor.squeeze().cpu().numpy()

    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
        img = np.mean(img, axis=2)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    heatmap = np.stack([
        img * 255,
        np.zeros_like(img),
        np.zeros_like(img)
    ], axis=2)

    heatmap = heatmap.astype(np.uint8)

    return Image.fromarray(heatmap)