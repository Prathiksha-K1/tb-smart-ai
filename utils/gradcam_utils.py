import numpy as np
from PIL import Image

def generate_simple_heatmap(input_image):
    """
    Deployment-safe heatmap (no torch, no cv2)
    """

    if isinstance(input_image, Image.Image):
        img = np.array(input_image.convert("L"))
    else:
        img = input_image

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    heatmap = np.stack([
        img * 255,
        np.zeros_like(img),
        np.zeros_like(img)
    ], axis=2).astype(np.uint8)

    heatmap_img = Image.fromarray(heatmap)

    return heatmap_img, heatmap_img