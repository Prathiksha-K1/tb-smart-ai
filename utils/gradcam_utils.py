import numpy as np
from PIL import Image

def generate_simple_heatmap(input_tensor):

    img = input_tensor.squeeze().cpu().numpy()

    # FIX DIMENSION ISSUE
    if img.ndim == 4:
        img = img[0]

    if img.ndim == 3:
        img = np.mean(img, axis=0)

    # Normalize
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Create RED heatmap
    heatmap = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    heatmap[:, :, 0] = (img * 255).astype(np.uint8)

    # Overlay
    base = (img * 255).astype(np.uint8)
    base_rgb = np.stack([base]*3, axis=-1)

    overlay = (0.5 * base_rgb + 0.5 * heatmap).astype(np.uint8)

    heatmap_img = Image.fromarray(heatmap)
    overlay_img = Image.fromarray(overlay)

    return heatmap_img, overlay_img