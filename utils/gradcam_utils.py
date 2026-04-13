import numpy as np
from PIL import Image
import cv2

def generate_simple_heatmap(input_tensor):
    """
    Generates a simple attention-like heatmap for UI visualization.
    """
    img = input_tensor.squeeze().cpu().numpy()

    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
        img = np.mean(img, axis=2)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_uint8 = np.uint8(img * 255)

    heatmap = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)
    base = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.45, heatmap, 0.55, 0)

    heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    return heatmap_pil, overlay_pil