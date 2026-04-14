import numpy as np
from PIL import Image
import cv2

def generate_simple_heatmap(input_tensor):
    """
    Generates a simple attention-like heatmap for UI visualization.
    """

    # 🔥 HANDLE ALL INPUT TYPES
    if isinstance(input_tensor, Image.Image):
        img = np.array(input_tensor.convert("L"))

    elif isinstance(input_tensor, np.ndarray):
        img = input_tensor

    else:
        # assume torch tensor
        img = input_tensor.squeeze().cpu().numpy()

    # 🔥 FIX DIMENSIONS (THIS WAS YOUR ERROR)
    while img.ndim > 2:
        img = img[0]

    # Normalize
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_uint8 = np.uint8(img * 255)

    # Heatmap
    heatmap = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)

    # Base grayscale → BGR
    base = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    # Overlay
    overlay = cv2.addWeighted(base, 0.45, heatmap, 0.55, 0)

    # Convert to PIL
    heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    return heatmap_pil, overlay_pil