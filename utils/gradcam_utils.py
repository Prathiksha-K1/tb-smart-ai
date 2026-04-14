import numpy as np
from PIL import Image

def generate_simple_heatmap(input_tensor):
    """
    Generates a simple heatmap WITHOUT using cv2 (deployment safe)
    """

    # Handle input types
    if isinstance(input_tensor, Image.Image):
        img = np.array(input_tensor.convert("L"))

    else:
        try:
            img = input_tensor.squeeze().cpu().numpy()
        except:
            img = np.array(input_tensor)

    # Fix dimensions
    while img.ndim > 2:
        img = img[0]

    # Normalize
    img = img.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # 🔥 Create RED heatmap manually
    heatmap = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    heatmap[..., 0] = (img * 255).astype(np.uint8)  # Red channel

    # Overlay (simple blend)
    base = np.stack([img*255]*3, axis=-1).astype(np.uint8)
    overlay = (0.6 * base + 0.4 * heatmap).astype(np.uint8)

    heatmap_pil = Image.fromarray(heatmap)
    overlay_pil = Image.fromarray(overlay)

    return heatmap_pil, overlay_pil