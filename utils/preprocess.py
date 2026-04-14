import cv2
import numpy as np
from PIL import Image

# ============================================================
# STRICT CHEST X-RAY VALIDATION
# ============================================================
def get_xray_validity_score(pil_image):
    """
    Stronger heuristic chest X-ray validator.
    Returns score from 0 to 100.
    Higher score = more likely to be a frontal chest X-ray.
    """

    img = np.array(pil_image.convert("RGB").resize((512, 512)))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # -----------------------------
    # 1) COLOR CHECK
    # Chest X-rays should be near grayscale
    # -----------------------------
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    color_diff = (
        np.mean(np.abs(r - g)) +
        np.mean(np.abs(g - b)) +
        np.mean(np.abs(r - b))
    ) / 3.0
    grayscale_score = max(0, 100 - min(color_diff * 3, 100))

    # -----------------------------
    # 2) BRIGHTNESS CHECK
    # Chest X-rays usually have moderate grayscale distribution
    # -----------------------------
    mean_intensity = np.mean(gray)
    brightness_score = max(0, 100 - abs(mean_intensity - 125))

    # -----------------------------
    # 3) CONTRAST CHECK
    # X-rays usually have medium-high contrast
    # -----------------------------
    std_intensity = np.std(gray)
    contrast_score = min(std_intensity * 2.2, 100)

    # -----------------------------
    # 4) EDGE DENSITY CHECK
    # Chest X-rays have moderate structural edges
    # -----------------------------
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    if 0.03 <= edge_density <= 0.18:
        edge_score = 100
    else:
        edge_score = max(0, 100 - abs(edge_density - 0.08) * 800)

    # -----------------------------
    # 5) SYMMETRY CHECK
    # Chest X-rays often have left-right rough symmetry
    # -----------------------------
    left_half = gray[:, :256]
    right_half = np.fliplr(gray[:, 256:])
    symmetry_diff = np.mean(np.abs(left_half.astype(np.float32) - right_half.astype(np.float32)))
    symmetry_score = max(0, 100 - min(symmetry_diff, 100))

    # -----------------------------
    # 6) TEXTURE CHECK
    # X-rays have smoother medical texture than random photos
    # -----------------------------
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if 100 <= lap_var <= 3000:
        texture_score = 100
    else:
        texture_score = max(0, 100 - abs(lap_var - 1200) / 20)

    # -----------------------------
    # FINAL WEIGHTED SCORE
    # -----------------------------
    final_score = (
        0.22 * grayscale_score +
        0.15 * brightness_score +
        0.18 * contrast_score +
        0.18 * edge_score +
        0.12 * symmetry_score +
        0.15 * texture_score
    )

    return round(final_score, 2)


def is_valid_chest_xray(pil_image, threshold=62):
    """
    Returns:
        valid (bool), score (float)
    """
    score = get_xray_validity_score(pil_image)
    return score >= threshold, score


# ============================================================
# DIGITAL IMAGE PROCESSING PIPELINE
# 8 METHODS
# ============================================================
def apply_dip_pipeline(pil_image):
    """
    Returns:
        stage_images (list of tuples): [("title", PIL_image), ...]
        dip_scores (dict)
    """

    # -----------------------------
    # Convert to grayscale
    # -----------------------------
    gray = np.array(pil_image.convert("L"))
    gray = cv2.resize(gray, (512, 512))

    # -----------------------------
    # 1) Histogram Equalization
    # -----------------------------
    hist_eq = cv2.equalizeHist(gray)

    # -----------------------------
    # 2) CLAHE Enhancement
    # -----------------------------
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    clahe_img = clahe.apply(hist_eq)

    # -----------------------------
    # 3) Median Filtering
    # -----------------------------
    median_img = cv2.medianBlur(clahe_img, 3)

    # -----------------------------
    # 4) Gaussian Denoising
    # -----------------------------
    gaussian_img = cv2.GaussianBlur(median_img, (3, 3), 0)

    # -----------------------------
    # 5) Edge Enhancement
    # -----------------------------
    edges = cv2.Canny(gaussian_img, 60, 140)
    edges_overlay = cv2.addWeighted(gaussian_img, 0.85, edges, 0.15, 0)

    # -----------------------------
    # 6) Morphological Cleanup
    # -----------------------------
    kernel = np.ones((2, 2), np.uint8)
    morph_img = cv2.morphologyEx(edges_overlay, cv2.MORPH_CLOSE, kernel)

    # -----------------------------
    # 7) Final Sharpening
    # -----------------------------
    sharpen_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    final_img = cv2.filter2D(morph_img, -1, sharpen_kernel)
    final_img = cv2.normalize(final_img, None, 0, 255, cv2.NORM_MINMAX)

    # ============================================================
    # DIP CONTRIBUTION METRICS
    # ============================================================

    # Contrast improvement
    contrast_before = np.std(gray)
    contrast_after = np.std(clahe_img)
    contrast_gain = ((contrast_after - contrast_before) / (contrast_before + 1e-8)) * 100

    # Noise reduction estimate
    noise_before = np.mean(np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)))
    noise_after = np.mean(np.abs(clahe_img.astype(np.float32) - gaussian_img.astype(np.float32)))
    noise_reduction = ((noise_before - noise_after) / (noise_before + 1e-8)) * 100

    # Sharpness gain
    sharpness_before = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_after = cv2.Laplacian(final_img, cv2.CV_64F).var()
    sharpness_gain = ((sharpness_after - sharpness_before) / (sharpness_before + 1e-8)) * 100

    # Edge enhancement
    edge_before = np.sum(cv2.Canny(gray, 50, 150) > 0)
    edge_after = np.sum(cv2.Canny(final_img, 50, 150) > 0)
    edge_gain = ((edge_after - edge_before) / (edge_before + 1e-8)) * 100

    # Morphological cleanup proxy
    morph_gain = max(5, min(95, (np.std(morph_img) / (np.std(gray) + 1e-8)) * 25))

    # Histogram equalization proxy
    hist_gain = max(5, min(95, (np.std(hist_eq) / (np.std(gray) + 1e-8)) * 30))

    dip_scores = {
        "Histogram Equalization": round(max(5, min(95, hist_gain)), 2),
        "CLAHE Contrast Enhancement": round(max(5, min(95, contrast_gain)), 2),
        "Median Filtering": round(max(5, min(95, abs(noise_reduction) * 0.7)), 2),
        "Gaussian Denoising": round(max(5, min(95, abs(noise_reduction))), 2),
        "Edge Enhancement": round(max(5, min(95, edge_gain)), 2),
        "Morphological Cleanup": round(max(5, min(95, morph_gain)), 2),
        "Image Sharpening": round(max(5, min(95, sharpness_gain)), 2),
    }

    # ============================================================
    # STAGE IMAGES FOR UI
    # ============================================================
    stage_images = [
        ("Grayscale", Image.fromarray(gray)),
        ("Histogram Equalized", Image.fromarray(hist_eq)),
        ("CLAHE Enhanced", Image.fromarray(clahe_img)),
        ("Median Filtered", Image.fromarray(median_img)),
        ("Gaussian Denoised", Image.fromarray(gaussian_img)),
        ("Edge Enhanced", Image.fromarray(edges_overlay)),
        ("Morphological Cleanup", Image.fromarray(morph_img)),
        ("Final Sharpened", Image.fromarray(final_img)),
    ]

    return stage_images, dip_scores, Image.fromarray(final_img)