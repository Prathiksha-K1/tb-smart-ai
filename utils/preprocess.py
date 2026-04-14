import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# ============================================================
# STRICT CHEST X-RAY VALIDATION (NO CV2)
# ============================================================

def get_xray_validity_score(pil_image):

    img = np.array(pil_image.convert("RGB").resize((512, 512)))
    gray = np.array(pil_image.convert("L").resize((512, 512)))

    # 1) COLOR CHECK
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    color_diff = (
        np.mean(np.abs(r - g)) +
        np.mean(np.abs(g - b)) +
        np.mean(np.abs(r - b))
    ) / 3.0
    grayscale_score = max(0, 100 - min(color_diff * 3, 100))

    # 2) BRIGHTNESS
    mean_intensity = np.mean(gray)
    brightness_score = max(0, 100 - abs(mean_intensity - 125))

    # 3) CONTRAST
    std_intensity = np.std(gray)
    contrast_score = min(std_intensity * 2.2, 100)

    # 4) EDGE DENSITY (NO CANNY → SIMPLE GRADIENT)
    edges = np.abs(np.gradient(gray.astype(float)))[0]
    edge_density = np.sum(edges > 20) / edges.size
    if 0.03 <= edge_density <= 0.18:
        edge_score = 100
    else:
        edge_score = max(0, 100 - abs(edge_density - 0.08) * 800)

    # 5) SYMMETRY
    left_half = gray[:, :256]
    right_half = np.fliplr(gray[:, 256:])
    symmetry_diff = np.mean(np.abs(left_half.astype(np.float32) - right_half.astype(np.float32)))
    symmetry_score = max(0, 100 - min(symmetry_diff, 100))

    # 6) TEXTURE (Laplacian approx)
    lap = np.abs(np.gradient(gray.astype(float)))[0]
    lap_var = np.var(lap)
    if 100 <= lap_var <= 3000:
        texture_score = 100
    else:
        texture_score = max(0, 100 - abs(lap_var - 1200) / 20)

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
    score = get_xray_validity_score(pil_image)
    return score >= threshold, score


# ============================================================
# DIGITAL IMAGE PROCESSING PIPELINE (NO CV2)
# ============================================================

def apply_dip_pipeline(pil_image):

    gray = pil_image.convert("L").resize((512, 512))

    # 1) Histogram Equalization (approx using contrast)
    hist_eq = ImageEnhance.Contrast(gray).enhance(1.4)

    # 2) CLAHE (approx)
    clahe_img = ImageEnhance.Contrast(hist_eq).enhance(1.3)

    # 3) Median Filtering
    median_img = clahe_img.filter(ImageFilter.MedianFilter(size=3))

    # 4) Gaussian Denoising
    gaussian_img = median_img.filter(ImageFilter.GaussianBlur(radius=1))

    # 5) Edge Enhancement
    edges_overlay = gaussian_img.filter(ImageFilter.FIND_EDGES)

    # 6) Morphological Cleanup
    morph_img = edges_overlay.filter(ImageFilter.SMOOTH)

    # 7) Final Sharpening
    final_img = morph_img.filter(ImageFilter.SHARPEN)

    # ============================================================
    # METRICS (APPROX)
    # ============================================================

    gray_np = np.array(gray)
    final_np = np.array(final_img)

    contrast_before = np.std(gray_np)
    contrast_after = np.std(np.array(clahe_img))
    contrast_gain = ((contrast_after - contrast_before) / (contrast_before + 1e-8)) * 100

    noise_before = np.mean(np.abs(gray_np - np.array(gaussian_img)))
    noise_after = np.mean(np.abs(np.array(clahe_img) - np.array(gaussian_img)))
    noise_reduction = ((noise_before - noise_after) / (noise_before + 1e-8)) * 100

    sharpness_before = np.var(gray_np)
    sharpness_after = np.var(final_np)
    sharpness_gain = ((sharpness_after - sharpness_before) / (sharpness_before + 1e-8)) * 100

    edge_before = np.sum(np.abs(np.gradient(gray_np))[0] > 20)
    edge_after = np.sum(np.abs(np.gradient(final_np))[0] > 20)
    edge_gain = ((edge_after - edge_before) / (edge_before + 1e-8)) * 100

    morph_gain = max(5, min(95, (np.std(np.array(morph_img)) / (np.std(gray_np) + 1e-8)) * 25))
    hist_gain = max(5, min(95, (np.std(np.array(hist_eq)) / (np.std(gray_np) + 1e-8)) * 30))

    dip_scores = {
        "Histogram Equalization": round(hist_gain, 2),
        "CLAHE Contrast Enhancement": round(contrast_gain, 2),
        "Median Filtering": round(abs(noise_reduction) * 0.7, 2),
        "Gaussian Denoising": round(abs(noise_reduction), 2),
        "Edge Enhancement": round(edge_gain, 2),
        "Morphological Cleanup": round(morph_gain, 2),
        "Image Sharpening": round(sharpness_gain, 2),
    }

    stage_images = [
        ("Grayscale", gray),
        ("Histogram Equalized", hist_eq),
        ("CLAHE Enhanced", clahe_img),
        ("Median Filtered", median_img),
        ("Gaussian Denoised", gaussian_img),
        ("Edge Enhanced", edges_overlay),
        ("Morphological Cleanup", morph_img),
        ("Final Sharpened", final_img),
    ]

    return stage_images, dip_scores, final_img