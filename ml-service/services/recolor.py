import logging
from typing import Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def hex_to_hsv(target_hex: str) -> Tuple[float, float, float]:
    hex_str = target_hex.lstrip("#")
    if len(hex_str) not in (6, 8):
        raise ValueError("Expected hex color like #RRGGBB")

    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)

    color_bgr = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])


def recolor_region_hsv(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    target_hex: str,
) -> np.ndarray:
    """
    Recolor only the masked garment region using HSV.

    Spec: Replace only the Hue channel with the selected color.
    Preserve Saturation and Value to keep shading, shadows, and fabric texture.
    """
    log.info("--- RECOLOR START ---")
    log.info("Target color: %s", target_hex)

    if image_bgr.dtype != np.uint8:
        raise ValueError("Expected image dtype uint8")

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = (mask > 0).astype(np.uint8)
    mask_pixels = int(mask.sum())
    log.info("Mask pixels to recolor: %d", mask_pixels)
    if mask_pixels == 0:
        log.warning("Mask is empty — returning original image unchanged")
        return image_bgr.copy()

    target_h, target_s, target_v = hex_to_hsv(target_hex)
    log.info("Target HSV: H=%.1f, S=%.1f, V=%.1f", target_h, target_s, target_v)

    # Minimal dilation so we don't bleed onto hair/skin
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    log.info("Dilated mask pixels: %d", int(mask_dilated.sum()))

    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Log original HSV stats in the masked region
    h_orig = image_hsv[:, :, 0][mask == 1]
    s_orig = image_hsv[:, :, 1][mask == 1]
    v_orig = image_hsv[:, :, 2][mask == 1]
    log.info("Original garment HSV — H: mean=%.1f, S: mean=%.1f, V: mean=%.1f",
             h_orig.mean(), s_orig.mean(), v_orig.mean())

    h = image_hsv[:, :, 0]
    s = image_hsv[:, :, 1]

    # Replace Hue where mask
    h = np.where(mask_dilated == 1, target_h % 180.0, h)
    image_hsv[:, :, 0] = h

    # Boost saturation so hue shift is visible on low-saturation (white/gray) fabrics.
    # Preserve relative S differences (patterns, folds, texture) by scaling toward target_s.
    # Formula: new_s = lerp(original_s, target_s, blend_factor)
    # blend_factor is higher when original S is low (white fabrics need more boost).
    mean_s = float(s_orig.mean()) if s_orig.size > 0 else 128.0
    if mean_s < 80:
        # Low-saturation garment (white, gray, pastel): strong blend toward target
        blend = 0.7
        log.info("Low saturation garment (mean_s=%.1f) — blending S at %.0f%%", mean_s, blend * 100)
    elif mean_s < 150:
        # Medium saturation: moderate blend
        blend = 0.4
        log.info("Medium saturation garment (mean_s=%.1f) — blending S at %.0f%%", mean_s, blend * 100)
    else:
        # Already saturated: light blend to stay faithful to texture
        blend = 0.2
        log.info("High saturation garment (mean_s=%.1f) — blending S at %.0f%%", mean_s, blend * 100)

    s_new = np.where(
        mask_dilated == 1,
        np.clip(s * (1.0 - blend) + target_s * blend, 0, 255),
        s,
    )
    image_hsv[:, :, 1] = s_new

    # Log new HSV stats
    h_after = image_hsv[:, :, 0][mask == 1]
    s_after = image_hsv[:, :, 1][mask == 1]
    v_after = image_hsv[:, :, 2][mask == 1]
    log.info("After recolor HSV  — H: mean=%.1f, S: mean=%.1f, V: mean=%.1f",
             h_after.mean(), s_after.mean(), v_after.mean())

    recolored_bgr = cv2.cvtColor(
        image_hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR,
    )

    # Feather at mask boundary to avoid hard edges
    dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    feather = np.clip(dist / 4.0, 0.0, 1.0)
    feather = feather[..., None]

    blended = (
        recolored_bgr.astype(np.float32) * (1 - feather)
        + image_bgr.astype(np.float32) * feather
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    log.info("--- RECOLOR DONE ---")
    return blended

