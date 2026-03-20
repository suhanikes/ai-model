"""
Mask building from stored SegFormer segmentation: lasso → garment class (mode) → final mask.
No YOLO or object detection; uses only the precomputed segmentation mask.
"""
import logging
from typing import List, Tuple

import cv2
import numpy as np

from models.segformer_b2_clothes import GARMENT_CLASS_IDS

log = logging.getLogger(__name__)

PointList = List[Tuple[float, float]]

# Human-readable names from the model config
CLASS_NAMES = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses",
    4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress",
    8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
    12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
    16: "Bag", 17: "Scarf",
}


def polygon_mask_from_lasso(image_shape: tuple, lasso_points: PointList) -> np.ndarray:
    """Rasterize the lasso polygon into a binary mask (1 = inside, 0 = outside)."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[p[0], p[1]] for p in lasso_points], dtype=np.int32)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def _smooth_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological close + optional small Gaussian blur to soften edges."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    closed = cv2.GaussianBlur(closed.astype(np.float32), (3, 3), 0.5)
    return (closed > 0.5).astype(np.uint8)


def build_final_garment_mask_from_segmentation(
    image_shape: Tuple[int, int],
    lasso_points: PointList,
    segmentation_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine garment class inside lasso (mode), build garment mask, intersect with lasso,
    then smooth. No model inference.
    """
    h, w = image_shape[:2]
    log.info("--- MASK BUILDING START ---")
    log.info("Image shape: %d x %d", w, h)
    log.info("Segmentation mask shape: %s, unique classes in full image: %s",
             segmentation_mask.shape, np.unique(segmentation_mask).tolist())
    log.info("Lasso points count: %d", len(lasso_points))
    if lasso_points:
        xs = [p[0] for p in lasso_points]
        ys = [p[1] for p in lasso_points]
        log.info("Lasso bounding box: x=[%.1f, %.1f], y=[%.1f, %.1f]",
                 min(xs), max(xs), min(ys), max(ys))

    if segmentation_mask.shape[:2] != (h, w):
        log.info("Resizing seg mask from %s to (%d, %d)", segmentation_mask.shape[:2], h, w)
        segmentation_mask = cv2.resize(
            segmentation_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    lasso_mask = polygon_mask_from_lasso(image_shape, lasso_points)
    lasso_pixel_count = int(lasso_mask.sum())
    log.info("Lasso polygon mask: %d pixels (%.1f%% of image)",
             lasso_pixel_count, 100.0 * lasso_pixel_count / (h * w))

    # ALL pixels inside lasso (including non-garment)
    pixels_inside = segmentation_mask[lasso_mask == 1]
    if pixels_inside.size == 0:
        log.warning("Lasso region is empty (0 pixels inside)")
        raise ValueError("No garment detected in selected area.")

    all_values, all_counts = np.unique(pixels_inside, return_counts=True)
    total_inside = pixels_inside.size
    log.info("--- ALL classes inside lasso (%d pixels) ---", total_inside)
    for v, c in sorted(zip(all_values, all_counts), key=lambda x: -x[1]):
        name = CLASS_NAMES.get(int(v), f"Unknown-{v}")
        is_garment = "GARMENT" if int(v) in GARMENT_CLASS_IDS else "non-garment"
        log.info("  class %2d %-15s : %6d pixels (%5.1f%%) [%s]",
                 v, name, c, 100.0 * c / total_inside, is_garment)

    # Keep only garment classes (ignore background, hair, face, limbs)
    garment_pixels = pixels_inside[np.isin(pixels_inside, list(GARMENT_CLASS_IDS))]
    if garment_pixels.size == 0:
        log.warning("No garment class pixels inside lasso! Only non-garment classes found.")
        raise ValueError("No garment detected in selected area.")

    log.info("Garment pixels inside lasso: %d (%.1f%% of lasso)",
             garment_pixels.size, 100.0 * garment_pixels.size / total_inside)

    # Use DOMINANT garment class (mode) — the most frequent garment class in the lasso.
    # This prevents recoloring other garments that partially overlap the lasso
    # (e.g. lasso around t-shirt shouldn't recolor the skirt).
    values, counts = np.unique(garment_pixels, return_counts=True)
    log.info("--- All garment classes in lasso ---")
    for v, c in zip(values, counts):
        name = CLASS_NAMES.get(int(v), f"Unknown-{v}")
        log.info("  class %2d %-15s : %6d pixels (%.1f%%)",
                 int(v), name, int(c), 100.0 * c / garment_pixels.size)

    dominant_idx = np.argmax(counts)
    target_class = int(values[dominant_idx])
    target_name = CLASS_NAMES.get(target_class, f"Unknown-{target_class}")
    log.info(">>> SELECTED dominant class: %d (%s) with %d pixels (%.1f%% of garment pixels)",
             target_class, target_name, int(counts[dominant_idx]),
             100.0 * counts[dominant_idx] / garment_pixels.size)

    garment_mask = (segmentation_mask == target_class).astype(np.uint8)
    full_garment_pixels = int(garment_mask.sum())
    log.info("Full garment mask (entire image, class %d): %d pixels", target_class, full_garment_pixels)

    # Use connected components to find which part(s) of the garment the lasso touches.
    # This recolors the ENTIRE garment piece the user selected (not just the lasso intersection),
    # while avoiding recoloring other disconnected garments of the same class (e.g. in multi-person images).
    num_labels, labels = cv2.connectedComponents(garment_mask, connectivity=8)
    log.info("Connected components in garment mask: %d", num_labels - 1)

    # Find which component labels overlap with the lasso
    labels_in_lasso = labels[lasso_mask == 1]
    touched_labels = set(np.unique(labels_in_lasso)) - {0}  # 0 = background in connectedComponents
    log.info("Components touched by lasso: %s", touched_labels)

    if not touched_labels:
        log.warning("No garment component overlaps with lasso!")
        raise ValueError("No garment detected in selected area.")

    # Build mask from all touched components (full garment, not just lasso intersection)
    final_mask = np.zeros_like(garment_mask, dtype=np.uint8)
    for label_id in touched_labels:
        component_pixels = int((labels == label_id).sum())
        final_mask[labels == label_id] = 1
        log.info("  component %d: %d pixels", label_id, component_pixels)

    log.info("Garment mask (full connected region): %d pixels", int(final_mask.sum()))

    if final_mask.max() == 0:
        log.warning("Final mask is empty!")
        raise ValueError("No garment detected in selected area.")

    final_mask = _smooth_mask(final_mask)
    display_mask = final_mask.copy()

    log.info("Final mask (after smoothing): %d pixels (%.1f%% of full garment)",
             int(final_mask.sum()), 100.0 * int(final_mask.sum()) / max(full_garment_pixels, 1))
    log.info("--- MASK BUILDING DONE ---")
    return final_mask, display_mask
