"""
YOLOS-Fashionpedia object detection to identify which garment(s) are in the user's lasso.
Maps detections to SegFormer label IDs for accurate masking.
See: https://huggingface.co/valentinafeve/yolos-fashionpedia
"""

import logging
import time
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

YOLOS_MODEL_ID = "valentinafeve/yolos-fashionpedia"

# Map YOLOS-Fashionpedia label names to SegFormer-b3-fashion label IDs.
# SegFormer: 1=Shirt/Blouse, 2=Top/T-shirt, 3=Sweater, 4=Cardigan, 5=Jacket, 6=Vest,
# 7=Pants, 8=Shorts, 9=Skirt, 10=Coat, 11=Dress, 12=Jumpsuit, 13=Cape,
# 28=Hood, 29=Collar, 30=Lapel, 31=Epaulette, 32=Sleeve, 33=Pocket, 34=Neckline
YOLOS_TO_SEGFORMER_LABELS = {
    "shirt, blouse": [1],
    "top, t-shirt, sweatshirt": [2, 32, 34],  # top + sleeve + neckline
    "sweater": [3],
    "cardigan": [4],
    "jacket": [4, 5],  # cardigan/jacket; SegFormer 5=Jacket
    "vest": [6],
    "pants": [7],
    "shorts": [8],
    "skirt": [9],
    "coat": [10],
    "dress": [11],
    "jumpsuit": [12],
    "cape": [13],
    "hood": [28],
    "collar": [29],
    "lapel": [30],
    "epaulette": [31],
    "sleeve": [32],
    "pocket": [33],
    "neckline": [34],
}
# Fix jacket -> 5 only
YOLOS_TO_SEGFORMER_LABELS["jacket"] = [5]


@lru_cache(maxsize=1)
def _load_detector():
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    detector = pipeline(
        task="object-detection",
        model=YOLOS_MODEL_ID,
        device=device,
    )
    return detector


def _box_overlaps_lasso(box: dict, poly_mask: np.ndarray) -> float:
    """Return overlap ratio: area of box inside poly_mask / area of box."""
    xmin = int(round(box["xmin"]))
    ymin = int(round(box["ymin"]))
    xmax = int(round(box["xmax"]))
    ymax = int(round(box["ymax"]))
    h, w = poly_mask.shape
    xmin, xmax = max(0, xmin), min(w, xmax)
    ymin, ymax = max(0, ymin), min(h, ymax)
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    box_area = (xmax - xmin) * (ymax - ymin)
    if box_area == 0:
        return 0.0
    crop = poly_mask[ymin:ymax, xmin:xmax]
    inside = int(crop.sum())
    return inside / box_area


def detect_garments_in_region(
    image_bgr: np.ndarray,
    poly_mask: np.ndarray,
    score_threshold: float = 0.2,
    overlap_threshold: float = 0.3,
) -> List[Tuple[str, float]]:
    """
    Run YOLOS-Fashionpedia on the image and return detections that overlap the lasso.
    Returns list of (label_name, score) sorted by score descending.
    """
    t0 = time.perf_counter()
    detector = _load_detector()
    image_rgb = image_bgr[:, :, ::-1]
    # Pipeline expects PIL or path; pass as PIL
    from PIL import Image
    pil_image = Image.fromarray(image_rgb)
    try:
        results = detector(pil_image, threshold=score_threshold)
    except TypeError:
        results = detector(pil_image)
    log.info("YOLOS detection finished in %.1f s (%d raw detections)", time.perf_counter() - t0, len(results or []))

    if not results:
        return []

    in_region = []
    for r in results:
        if float(r.get("score", 0)) < score_threshold:
            continue
        label = r.get("label", "")
        score = float(r.get("score", 0))
        box = r.get("box", {})
        if not box or label not in YOLOS_TO_SEGFORMER_LABELS:
            continue
        overlap = _box_overlaps_lasso(box, poly_mask)
        if overlap >= overlap_threshold:
            in_region.append((label, score))

    # Dedupe by label, keep max score
    best = {}
    for label, score in in_region:
        if label not in best or score > best[label]:
            best[label] = score
    out = sorted(best.items(), key=lambda x: -x[1])
    log.info("YOLOS garments in lasso: %s", [x[0] for x in out])
    return out


def yolos_labels_to_segformer_ids(detected_labels: List[Tuple[str, float]]) -> List[int]:
    """Map YOLOS label names to a flat list of SegFormer label IDs (for masking)."""
    segformer_ids = []
    seen = set()
    for label, _ in detected_labels:
        ids = YOLOS_TO_SEGFORMER_LABELS.get(label, [])
        for i in ids:
            if i not in seen:
                seen.add(i)
                segformer_ids.append(i)
    return segformer_ids
