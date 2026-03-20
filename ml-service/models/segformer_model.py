import logging
import os
import time
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

log = logging.getLogger(__name__)

FASHION_MODEL_ID = os.getenv(
    "SEGFORMER_FASHION_MODEL_ID",
    "sayeed99/segformer-b3-fashion",
)

# Label map from the model card:
# 0: Unlabelled
# 1: Shirt/Blouse
# 2: Top/T-shirt/Sweatshirt
# 3: Sweater
# 4: Cardigan
# 5: Jacket
# 6: Vest
# 7: Pants
# 8: Shorts
# 9: Skirt
# 10: Coat
# 11: Dress
# 12: Jumpsuit
# 13: Cape
# 28: Hood
# 29: Collar
# 30: Lapel
# 32: Sleeve
# 33: Pocket
# 34: Neckline
#
# All clothing: tops (shirt, t-shirt, etc.) + bottoms (pants, shorts, skirt) + details.
# SegFormer alone is used for masking; works for both upper and lower garments.
GARMENT_LABELS = np.array(
    [
        1,  # Shirt/Blouse
        2,  # Top/T-shirt/Sweatshirt
        3,  # Sweater
        4,  # Cardigan
        5,  # Jacket
        6,  # Vest
        7,  # Pants
        8,  # Shorts
        9,  # Skirt
        10,  # Coat
        11,  # Dress
        12,  # Jumpsuit
        13,  # Cape
        28,  # Hood
        29,  # Collar
        30,  # Lapel
        32,  # Sleeve
        33,  # Pocket
        34,  # Neckline
    ],
    dtype=np.int32,
)


@lru_cache(maxsize=1)
def load_segformer() -> Tuple[AutoImageProcessor, AutoModelForSemanticSegmentation]:
    t0 = time.perf_counter()
    log.info("Loading SegFormer model (first time only): %s", FASHION_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(FASHION_MODEL_ID)
    model = AutoModelForSemanticSegmentation.from_pretrained(FASHION_MODEL_ID)
    model.to(device)
    model.eval()
    log.info("SegFormer loaded in %.1f s (device=%s)", time.perf_counter() - t0, device)
    return processor, model


def _run_segformer_and_get_seg(image_bgr: np.ndarray) -> np.ndarray:
    """Run SegFormer and return the integer segmentation map (H, W) with class IDs."""
    processor, model = load_segformer()
    image_rgb = image_bgr[:, :, ::-1]
    inputs = processor(images=image_rgb, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t_infer = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    log.info("SegFormer inference in %.1f s", time.perf_counter() - t_infer)
    logits = outputs.logits  # (1, num_classes, H, W)
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
    return seg


def clothing_mask_from_segformer(
    image_bgr: np.ndarray,
    allowed_label_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Run SegFormer and return a binary mask of garment pixels.
    If allowed_label_ids is set (from YOLOS detection), keep only those SegFormer labels
    for better accuracy. Otherwise keep all GARMENT_LABELS.
    """
    seg = _run_segformer_and_get_seg(image_bgr)
    labels_to_keep = np.array(allowed_label_ids, dtype=np.int32) if allowed_label_ids else GARMENT_LABELS
    mask = np.isin(seg, labels_to_keep)
    clothing_mask = mask.astype(np.uint8)
    log.info(
        "SegFormer garment pixels: %d (labels=%s)",
        int(clothing_mask.sum()),
        allowed_label_ids if allowed_label_ids else "all",
    )
    return clothing_mask

