"""
SegFormer B2 Clothes — semantic segmentation only (no YOLO/object detection).
Model: mattmdjaga/segformer_b2_clothes
Runs once per image at 512x512; returns full-image segmentation mask at original dimensions.
"""
import logging
import os
import time
from functools import lru_cache
from typing import Tuple

# Long timeout for 109MB model download; avoids Read timed out on HuggingFace CDN
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

log = logging.getLogger(__name__)

MODEL_ID = "mattmdjaga/segformer_b2_clothes"
INPUT_SIZE = 512  # model input resolution

# Garment class IDs for "mode" detection (exclude background, hair, face, limbs)
# From model config: 0 Background, 1 Hat, 2 Hair, 3 Sunglasses, 4 Upper-clothes, 5 Skirt, 6 Pants,
# 7 Dress, 8 Belt, 9 Left-shoe, 10 Right-shoe, 11 Face, 12 Left-leg, 13 Right-leg, 14 Left-arm, 15 Right-arm, 16 Bag, 17 Scarf
GARMENT_CLASS_IDS = frozenset({1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17})


@lru_cache(maxsize=1)
def _load_model() -> Tuple[AutoImageProcessor, AutoModelForSemanticSegmentation]:
    t0 = time.perf_counter()
    log.info("Loading SegFormer B2 Clothes (first time only): %s", MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, resume_download=True)
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID, resume_download=True)
    model.to(device)
    model.eval()
    log.info("SegFormer B2 Clothes loaded in %.1f s (device=%s)", time.perf_counter() - t0, device)
    return processor, model


def run_segmentation_on_image(image_bgr: np.ndarray) -> np.ndarray:
    """
    Run SegFormer on the full image. Resize to INPUT_SIZE for inference,
    then return segmentation mask at original image dimensions (H, W) with class IDs 0–17.
    """
    h_orig, w_orig = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize to model input (512x512) for inference
    resized = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    processor, model = _load_model()
    inputs = processor(images=resized, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t_infer = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    log.info("SegFormer inference in %.1f s", time.perf_counter() - t_infer)

    logits = outputs.logits  # (1, num_classes, H, W)
    # Upsample logits to input size (512x512)
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    seg_512 = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

    # Resize segmentation back to original image dimensions (nearest neighbor to keep class IDs)
    seg_orig = cv2.resize(
        seg_512,
        (w_orig, h_orig),
        interpolation=cv2.INTER_NEAREST,
    )
    return seg_orig


