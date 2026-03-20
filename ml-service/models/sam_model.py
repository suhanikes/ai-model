import logging
import os
import time
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
import cv2

log = logging.getLogger(__name__)

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError as exc:
    raise ImportError(
        "segment-anything is required. Install from GitHub:\n"
        "pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from exc


PointList = List[Tuple[float, float]]


@lru_cache(maxsize=1)
def load_sam_predictor() -> SamPredictor:
    t0 = time.perf_counter()
    log.info("Loading SAM model (first time only)...")
    model_type = os.getenv("SAM_MODEL_TYPE", "vit_h")
    checkpoint_path = os.getenv("SAM_CHECKPOINT_PATH")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise RuntimeError(
            "SAM_CHECKPOINT_PATH is not set or does not exist. "
            "Download a SAM checkpoint (e.g. sam_vit_h_4b8939.pth) and set env var."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("SAM device=%s, type=%s", device, model_type)

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    log.info("SAM model loaded in %.1f s", time.perf_counter() - t0)
    return predictor


def sam_mask_from_lasso(
    image_bgr: np.ndarray,
    lasso_points: PointList,
) -> np.ndarray:
    """
    Generate a precise binary mask using SAM from free-form lasso points.

    We convert the polygon into a set of interior points and pass those as
    positive point prompts to SAM. This approach refines the loose user lasso
    into a precise object boundary.
    """
    if image_bgr.ndim != 3:
        raise ValueError("Expected color image (H, W, 3)")

    predictor = load_sam_predictor()

    # SAM expects RGB input
    t_set = time.perf_counter()
    image_rgb = image_bgr[:, :, ::-1].copy()
    predictor.set_image(image_rgb)
    log.info("SAM set_image done in %.1f ms", (time.perf_counter() - t_set) * 1000)

    # Build a dense interior-point set from the lasso polygon
    poly = np.array(lasso_points, dtype=np.float32)
    h, w = image_bgr.shape[:2]

    lasso_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(
        lasso_mask,
        [poly.astype(np.int32)],
        color=1,
    )

    ys, xs = np.where(lasso_mask == 1)
    if xs.size == 0:
        raise ValueError("Lasso polygon did not cover any pixels")

    # Subsample up to ~32 prompt points for efficiency
    idx = np.linspace(0, xs.size - 1, num=min(32, xs.size), dtype=int)
    xs_sampled = xs[idx]
    ys_sampled = ys[idx]
    point_coords = np.stack([xs_sampled, ys_sampled], axis=1).astype(
        np.float32
    )
    point_labels = np.ones(point_coords.shape[0], dtype=np.int32)
    log.info("SAM predict with %d prompt points...", point_coords.shape[0])

    t_pred = time.perf_counter()
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    log.info("SAM predict done in %.1f s", time.perf_counter() - t_pred)

    sam_mask = masks[0].astype(np.uint8)
    return sam_mask


