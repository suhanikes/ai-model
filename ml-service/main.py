import base64
import io
import json as _json
import logging
import time
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.formparsers import MultiPartParser

from services.mask_processing import build_final_garment_mask_from_segmentation

from models.segformer_b2_clothes import run_segmentation_on_image

# Starlette defaults to 1024KB; segmentation_mask_b64 and images can exceed that.
# Monkey-patch before any request is processed. 64MB comfortably supports high-res images + large masks.
MultiPartParser.max_part_size = 64 * 1024 * 1024  # 64MB
MultiPartParser.max_file_size = 64 * 1024 * 1024  # 64MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PointList = List[Tuple[float, float]]

app = FastAPI(title="Garment Recolor ML Service (SegFormer B2 Clothes)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    if exc.status_code >= 400:
        log.warning("HTTP %s: %s", exc.status_code, exc.detail)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def load_image_from_upload(upload: UploadFile) -> tuple[np.ndarray, float]:
    """Load image; return (image_bgr, scale) with scale used if we resized (for lasso coord scaling)."""
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    upload.file.seek(0)
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    img = np.array(pil)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr, 1.0


def encode_mask_preview(mask: np.ndarray) -> str:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    colored = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    colored[..., 0] = 255
    colored[..., 3] = (mask > 0).astype(np.uint8) * 160
    pil = Image.fromarray(colored, mode="RGBA")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_png_b64(image_bgr: np.ndarray) -> str:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_segmentation_mask(b64: str, height: int, width: int) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width)


@app.get("/health")
async def health():
    return {"ok": True, "service": "ml-service"}


@app.post("/segmentation/run-once")
async def segmentation_run_once(image: UploadFile = File(...)):
    """
    Run SegFormer B2 Clothes on the full image once. Resize to 512x512 for inference,
    return segmentation mask at original dimensions. No lasso; used at upload time.
    """
    t0 = time.perf_counter()
    log.info("=== Run-once segmentation request ===")
    image_bgr, _ = load_image_from_upload(image)
    h, w = image_bgr.shape[:2]
    t1 = time.perf_counter()
    seg_mask = run_segmentation_on_image(image_bgr)
    log.info("Segmentation done in %.1f s", time.perf_counter() - t1)

    # Log detected classes in the full image
    CLASS_NAMES = {
        0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses",
        4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress",
        8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
        12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
        16: "Bag", 17: "Scarf",
    }
    vals, cnts = np.unique(seg_mask, return_counts=True)
    total_px = seg_mask.size
    log.info("--- Classes detected in full image (%dx%d = %d px) ---", w, h, total_px)
    for v, c in sorted(zip(vals, cnts), key=lambda x: -x[1]):
        log.info("  class %2d %-15s : %7d px (%5.1f%%)",
                 v, CLASS_NAMES.get(int(v), "?"), c, 100.0 * c / total_px)

    mask_bytes = seg_mask.astype(np.uint8).tobytes()
    mask_b64 = base64.b64encode(mask_bytes).decode("ascii")
    log.info("Mask b64 length: %d chars", len(mask_b64))
    log.info("=== Run-once total: %.1f s ===", time.perf_counter() - t0)
    return {
        "segmentation_mask_b64": mask_b64,
        "height": int(seg_mask.shape[0]),
        "width": int(seg_mask.shape[1]),
    }


@app.post("/segmentation/recolor")
async def segmentation_recolor(
    image: UploadFile = File(...),
    segmentation_mask_b64: str = Form(...),
    mask_height: int = Form(...),
    mask_width: int = Form(...),
    lasso_points: str = Form(...),
    selected_color: str = Form("#ff3366"),
):
    """
    Build the final garment mask using precomputed segmentation + lasso.
    Returns the binary garment mask (no recolor).
    """
    t0 = time.perf_counter()
    log.info("=== Recolor/mask request (stored mask) ===")
    log.info("  mask dims: %d x %d, lasso_points length: %d chars, color: %s",
             mask_height, mask_width, len(lasso_points), selected_color)
    log.info("  segmentation_mask_b64 length: %d chars", len(segmentation_mask_b64))

    try:
        return await _do_recolor(
            image, segmentation_mask_b64, mask_height, mask_width, lasso_points, selected_color, t0
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Mask build failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


async def _do_recolor(image, segmentation_mask_b64, mask_height, mask_width, lasso_points, selected_color, t0):
    # Parse lasso points (allow both {"x","y"} and {"X","Y"} for compatibility)
    try:
        pts_raw = _json.loads(lasso_points)
        if not isinstance(pts_raw, list) or len(pts_raw) < 3:
            raise ValueError("lasso_points must be a JSON array of at least 3 points")
        lasso: PointList = []
        for p in pts_raw:
            if not isinstance(p, dict):
                raise ValueError(f"Each point must be an object with x,y: got {type(p)}")
            x = p.get("x") if "x" in p else p.get("X")
            y = p.get("y") if "y" in p else p.get("Y")
            if x is None or y is None:
                raise ValueError(f"Point must have x and y: {p}")
            lasso.append((float(x), float(y)))
    except _json.JSONDecodeError as exc:
        log.warning("Invalid lasso_points JSON: %s", exc)
        raise HTTPException(status_code=400, detail=f"Invalid lasso_points JSON: {exc}")
    except (ValueError, TypeError) as exc:
        log.warning("Invalid lasso_points: %s", exc)
        raise HTTPException(status_code=400, detail=f"Invalid lasso_points: {exc}")

    try:
        image_bgr, _ = load_image_from_upload(image)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Failed to load image")
        raise HTTPException(status_code=400, detail=f"Failed to load image: {exc}")

    try:
        seg_mask = decode_segmentation_mask(segmentation_mask_b64, mask_height, mask_width)
    except Exception as exc:
        log.warning("Failed to decode segmentation mask: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid segmentation mask or dimensions (h={mask_height}, w={mask_width}): {exc}",
        )

    try:
        final_mask, display_mask = build_final_garment_mask_from_segmentation(
            image_bgr.shape[:2], lasso, seg_mask
        )
    except ValueError as e:
        if "No garment detected" in str(e):
            raise HTTPException(status_code=422, detail="No garment detected in selected area.")
        log.warning("Mask build error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    if final_mask.max() == 0:
        raise HTTPException(status_code=422, detail="No garment detected in selected area.")

    # Return the binary garment mask (uint8 0/1), plus a preview PNG for overlay.
    mask_bytes = final_mask.astype(np.uint8).tobytes()
    garment_mask_b64 = base64.b64encode(mask_bytes).decode("ascii")
    mask_preview_b64 = encode_mask_preview(display_mask)
    log.info("Mask ready; bytes=%d, b64_len=%d", len(mask_bytes), len(garment_mask_b64))
    log.info("=== Mask build total: %.1f s ===", time.perf_counter() - t0)
    return {
        "garment_mask_b64": garment_mask_b64,
        "height": int(final_mask.shape[0]),
        "width": int(final_mask.shape[1]),
        "mask_preview_png": mask_preview_b64,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
