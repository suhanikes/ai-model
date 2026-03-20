import express from "express";
import cors from "cors";
import multer from "multer";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";
import dotenv from "dotenv";
import { v4 as uuidv4 } from "uuid";
import axios from "axios";
import FormData from "form-data";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 4000;

app.use(
  cors({
    origin: process.env.CORS_ORIGIN || "http://localhost:5173",
  }),
);
app.use(express.json({ limit: "25mb" }));

const uploadsDir = path.join(__dirname, "..", "uploads");
const staticDir = path.join(__dirname, "..", "static");
const editedDir = path.join(staticDir, "edited");
const maskDir = path.join(staticDir, "masks");

for (const dir of [uploadsDir, staticDir, editedDir, maskDir]) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

app.use("/static", express.static(staticDir));

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname || ".png");
    cb(null, `${Date.now()}-${uuidv4()}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 10 * 1024 * 1024,
  },
});

const imageRegistry = new Map();
const mlBaseUrl = (process.env.ML_SERVICE_URL || "http://127.0.0.1:8000").replace(/\/segmentation\/recolor\/?$/, "");

function getBaseUrl() {
  const fromEnv = process.env.PUBLIC_BASE_URL;
  if (fromEnv) return fromEnv.replace(/\/$/, "");
  return `http://localhost:${PORT}`;
}

app.post("/api/upload", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image uploaded" });
  }

  const id = uuidv4();
  const filePath = req.file.path;

  // Always store the image so upload succeeds; then run segmentation (optional)
  const meta = {
    id,
    path: filePath,
    originalName: req.file.originalname,
    maskB64: null,
    maskHeight: null,
    maskWidth: null,
  };
  imageRegistry.set(id, meta);

  try {
    const formData = new FormData();
    formData.append("image", fs.createReadStream(filePath), {
      filename: path.basename(filePath),
      contentType: req.file.mimetype || "image/jpeg",
    });
    const runOnceRes = await axios.post(`${mlBaseUrl}/segmentation/run-once`, formData, {
      headers: formData.getHeaders(),
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      responseType: "json",
      timeout: 120000,
    });
    const { segmentation_mask_b64: maskB64, height: h, width: w } = runOnceRes.data || {};
    if (maskB64 != null && h != null && w != null) {
      meta.maskB64 = maskB64;
      meta.maskHeight = h;
      meta.maskWidth = w;
    }
  } catch (err) {
    console.error("Run-once segmentation error", err?.response?.data || err.message);
  }

  const dims = meta.maskHeight != null && meta.maskWidth != null
    ? { width: Number(meta.maskWidth), height: Number(meta.maskHeight) }
    : {};
  return res.json({
    imageId: id,
    filename: path.basename(filePath),
    ...dims,
  });
});

app.post("/api/lasso-segmentation", async (req, res) => {
  const body = req.body || {};
  let { imageId, lasso_points, selected_color } = body;

  if (!imageId) {
    return res.status(400).json({
      error: "imageId is required. Re-upload the image and try again.",
    });
  }

  // Normalize lasso_points: allow array of {x,y} or flat [x,y,x,y,...]
  if (Array.isArray(lasso_points) && lasso_points.length >= 6 && typeof lasso_points[0] === "number") {
    const points = [];
    for (let i = 0; i < lasso_points.length; i += 2) {
      points.push({ x: lasso_points[i], y: lasso_points[i + 1] });
    }
    lasso_points = points;
  }
  if (!Array.isArray(lasso_points) || lasso_points.length < 3) {
    return res.status(400).json({
      error: "lasso_points must be an array of at least 3 points (e.g. [{x,y}, ...]).",
    });
  }

  const meta = imageRegistry.get(imageId);
  if (!meta) {
    return res.status(404).json({ error: "Image not found. Re-upload it." });
  }

  // If no mask yet (e.g. run-once failed at upload), run segmentation now
  if (!meta.maskB64) {
    try {
      const runOnceForm = new FormData();
      runOnceForm.append("image", fs.createReadStream(meta.path), {
        filename: path.basename(meta.path),
        contentType: "image/jpeg",
      });
      const runOnceRes = await axios.post(`${mlBaseUrl}/segmentation/run-once`, runOnceForm, {
        headers: runOnceForm.getHeaders(),
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
        responseType: "json",
        timeout: 120000,
      });
      const { segmentation_mask_b64: maskB64, height: h, width: w } = runOnceRes.data || {};
      if (maskB64 != null && h != null && w != null) {
        meta.maskB64 = maskB64;
        meta.maskHeight = h;
        meta.maskWidth = w;
      }
    } catch (err) {
      console.error("Run-once (on-demand) error", err?.response?.data || err.message);
      return res.status(502).json({
        error: "Segmentation failed. Is the ML service running on port 8000?",
        details: err?.response?.data?.detail ?? err?.message,
      });
    }
    if (!meta.maskB64) {
      return res.status(502).json({ error: "Could not get segmentation mask from ML service." });
    }
  }

  const mlUrl = `${mlBaseUrl}/segmentation/recolor`;
  try {
    const formData = new FormData();
    // Scalars first
    formData.append("segmentation_mask_b64", meta.maskB64);
    formData.append("mask_height", String(meta.maskHeight));
    formData.append("mask_width", String(meta.maskWidth));
    formData.append("lasso_points", JSON.stringify(lasso_points));
    formData.append("selected_color", selected_color || "#ff3366");
    // Image last
    formData.append("image", fs.createReadStream(meta.path), {
      filename: path.basename(meta.path),
      contentType: path.extname(meta.path).toLowerCase() === ".png" ? "image/png" : "image/jpeg",
    });

    const response = await axios.post(mlUrl, formData, {
      headers: formData.getHeaders(),
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      responseType: "json",
      timeout: 60000,
    });

    const { garment_mask_b64, height, width, mask_preview_png } = response.data || {};

    if (!garment_mask_b64 || !height || !width) {
      return res
        .status(500)
        .json({ error: "ML service did not return garment mask" });
    }

    const baseUrl = getBaseUrl();
    let maskUrl = null;
    if (mask_preview_png) {
      const maskFilename = `${Date.now()}-${imageId}-mask.png`;
      const maskPath = path.join(maskDir, maskFilename);
      const maskBuffer = Buffer.from(mask_preview_png, "base64");
      fs.writeFileSync(maskPath, maskBuffer);
      maskUrl = `${baseUrl}/static/masks/${maskFilename}`;
    }

    return res.json({
      garment_mask_b64,
      height,
      width,
      mask_preview: maskUrl,
    });
  } catch (err) {
    const status = err?.response?.status;
    const detail = err?.response?.data?.detail;
    console.error("Error from ML service", status, detail ?? err?.response?.data ?? err.message);
    if (status === 422 && typeof detail === "string" && detail.includes("No garment detected")) {
      return res.status(422).json({ error: "No garment detected in selected area." });
    }
    if (status === 400 && detail != null) {
      return res.status(400).json({
        error: "Invalid request to recolor service",
        details: typeof detail === "string" ? detail : JSON.stringify(detail),
      });
    }
    const isConnectionError =
      err.code === "ECONNREFUSED" ||
      err.code === "ENOTFOUND" ||
      err.message?.includes("Network Error");
    const details = isConnectionError
      ? "ML service unreachable. Start the Python ML service (port 8000)."
      : detail ?? err?.response?.data?.details ?? err?.message;
    return res.status(status && status >= 400 ? status : 500).json({
      error: "Failed to process segmentation request",
      details: typeof details === "string" ? details : JSON.stringify(details),
    });
  }
});

app.get("/health", (_req, res) => {
  res.json({ ok: true, service: "node-backend" });
});

app.listen(PORT, () => {
  console.log(`Node backend listening on http://localhost:${PORT}`);
});

