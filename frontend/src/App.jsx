import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ImageUploader } from "./components/ImageUploader.jsx";
import { LassoCanvas } from "./components/LassoCanvas.jsx";
import { ColorControls } from "./components/ColorControls.jsx";
import { useHistoryStack } from "./hooks/useHistoryStack.js";
import { lassoSegmentation, uploadImage } from "./api.js";
import { recolorGarmentOKLCH } from "./oklchRecolor.ts";
import { oklchToRGB, rgbToHex } from "./oklchRecolor.ts";

function Header() {
  return (
    <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
        <div className="flex items-center gap-2">
          <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-indigo-500 via-fuchsia-500 to-emerald-400" />
          <div className="flex flex-col">
            <span className="text-sm font-semibold tracking-tight text-white">
              Garment Recolor Studio
            </span>
            <span className="text-[11px] text-slate-400">
              Lasso-select + AI-powered garment masking
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function App() {
  const [imageMeta, setImageMeta] = useState(null);
  const [imageId, setImageId] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [maskPreview, setMaskPreview] = useState(null);
  const [lastLassoPoints, setLastLassoPoints] = useState(null);
  const colorDebounceRef = useRef(null);
  const recolorRequestIdRef = useRef(0);
  const baseCanvasRef = useRef(null);
  const baseImageDataRef = useRef(null);

  // OKLCH slider state (UI ranges: L in 0..100, C in 0..0.4, h in 0..360)
  const [lightness, setLightness] = useState(65);
  const [chroma, setChroma] = useState(0.1);
  const [hue, setHue] = useState(230);

  const targetOKLCH = useMemo(
    () => ({
      L: lightness / 100,
      C: chroma,
      h: hue,
    }),
    [lightness, chroma, hue],
  );

  const {
    current: editedImageUrl,
    push: pushHistory,
    undo,
    redo,
    canUndo,
    canRedo,
  } = useHistoryStack(null);

  const handleImageSelected = useCallback(async ({ file, url }) => {
    setMaskPreview(null);
    setLastLassoPoints(null);
    // Show image immediately so it appears while upload/segmentation runs
    setImageMeta({ file, url });
    pushHistory(url);
    setIsUploading(true);
    try {
      const data = await uploadImage(file);
      setImageMeta((prev) => (prev ? { ...prev, width: data.width, height: data.height } : prev));
      setImageId(data.imageId);

      // Prepare base ImageData for client-side recoloring
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const canvas = baseCanvasRef.current;
        if (!canvas) return;
        canvas.width = data.width;
        canvas.height = data.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, data.width, data.height);
        baseImageDataRef.current = ctx.getImageData(0, 0, data.width, data.height);
      };
      img.src = url;
    } catch (err) {
      console.error(err);
      const msg = err.response?.data?.error || err.response?.data?.details || err.message;
      alert(msg || "Failed to upload image. Please try again.");
      setImageId(null);
      // Keep image visible so user can retry
    } finally {
      setIsUploading(false);
    }
  }, [pushHistory]);

  const runRecolor = useCallback(
    async (lassoPoints, lch) => {
      if (!imageId || !lassoPoints || lassoPoints.length < 3) return;
      const requestId = ++recolorRequestIdRef.current;
      setIsProcessing(true);
      try {
        const res = await lassoSegmentation({
          imageId,
          lassoPoints,
          selectedColor: rgbToHex(oklchToRGB(lch)),
        });
        if (requestId !== recolorRequestIdRef.current) return;
        if (res.mask_preview) setMaskPreview(res.mask_preview);

        const baseImageData = baseImageDataRef.current;
        if (!baseImageData) {
          console.warn("No base image data available for recolor.");
          return;
        }

        // Decode garment_mask_b64 into Uint8Array
        const b64ToUint8 = (b64) => {
          const binary = atob(b64);
          const len = binary.length;
          const arr = new Uint8Array(len);
          for (let i = 0; i < len; i++) {
            arr[i] = binary.charCodeAt(i);
          }
          return arr;
        };

        const { garment_mask_b64, height, width } = res;
        if (!garment_mask_b64 || !height || !width) {
          console.warn("Missing garment mask from backend response.");
          return;
        }
        const maskArr = b64ToUint8(garment_mask_b64);

        // Run OKLCH recolor on client
        const result = recolorGarmentOKLCH(
          baseImageData.data,
          baseImageData.width,
          baseImageData.height,
          maskArr,
          lch,
        );

        const canvas = baseCanvasRef.current;
        if (!canvas) return;
        canvas.width = result.width;
        canvas.height = result.height;
        const ctx = canvas.getContext("2d");
        const outImageData = new ImageData(result.data, result.width, result.height);
        ctx.putImageData(outImageData, 0, 0);
        const url = canvas.toDataURL("image/png");
        pushHistory(url);
      } catch (err) {
        if (requestId !== recolorRequestIdRef.current) return;
        const data = err.response?.data;
        const msg = data?.error || err.message || "Recolor failed.";
        const details = data?.details;
        const fullMsg = details ? `${msg}\n\nDetails: ${details}` : msg;
        if (err.response?.status === 422) {
          alert(msg);
        } else {
          console.error("Recolor error:", err.response?.data ?? err);
          alert(fullMsg);
        }
      } finally {
        setIsProcessing(false);
      }
    },
    [imageId, pushHistory],
  );

  const handleLassoComplete = useCallback(async (lassoPoints) => {
    setLastLassoPoints(lassoPoints);
    await runRecolor(lassoPoints, targetOKLCH);
  }, [runRecolor, targetOKLCH]);

  // Real-time preview: when color changes, re-run recolor with last lasso (debounced)
  useEffect(() => {
    if (!lastLassoPoints || !imageId) return;
    if (colorDebounceRef.current) clearTimeout(colorDebounceRef.current);
    colorDebounceRef.current = setTimeout(() => {
      runRecolor(lastLassoPoints, targetOKLCH);
      colorDebounceRef.current = null;
    }, 400);
    return () => {
      if (colorDebounceRef.current) clearTimeout(colorDebounceRef.current);
    };
    // Intentionally omit lastLassoPoints/runRecolor to avoid double call on first lasso.
  }, [lightness, chroma, hue, imageId, runRecolor, targetOKLCH]);

  const liveImageUrl = editedImageUrl || imageMeta?.url || null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <Header />
      <canvas ref={baseCanvasRef} style={{ display: "none" }} />
      <main className="mx-auto flex max-w-6xl flex-col gap-6 px-4 pb-10 pt-6 md:flex-row">
        <section className="flex-1 space-y-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h2 className="mb-3 text-sm font-semibold text-slate-100">
              Workflow
            </h2>
            <ol className="space-y-1 text-xs text-slate-300">
              <li>1. Upload a garment image.</li>
              <li>2. Adjust target LCH sliders.</li>
              <li>3. Draw a lasso around the clothing area.</li>
              <li>4. Release to run AI masking and recolor.</li>
            </ol>
          </div>
          <div className="space-y-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <ImageUploader onImageSelected={handleImageSelected} />
            <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-slate-200">
                    Color controls
                  </span>
                  <span className="rounded-full bg-slate-800 px-3 py-1 text-[11px] text-slate-300">
                    Real-time preview
                  </span>
                </div>
                <div className="space-y-3">
                  {imageMeta?.url ? (
                    <ColorControls
                      lightness={lightness}
                      chroma={chroma}
                      hue={hue}
                      onLightnessChange={setLightness}
                      onChromaChange={setChroma}
                      onHueChange={setHue}
                    />
                  ) : (
                    <p className="text-[11px] text-slate-400">
                      Upload an image to enable LCH sliders.
                    </p>
                  )}
                  <p className="text-[11px] text-slate-400">
                    Your recolor target is controlled only by the LCH sliders.
                  </p>
                </div>
              </div>
              <div className="flex flex-col justify-between gap-4">
                <div className="space-y-2">
                  <span className="text-sm font-medium text-slate-200">
                    History
                  </span>
                  <div className="inline-flex gap-2">
                    <button
                      type="button"
                      onClick={undo}
                      disabled={!canUndo}
                      className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      Undo
                    </button>
                    <button
                      type="button"
                      onClick={redo}
                      disabled={!canRedo}
                      className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      Redo
                    </button>
                  </div>
                </div>
                <button
                  type="button"
                  disabled={!editedImageUrl}
                  onClick={() => {
                    if (!editedImageUrl) return;
                    const link = document.createElement("a");
                    link.href = editedImageUrl;
                    link.download = "garment-recolor.png";
                    link.click();
                  }}
                  className="mt-auto inline-flex items-center justify-center rounded-md bg-emerald-500 px-4 py-2 text-xs font-semibold text-emerald-950 shadow-sm shadow-emerald-500/40 hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Download edited image
                </button>
              </div>
            </div>
            {isUploading && (
              <p className="text-xs text-indigo-300">
                Uploading image and running segmentation (one-time)...
              </p>
            )}
          </div>
        </section>
        <section className="flex-1 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-slate-100">
                Left: Image + lasso · Right: Recolored preview
              </h2>
              <p className="text-xs text-slate-400">
                Draw a lasso around the garment. SegFormer detects the garment class;
                preview updates when you change the color.
              </p>
            </div>
            {isProcessing && (
              <span className="inline-flex items-center gap-2 rounded-full border border-indigo-500/50 bg-indigo-500/10 px-3 py-1 text-[11px] text-indigo-200">
                <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-300" />
                Recoloring...
              </span>
            )}
          </div>
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-2">
              <p className="mb-2 text-[11px] font-medium text-slate-400">Left: Image with lasso overlay</p>
              <LassoCanvas
                imageUrl={imageMeta?.url ?? null}
                onLassoComplete={handleLassoComplete}
                maskPreview={maskPreview}
                isProcessing={isProcessing}
              />
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-2">
              <p className="mb-2 text-[11px] font-medium text-slate-400">Right: Recolored preview</p>
              <div className="relative flex min-h-[320px] items-center justify-center overflow-hidden rounded-lg bg-slate-800/50">
                {liveImageUrl ? (
                  <img
                    src={liveImageUrl}
                    alt="Recolored preview"
                    className="max-h-[520px] w-full object-contain"
                  />
                ) : (
                  <p className="text-sm text-slate-500">Recolored result appears here</p>
                )}
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

