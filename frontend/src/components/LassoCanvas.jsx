import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Stage, Layer, Image as KonvaImage, Line } from "react-konva";

function useImage(url) {
  const [image, setImage] = useState(null);

  useEffect(() => {
    if (!url) {
      setImage(null);
      return;
    }
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.onload = () => setImage(img);
    img.src = url;
  }, [url]);

  return image;
}

export function LassoCanvas({
  imageUrl,
  onLassoComplete,
  maskPreview,
  isProcessing,
}) {
  const image = useImage(imageUrl);
  const [points, setPoints] = useState([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const stageRef = useRef(null);

  const [stageSize, setStageSize] = useState({ width: 800, height: 600 });

  useEffect(() => {
    const handleResize = () => {
      const container = document.getElementById("canvas-container");
      if (!container) return;
      const width = container.offsetWidth;
      const height = container.offsetHeight || width * 0.75;
      setStageSize({ width, height });
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleMouseDown = useCallback(
    (e) => {
      if (!image || isProcessing) return;
      const stage = e.target.getStage();
      const pos = stage.getPointerPosition();
      if (!pos) return;
      setIsDrawing(true);
      setPoints([pos.x, pos.y]);
    },
    [image, isProcessing],
  );

  const handleMouseMove = useCallback(
    (e) => {
      if (!isDrawing || !image || isProcessing) return;
      const stage = e.target.getStage();
      const pos = stage.getPointerPosition();
      if (!pos) return;
      setPoints((prev) => [...prev, pos.x, pos.y]);
    },
    [isDrawing, image, isProcessing],
  );

  const handleMouseUp = useCallback(() => {
    if (!isDrawing || !image) return;
    setIsDrawing(false);
    if (points.length < 6) {
      setPoints([]);
      return;
    }

    const stage = stageRef.current;
    if (!stage) return;
    const scaleX = image.width / stage.width();
    const scaleY = image.height / stage.height();

    const lassoPoints = [];
    for (let i = 0; i < points.length; i += 2) {
      lassoPoints.push({
        x: points[i] * scaleX,
        y: points[i + 1] * scaleY,
      });
    }
    onLassoComplete(lassoPoints);
  }, [image, isDrawing, onLassoComplete, points]);

  const maskPreviewImage = useImage(maskPreview || null);

  const scaledImageProps = useMemo(() => {
    if (!image) return null;
    const aspect = image.width / image.height;
    let width = stageSize.width;
    let height = width / aspect;
    if (height > stageSize.height) {
      height = stageSize.height;
      width = height * aspect;
    }
    const offsetX = (stageSize.width - width) / 2;
    const offsetY = (stageSize.height - height) / 2;
    return { width, height, x: offsetX, y: offsetY };
  }, [image, stageSize.height, stageSize.width]);

  return (
    <div className="relative h-full w-full rounded-xl border border-slate-800 bg-slate-900/60">
      <div
        id="canvas-container"
        className="relative h-[520px] w-full overflow-hidden rounded-xl"
      >
        <Stage
          width={stageSize.width}
          height={stageSize.height}
          ref={stageRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          <Layer>
            {image && scaledImageProps && (
              <KonvaImage image={image} {...scaledImageProps} />
            )}
            {maskPreviewImage && scaledImageProps && (
              <KonvaImage
                image={maskPreviewImage}
                {...scaledImageProps}
                opacity={0.5}
              />
            )}
            {points.length > 0 && (
              <Line
                points={points}
                stroke="#4f46e5"
                strokeWidth={2}
                tension={0.4}
                lineCap="round"
                lineJoin="round"
              />
            )}
          </Layer>
        </Stage>
      </div>
      {!image && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-slate-500">
          Upload an image to start lasso selection
        </div>
      )}
    </div>
  );
}

