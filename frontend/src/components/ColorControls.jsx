import React, { useMemo } from "react";
import { oklchToRGB, rgbToHex } from "../oklchRecolor.ts";

export function ColorControls({
  lightness,
  chroma,
  hue,
  onLightnessChange,
  onChromaChange,
  onHueChange,
}) {
  // Slider L is in 0..100; convert to OKLCH L in 0..1 for conversions.
  const testColor = useMemo(() => {
    const target = { L: lightness / 100, C: chroma, h: hue };
    const rgb = oklchToRGB(target);
    return rgbToHex(rgb);
  }, [lightness, chroma, hue]);

  return (
    <div className="bg-white/5 border border-slate-800 p-5 rounded-lg">
      <h3 className="text-sm font-semibold text-slate-100 mb-1.5 uppercase tracking-wide">
        Color Controls
      </h3>

      {/* Lightness Slider */}
      <div className="mb-3">
        <div className="flex justify-between items-center mb-1">
          <label className="text-sm font-medium text-slate-200">Lightness (L)</label>
          <span className="text-sm font-mono text-slate-100">{lightness.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0"
          max="100"
          value={lightness}
          onChange={(e) => onLightnessChange(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      {/* Chroma Slider */}
      <div className="mb-3">
        <div className="flex justify-between items-center mb-1">
          <label className="text-sm font-medium text-slate-200">Chroma (C)</label>
          <span className="text-sm font-mono text-slate-100">{chroma.toFixed(3)}</span>
        </div>
        <input
          type="range"
          min="0"
          max="0.4"
          step="0.001"
          value={chroma}
          onChange={(e) => onChromaChange(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      {/* Hue Slider */}
      <div className="mb-3">
        <div className="flex justify-between items-center mb-1">
          <label className="text-sm font-medium text-slate-200">Hue (h)</label>
          <span className="text-sm font-mono text-slate-100">{hue.toFixed(2)}°</span>
        </div>
        <input
          type="range"
          min="0"
          max="360"
          value={hue}
          onChange={(e) => onHueChange(Number(e.target.value))}
          className="w-full h-2 rounded-lg appearance-none cursor-pointer"
          style={{
            background:
              "linear-gradient(to right, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff, #ff00ff, #ff0000)",
          }}
        />
      </div>

      {/* Current Test Color Preview */}
      <div className="mt-3 p-2.5 bg-white/5 rounded-lg border border-slate-800">
        <p className="text-xs text-slate-300 mb-1">Test Color</p>
        <div
          className="w-full h-12 rounded-md border border-slate-700"
          style={{ backgroundColor: testColor }}
        />
        <p className="text-xs text-slate-200 mt-1 font-mono">{testColor}</p>
      </div>
    </div>
  );
}

