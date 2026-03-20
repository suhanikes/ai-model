export type RGB = {
  r: number;
  g: number;
  b: number;
};

export type OKLab = {
  L: number;
  a: number;
  b: number;
};

export type OKLCH = {
  L: number;
  C: number;
  h: number;
};

export type DominanceScores = {
  L: number;
  C: number;
  H: number;
  primary: "L" | "C" | "H";
  secondary: "L" | "C" | "H";
};

// === sRGB -> Linear
function srgbToLinear(value: number): number {
  const v = Math.max(0, Math.min(255, value)) / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

// === RGB -> OKLab
export function rgbToOKLab(rgb: RGB): OKLab {
  const r = srgbToLinear(rgb.r);
  const g = srgbToLinear(rgb.g);
  const b = srgbToLinear(rgb.b);

  const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
  const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
  const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

  const l_ = Math.cbrt(l);
  const m_ = Math.cbrt(m);
  const s_ = Math.cbrt(s);

  return {
    L: 0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
    a: 1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
    b: 0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_,
  };
}

export function okLabToOKLCH(oklab: OKLab): OKLCH {
  const C = Math.hypot(oklab.a, oklab.b);
  let h = 0;
  if (C > 0.0001) {
    h = (Math.atan2(oklab.b, oklab.a) * 180) / Math.PI;
    if (h < 0) h += 360;
  }
  return { L: oklab.L, C, h };
}

export function rgbToOKLCH(rgb: RGB): OKLCH {
  return okLabToOKLCH(rgbToOKLab(rgb));
}

export function okLCHToOKLab(lch: OKLCH): OKLab {
  const { L, C, h } = lch;
  const hRad = (h * Math.PI) / 180;
  const a = C * Math.cos(hRad);
  const b = C * Math.sin(hRad);
  return { L, a, b };
}

export function okLabToRGB(oklab: OKLab): RGB {
  const l_ = oklab.L + 0.3963377774 * oklab.a + 0.2158037573 * oklab.b;
  const m_ = oklab.L - 0.1055613458 * oklab.a - 0.0638541728 * oklab.b;
  const s_ = oklab.L - 0.0894841775 * oklab.a - 1.291485548 * oklab.b;

  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;

  const r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  const b = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s;

  const linearToSRGB = (value: number): number => {
    const v = Math.max(0, Math.min(1, value));
    return v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
  };

  return {
    r: Math.round(linearToSRGB(r) * 255),
    g: Math.round(linearToSRGB(g) * 255),
    b: Math.round(linearToSRGB(b) * 255),
  };
}

export function oklchToRGB(lch: OKLCH): RGB {
  return okLabToRGB(okLCHToOKLab(lch));
}

export function rgbToHex(rgb: RGB): string {
  const toHex = (n: number) => {
    const clamped = Math.max(0, Math.min(255, Math.round(n)));
    return clamped.toString(16).padStart(2, "0");
  };
  return `#${toHex(rgb.r)}${toHex(rgb.g)}${toHex(rgb.b)}`;
}

// Dominance scoring
const L_NEUTRAL = 0.65;
const C_NEUTRAL = 0.1;
const H_WARM_CENTER = 50;
const H_COOL_CENTER = 230;

function hueDistance(h1: number, h2: number): number {
  const diff = Math.abs(h1 - h2);
  return Math.min(diff, 360 - diff);
}

export function computeDominanceScores(lch: OKLCH): DominanceScores {
  const lScore = Math.abs(lch.L - L_NEUTRAL);
  const cScore = Math.abs(lch.C - C_NEUTRAL);
  const warmDist = hueDistance(lch.h, H_WARM_CENTER);
  const coolDist = hueDistance(lch.h, H_COOL_CENTER);
  const hScore = Math.max(1 - warmDist / 180, 1 - coolDist / 180);

  const rawScores = { L: lScore, C: cScore, H: hScore };
  const sorted = Object.entries(rawScores)
    .sort((a, b) => b[1] - a[1])
    .map((entry) => entry[0]) as ("L" | "C" | "H")[];

  return {
    L: rawScores.L,
    C: rawScores.C,
    H: rawScores.H,
    primary: sorted[0],
    secondary: sorted[1],
  };
}

export interface RecolorResult {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  previewHex?: string;
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

const MIN_CHROMA_FOR_HUE = 0.03;
const DARK_GARMENT_MEAN_L_THRESHOLD = 0.42;
const NEAR_NEUTRAL_DARK_MEAN_C_THRESHOLD = 0.06;

function computeMaskedMeanLCH(
  srcData: Uint8ClampedArray,
  mask: Uint8Array | Uint8ClampedArray,
  pixelCount: number,
): OKLCH {
  let sumL = 0;
  let sumC = 0;
  let sumSin = 0;
  let sumCos = 0;
  let hueCount = 0;
  let count = 0;

  for (let i = 0; i < pixelCount; i++) {
    if (mask[i] === 0) continue;
    const idx = i * 4;
    const lch = rgbToOKLCH({
      r: srcData[idx],
      g: srcData[idx + 1],
      b: srcData[idx + 2],
    });

    sumL += lch.L;
    sumC += lch.C;
    count++;

    if (lch.C >= MIN_CHROMA_FOR_HUE) {
      const rad = (lch.h * Math.PI) / 180;
      sumSin += Math.sin(rad);
      sumCos += Math.cos(rad);
      hueCount++;
    }
  }

  if (count === 0) return { L: 0, C: 0, h: 0 };

  const meanL = sumL / count;
  const meanC = sumC / count;
  let meanH = hueCount > 0 ? (Math.atan2(sumSin, sumCos) * 180) / Math.PI : 0;
  if (meanH < 0) meanH += 360;
  return { L: meanL, C: meanC, h: meanH };
}

export function recolorGarmentOKLCH(
  srcData: Uint8ClampedArray,
  width: number,
  height: number,
  mask: Uint8Array | Uint8ClampedArray,
  target: RGB | string | OKLCH,
): RecolorResult {
  const pixelCount = width * height;
  if (srcData.length !== pixelCount * 4) {
    throw new Error("srcData length does not match width * height * 4");
  }
  if (mask.length !== pixelCount) {
    throw new Error("mask length does not match width * height");
  }

  const isOKLCH = (v: any): v is OKLCH =>
    v && typeof v === "object" && typeof v.L === "number" && typeof v.C === "number" && typeof v.h === "number";

  const targetLCH: OKLCH =
    typeof target === "string"
      ? (() => {
          const hex = target.replace("#", "");
          const r = parseInt(hex.slice(0, 2), 16);
          const g = parseInt(hex.slice(2, 4), 16);
          const b = parseInt(hex.slice(4, 6), 16);
          return rgbToOKLCH({ r, g, b });
        })()
      : isOKLCH(target)
      ? target
      : rgbToOKLCH(target);

  // Keep algorithm stable by clamping to the same ranges used during per-pixel adjustment.
  const targetHue = targetLCH.h;
  const targetChroma = clamp(targetLCH.C, 0, 0.4);
  const targetLightness = clamp(targetLCH.L, 0, 1);
  const isVeryDarkTarget = targetLightness < 0.12; // Prevent "black becoming grey/white"
  const sourceMeanLCH = computeMaskedMeanLCH(srcData, mask, pixelCount);
  const isDarkGarment = sourceMeanLCH.L < DARK_GARMENT_MEAN_L_THRESHOLD;
  const isNearNeutralDarkGarment =
    isDarkGarment && sourceMeanLCH.C < NEAR_NEUTRAL_DARK_MEAN_C_THRESHOLD;

  const out = new Uint8ClampedArray(srcData.length);

  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let garmentCount = 0;

  for (let i = 0; i < pixelCount; i++) {
    const srcIndex = i * 4;
    const r = srcData[srcIndex];
    const g = srcData[srcIndex + 1];
    const b = srcData[srcIndex + 2];
    const a = srcData[srcIndex + 3];

    if (mask[i] === 0) {
      out[srcIndex] = r;
      out[srcIndex + 1] = g;
      out[srcIndex + 2] = b;
      out[srcIndex + 3] = a;
      continue;
    }

    const lch = rgbToOKLCH({ r, g, b });
    if (isDarkGarment) {
      // Dark-garment-only path: preserve local texture via relative transforms.
      const deltaL = lch.L - sourceMeanLCH.L;
      lch.L = clamp(targetLightness + deltaL, 0, 1);

      if (isNearNeutralDarkGarment) {
        // Black/near-black garments have unreliable per-pixel chroma+hue.
        // Keep hue uniform and modulate chroma only slightly by lightness
        // contrast to preserve folds without introducing color noise.
        const lContrastFactor = clamp(
          1 + (lch.L - targetLightness) * 0.8,
          0.75,
          1.25,
        );
        lch.C = clamp(targetChroma * lContrastFactor, 0, 0.4);
      } else {
        // For dark but chromatic garments, keep a bounded chroma ratio so
        // textures are preserved without ratio explosions (color speckles).
        const safeMeanC = sourceMeanLCH.C > 1e-6 ? sourceMeanLCH.C : 1e-6;
        const chromaRatio = clamp(lch.C / safeMeanC, 0.55, 1.45);
        lch.C = clamp(Math.max(0, targetChroma * chromaRatio), 0, 0.4);
      }

      // Force stable target hue across dark garments to avoid random green/yellow patches.
      // Hue from near-achromatic pixels is unstable and causes rainbow artifacts.
      lch.h = targetHue;
    } else {
      // Keep existing light/medium behavior unchanged.
      lch.h = targetHue;

      if (lch.L < 0.35) {
        // If user selects a very dark target, don't lift shadows up.
        if (isVeryDarkTarget) {
          lch.L = lch.L * 0.98;
          lch.C = lch.C * 0.98;
        } else {
          lch.L = lch.L * 1.2;
          lch.C = lch.C * 1.3;
        }
      }

      const dominance = computeDominanceScores(lch);

      // Ensure slider responsiveness:
      // - Always nudge L towards targetLightness (small, but non-zero)
      // - Always nudge C towards targetChroma (small, but non-zero)
      // - Use dominance to slightly increase strength when that dimension "matters most"
      const lBlend = dominance.primary === "L" ? 0.18 : 0.06;
      const cBlend = dominance.primary === "C" ? 0.18 : 0.06;

      lch.L = lch.L * (1 - lBlend) + targetLightness * lBlend;
      lch.C = lch.C * (1 - cBlend) + targetChroma * cBlend;

      // Keep the original texture-friendly behavior: hue is replaced, but L/C are constrained.
      lch.L = clamp(lch.L, 0, 1);
      lch.C = clamp(lch.C, 0, 0.4);

      // Additional stabilizer: don't overshoot chroma beyond the target.
      lch.C = Math.min(lch.C * 1.05, targetChroma);
    }

    const newRGB = oklchToRGB(lch);
    const nr = clamp(Math.round(newRGB.r), 0, 255);
    const ng = clamp(Math.round(newRGB.g), 0, 255);
    const nb = clamp(Math.round(newRGB.b), 0, 255);

    out[srcIndex] = nr;
    out[srcIndex + 1] = ng;
    out[srcIndex + 2] = nb;
    out[srcIndex + 3] = a;

    sumR += nr;
    sumG += ng;
    sumB += nb;
    garmentCount++;
  }

  const previewHex =
    garmentCount > 0
      ? rgbToHex({
          r: Math.round(sumR / garmentCount),
          g: Math.round(sumG / garmentCount),
          b: Math.round(sumB / garmentCount),
        })
      : undefined;

  return { data: out, width, height, previewHex };
}

