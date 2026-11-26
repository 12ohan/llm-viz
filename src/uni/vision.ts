/**
 * UNI2 Slice 2A: Procedural RGB Scenes, Integer-Centered Letterbox,
 * Miniature Continuous RADIO-H Vision Tower (2L ViT), RadioHProjector (`D3`),
 * and Prompt Visual-Token Expansion/Splicing (`src/uni/vision.ts`)
 *
 * Clean-room first-principles TypeScript implementation derived from
 * `UNI0_ARCHITECTURE_AUDIT.md` (§2.1, §4, §8, `D2`, `D3`, `U8`, `U18`, `U30`):
 *   - 3 deterministic procedural CHW RGB scenes (`16x16` and `16x8` panorama)
 *   - Integer-centered black-padded (`0.0`) letterbox resize (`U18`, `U30`)
 *   - Isolated per-tensor FNV-1a(seed, tensorName) + mulberry32 + Box-Muller
 *     weights for exact `31` tensors (`8,296` floats):
 *       * `27` Mini RADIO-H ViT tensors (`7,648` floats)
 *       * `3` Mini RadioHProjector tensors (`640` floats)
 *       * `1` dead `encoder.image_newline` tensor (`8` floats, `unusedInForward: true`, `U8`)
 *   - Exact FP32 forward math matching PyTorch `2.15.0` (`<= 2e-5`):
 *       * CLIP normalization
 *       * `4x4` CHW patch extraction + bias-free patch embedder
 *       * Bilinear CPE `pos_embed` interpolation (`align_corners=False`)
 *       * `4` prefix tokens (`2` CLS + `2` reg) + `2` timm ViT blocks
 *         (`LayerNorm(1e-6)` -> `qkv` -> 4-head bidirectional SDPA -> `proj`
 *         -> `LayerNorm(1e-6)` -> `fc1` -> exact `GELU` -> `fc2`) -> `[:, 4:]` strip
 *       * `2x2` `ps_version='v2'` row-major spatial gather -> FP32 `RMSNorm(64, 1e-5)`
 *         -> bias-free `fc1` -> `SquaredReLU` -> bias-free `fc2` -> `D3` gain (`0.25` / `0.01`)
 *   - `fixOnePrompt` (`<SPECIAL_23>` + literal `\n` `24`, never reading `image_newline`)
 *     and non-mutating `spliceVisualEmbeddingsIntoPrompt`
 *
 * Pure headless TypeScript — zero DOM, WebGL, ambient RNG, or sibling-app imports.
 */

import { applyUniRmsNorm, linearProject, softmaxRows } from './attention';
import {
  UNI_CLIP_MEAN,
  UNI_CLIP_STD,
  UNI_GAIN_PRESETS,
  UNI_IMAGE_NEWLINE_TENSOR_NAME,
  UNI_MINI_CONFIG,
  UNI_MINI_IMAGE_NEWLINE_FLOAT_COUNT,
  UNI_MINI_IMAGE_NEWLINE_TENSOR_COUNT,
  UNI_MINI_NEWLINE_TOKEN_ID,
  UNI_MINI_PROJECTOR_FLOAT_COUNT,
  UNI_MINI_PROJECTOR_IN_DIM,
  UNI_MINI_PROJECTOR_NORM_EPS,
  UNI_MINI_PROJECTOR_TENSOR_COUNT,
  UNI_MINI_UND_PAD_TOKEN_ID,
  UNI_MINI_VISION_SUBSYSTEM_FLOAT_COUNT,
  UNI_MINI_VISION_SUBSYSTEM_TENSOR_COUNT,
  UNI_MINI_VISION_TOWER_FLOAT_COUNT,
  UNI_MINI_VISION_TOWER_TENSOR_COUNT,
  UNI_MINI_VIT_BASE_GRID,
  UNI_MINI_VIT_EMBED_DIM,
  UNI_MINI_VIT_HEAD_DIM,
  UNI_MINI_VIT_LAYER_NORM_EPS,
  UNI_MINI_VIT_MLP_DIM,
  UNI_MINI_VIT_NUM_HEADS,
  UNI_MINI_VIT_NUM_LAYERS,
  UNI_MINI_VIT_PATCH_DIM,
  UNI_MINI_VIT_PATCH_SIZE,
  UNI_MINI_VIT_PREFIX_TOKENS,
  UNI_MULTIMODAL_SEED,
  type UniFixedPromptResult,
  type UniGainPresetId,
  type UniLetterboxInfo,
  type UniLetterboxResult,
  type UniMultimodalTensorEntry,
  type UniRadioProjectorTrace,
  type UniRadioProjectorWeights,
  type UniRadioVisionTowerWeights,
  type UniRadioVisionTrace,
  type UniRadioVitBlockTrace,
  type UniRadioVitBlockWeights,
  type UniRgbImage,
  type UniSceneGroundTruthBox,
  type UniScenePresetId,
  type UniVisionPipelineTrace,
  type UniVisionSubsystemWeights,
} from './types';

// ============================================================================
// CONSTANTS & DETERMINISTIC PER-TENSOR PRNG (MATCHING `src/uni/parameters.ts`)
// ============================================================================

/** Exact black (`0.0`) letterbox fill value used by `padToSquareAndResize` (`U18`, `U30`). */
export const UNI_LETTERBOX_PAD_VALUE = 0.0 as const;

export const UNI_SCENE_PRESET_IDS: readonly UniScenePresetId[] = Object.freeze([
  'sunlit_cat_16x16',
  'harbor_panorama_16x8',
  'geometric_blocks_16x16',
] as const);

const FNV1A_32_OFFSET = 0x811c9dc5;
const FNV1A_32_PRIME = 0x01000193;
const SQRT_PI = Math.sqrt(Math.PI);
const TWO_OVER_SQRT_PI = 2.0 / SQRT_PI;

function fnv1a32Seed(baseSeed: number, label: string): number {
  let h = (FNV1A_32_OFFSET ^ (baseSeed >>> 0)) >>> 0;
  h = Math.imul(h, FNV1A_32_PRIME) >>> 0;
  for (let i = 0; i < label.length; i++) {
    h ^= label.charCodeAt(i) & 0xff;
    h = Math.imul(h, FNV1A_32_PRIME) >>> 0;
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createGaussianGenerator(seed: number): () => number {
  const nextUniform = mulberry32(seed);
  let spare: number | null = null;

  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u1 = nextUniform();
    while (u1 <= 1e-12) {
      u1 = nextUniform();
    }
    const u2 = nextUniform();
    const mag = Math.sqrt(-2.0 * Math.log(u1));
    const angle = 2.0 * Math.PI * u2;
    spare = mag * Math.sin(angle);
    return mag * Math.cos(angle);
  };
}

function sampleGaussianTensor(
  baseSeed: number,
  tensorName: string,
  length: number,
  std: number,
): Float32Array {
  const gauss = createGaussianGenerator(fnv1a32Seed(baseSeed, tensorName));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = Math.fround(gauss() * std);
  }
  return out;
}

function sampleNormWeightTensor(
  baseSeed: number,
  tensorName: string,
  length: number,
): Float32Array {
  const gauss = createGaussianGenerator(fnv1a32Seed(baseSeed, tensorName));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const v = 1.0 + 0.05 * gauss();
    out[i] = Math.fround(v < 0.5 ? 0.5 : v);
  }
  return out;
}

// ============================================================================
// 1. PROCEDURAL RGB SCENES & INTEGER-CENTERED LETTERBOX (`U18`, `U30`)
// ============================================================================

function clamp01(v: number): number {
  if (v <= 0) return 0;
  if (v >= 1) return 1;
  return v;
}

function setRgbPixel(
  data: Float32Array,
  width: number,
  height: number,
  x: number,
  y: number,
  r: number,
  g: number,
  b: number,
): void {
  if (x < 0 || x >= width || y < 0 || y >= height) return;
  const hw = width * height;
  const idx = y * width + x;
  data[idx] = Math.fround(clamp01(r));
  data[hw + idx] = Math.fround(clamp01(g));
  data[2 * hw + idx] = Math.fround(clamp01(b));
}

function buildSunlitCatScene(): UniRgbImage {
  const width = 16;
  const height = 16;
  const data = new Float32Array(3 * width * height);

  // Sky (y = 0..9) and sunlit grass/ground (y = 10..15)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (y < 10) {
        const t = y / 9;
        setRgbPixel(
          data,
          width,
          height,
          x,
          y,
          0.32 + 0.28 * t,
          0.62 + 0.22 * t,
          0.95 - 0.08 * t,
        );
      } else {
        const t = (y - 10) / 5;
        const grassShade = ((x + y) & 1) === 0 ? 0.04 : -0.02;
        setRgbPixel(
          data,
          width,
          height,
          x,
          y,
          0.22 + 0.06 * t + grassShade,
          0.56 - 0.12 * t + grassShade,
          0.20 - 0.04 * t,
        );
      }
    }
  }

  // Warm golden sun disk in upper-right (box [11, 1, 15, 5])
  const sunCx = 12.5;
  const sunCy = 2.5;
  for (let y = 0; y <= 5; y++) {
    for (let x = 10; x <= 15; x++) {
      const dx = x - sunCx;
      const dy = y - sunCy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist <= 1.75) {
        setRgbPixel(data, width, height, x, y, 0.99, 0.90, 0.26);
      } else if (dist <= 2.45) {
        setRgbPixel(data, width, height, x, y, 0.94, 0.80, 0.44);
      }
    }
  }

  // Ginger cat silhouette (box [3, 6, 10, 15])
  // Ears at (4,6) and (7,6)
  setRgbPixel(data, width, height, 4, 6, 0.86, 0.46, 0.16);
  setRgbPixel(data, width, height, 7, 6, 0.86, 0.46, 0.16);

  // Head (x = 4..7, y = 7..9)
  for (let y = 7; y <= 9; y++) {
    for (let x = 4; x <= 7; x++) {
      setRgbPixel(data, width, height, x, y, 0.90, 0.52, 0.20);
    }
  }
  // Eyes & nose
  setRgbPixel(data, width, height, 4, 8, 0.12, 0.34, 0.18);
  setRgbPixel(data, width, height, 7, 8, 0.12, 0.34, 0.18);
  setRgbPixel(data, width, height, 5, 9, 0.94, 0.66, 0.62);
  setRgbPixel(data, width, height, 6, 9, 0.94, 0.66, 0.62);

  // Body (x = 3..8, y = 10..14) + curled tail (x = 9, y = 11..13)
  for (let y = 10; y <= 14; y++) {
    for (let x = 3; x <= 8; x++) {
      const stripe = (y & 1) === 0 ? 0.0 : -0.10;
      const chest = x >= 5 && x <= 6 && y <= 12 ? 0.12 : 0.0;
      setRgbPixel(
        data,
        width,
        height,
        x,
        y,
        0.84 + stripe + chest,
        0.46 + stripe * 0.7 + chest,
        0.18 + chest * 0.8,
      );
    }
  }
  for (let y = 11; y <= 13; y++) {
    setRgbPixel(data, width, height, 9, y, 0.78, 0.40, 0.14);
  }
  // Paws
  setRgbPixel(data, width, height, 4, 14, 0.95, 0.88, 0.78);
  setRgbPixel(data, width, height, 7, 14, 0.95, 0.88, 0.78);

  const groundTruthBoxes: readonly UniSceneGroundTruthBox[] = Object.freeze([
    Object.freeze({ tag: 'cat', pixelBox: Object.freeze([3, 6, 10, 15] as const) }),
    Object.freeze({ tag: 'sun', pixelBox: Object.freeze([11, 1, 15, 5] as const) }),
  ]);

  return Object.freeze({
    presetId: 'sunlit_cat_16x16' as const,
    label: 'Sunlit Cat (16x16)',
    width,
    height,
    data,
    groundTruthBoxes,
  });
}

function buildHarborPanoramaScene(): UniRgbImage {
  const width = 16;
  const height = 8;
  const data = new Float32Array(3 * width * height);

  // Sky (y = 0..3) and harbor water (y = 4..7)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (y < 4) {
        const t = y / 3;
        setRgbPixel(
          data,
          width,
          height,
          x,
          y,
          0.26 + 0.44 * t,
          0.46 + 0.28 * t,
          0.84 - 0.06 * t,
        );
      } else {
        const t = (y - 4) / 3;
        const ripple = ((x + y) % 3) === 0 ? 0.08 : 0.0;
        setRgbPixel(
          data,
          width,
          height,
          x,
          y,
          0.08 + 0.04 * t + ripple * 0.5,
          0.28 + 0.12 * (1 - t) + ripple,
          0.62 - 0.10 * t + ripple,
        );
      }
    }
  }

  // Harbor boat with sail on the left (x = 2..6, y = 2..5)
  setRgbPixel(data, width, height, 4, 2, 0.96, 0.96, 0.92);
  setRgbPixel(data, width, height, 4, 3, 0.96, 0.96, 0.92);
  setRgbPixel(data, width, height, 5, 3, 0.92, 0.92, 0.86);
  for (let x = 2; x <= 6; x++) {
    setRgbPixel(data, width, height, x, 4, 0.58, 0.30, 0.14);
  }
  for (let x = 3; x <= 5; x++) {
    setRgbPixel(data, width, height, x, 5, 0.44, 0.22, 0.10);
  }

  // Pier & red-roofed harbor lighthouse/house on the right (box [11, 1, 15, 6])
  for (let x = 11; x <= 14; x++) {
    setRgbPixel(data, width, height, x, 5, 0.38, 0.34, 0.30); // stone pier
  }
  for (let x = 11; x <= 14; x++) {
    setRgbPixel(data, width, height, x, 2, 0.84, 0.22, 0.18); // red roof
  }
  setRgbPixel(data, width, height, 12, 1, 0.96, 0.88, 0.32); // lighthouse lantern
  setRgbPixel(data, width, height, 13, 1, 0.96, 0.88, 0.32);
  for (let y = 3; y <= 4; y++) {
    for (let x = 11; x <= 14; x++) {
      setRgbPixel(data, width, height, x, y, 0.92, 0.90, 0.84); // white walls
    }
  }
  // Warm window
  setRgbPixel(data, width, height, 12, 3, 0.98, 0.84, 0.28);

  const groundTruthBoxes: readonly UniSceneGroundTruthBox[] = Object.freeze([
    Object.freeze({ tag: 'house', pixelBox: Object.freeze([11, 1, 15, 6] as const) }),
    Object.freeze({ tag: 'water', pixelBox: Object.freeze([0, 4, 16, 8] as const) }),
  ]);

  return Object.freeze({
    presetId: 'harbor_panorama_16x8' as const,
    label: 'Harbor Panorama (16x8)',
    width,
    height,
    data,
    groundTruthBoxes,
  });
}

function buildGeometricBlocksScene(): UniRgbImage {
  const width = 16;
  const height = 16;
  const data = new Float32Array(3 * width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const right = x >= 8;
      const bottom = y >= 8;
      const lx = x % 8;
      const ly = y % 8;
      const isBorder = lx === 0 || lx === 7 || ly === 0 || ly === 7;
      const grad = (lx + ly) / 14;

      if (!right && !bottom) {
        // Top-left: red block
        const r = isBorder ? 0.55 : 0.88 - 0.12 * grad;
        const g = isBorder ? 0.14 : 0.20 + 0.08 * grad;
        const b = isBorder ? 0.14 : 0.18 + 0.06 * grad;
        setRgbPixel(data, width, height, x, y, r, g, b);
      } else if (right && !bottom) {
        // Top-right: blue block
        const r = isBorder ? 0.14 : 0.18 + 0.08 * grad;
        const g = isBorder ? 0.24 : 0.42 + 0.14 * grad;
        const b = isBorder ? 0.56 : 0.90 - 0.10 * grad;
        setRgbPixel(data, width, height, x, y, r, g, b);
      } else if (!right && bottom) {
        // Bottom-left: green block
        const r = isBorder ? 0.12 : 0.18 + 0.08 * grad;
        const g = isBorder ? 0.50 : 0.82 - 0.10 * grad;
        const b = isBorder ? 0.22 : 0.30 + 0.08 * grad;
        setRgbPixel(data, width, height, x, y, r, g, b);
      } else {
        // Bottom-right: bright amber block
        const r = isBorder ? 0.62 : 0.96 - 0.06 * grad;
        const g = isBorder ? 0.48 : 0.80 - 0.08 * grad;
        const b = isBorder ? 0.14 : 0.22 + 0.10 * grad;
        setRgbPixel(data, width, height, x, y, r, g, b);
      }
    }
  }

  const groundTruthBoxes: readonly UniSceneGroundTruthBox[] = Object.freeze([
    Object.freeze({ tag: 'red', pixelBox: Object.freeze([1, 1, 7, 7] as const) }),
    Object.freeze({ tag: 'blue', pixelBox: Object.freeze([9, 1, 15, 7] as const) }),
    Object.freeze({ tag: 'green', pixelBox: Object.freeze([1, 9, 7, 15] as const) }),
    Object.freeze({ tag: 'bright', pixelBox: Object.freeze([9, 9, 15, 15] as const) }),
  ]);

  return Object.freeze({
    presetId: 'geometric_blocks_16x16' as const,
    label: 'Geometric Blocks (16x16)',
    width,
    height,
    data,
    groundTruthBoxes,
  });
}

/**
 * Create one of the three deterministic procedural CHW RGB scenes (`[3, H, W]` in `[0, 1]`).
 */
export function createUniScenePreset(presetId: UniScenePresetId): UniRgbImage {
  switch (presetId) {
    case 'sunlit_cat_16x16':
      return buildSunlitCatScene();
    case 'harbor_panorama_16x8':
      return buildHarborPanoramaScene();
    case 'geometric_blocks_16x16':
      return buildGeometricBlocksScene();
    default: {
      const exhaustive: never = presetId;
      throw new RangeError(`Unknown UniScenePresetId: ${String(exhaustive)}`);
    }
  }
}

function validateRgbImage(image: UniRgbImage): void {
  if (!Number.isInteger(image.width) || image.width < 1) {
    throw new RangeError(`image.width must be an integer >= 1, got ${image.width}`);
  }
  if (!Number.isInteger(image.height) || image.height < 1) {
    throw new RangeError(`image.height must be an integer >= 1, got ${image.height}`);
  }
  const expected = 3 * image.width * image.height;
  if (image.data.length !== expected) {
    throw new RangeError(
      `image.data length (${image.data.length}) must equal 3 * width * height (${expected})`,
    );
  }
}

/**
 * Square-pad and resize a CHW RGB image to `targetSize x targetSize` (default `16`)
 * using exact black (`0.0`) letterbox padding (`U18`, `U30`).
 *
 * - For `16x16 -> 16`: identity copy (`padLeft=padTop=padRight=padBottom=0`, `scale=1`, `exactIntegerCentered=true`).
 * - For `16x8 -> 16`: exact integer-centered (`scale=1`, `resizedWidth=16`, `resizedHeight=8`,
 *   `padLeft=0, padTop=4, padRight=0, padBottom=4`, `exactIntegerCentered=true`) — rows `0..3`
 *   and `12..15` are `0.0`, rows `4..11` are exact source pixels with zero rounding drift.
 * - For arbitrary `(W, H)` where `max(W, H) !== targetSize`: bilinear resize (`align_corners=False`)
 *   to `(resizedWidth, resizedHeight)` followed by centered `0.0` padding.
 */
export function padToSquareAndResize(
  image: UniRgbImage,
  targetSize = UNI_MINI_CONFIG.imageResolution,
): UniLetterboxResult {
  validateRgbImage(image);
  if (!Number.isInteger(targetSize) || targetSize < 1) {
    throw new RangeError(`targetSize must be an integer >= 1, got ${targetSize}`);
  }

  const srcW = image.width;
  const srcH = image.height;
  const maxDim = Math.max(srcW, srcH);
  const scale = targetSize / maxDim;

  const isNativeScale = maxDim === targetSize;
  const resizedWidth = isNativeScale
    ? srcW
    : Math.max(1, Math.min(targetSize, Math.round(srcW * scale)));
  const resizedHeight = isNativeScale
    ? srcH
    : Math.max(1, Math.min(targetSize, Math.round(srcH * scale)));

  const totalPadX = targetSize - resizedWidth;
  const totalPadY = targetSize - resizedHeight;
  const padLeft = Math.floor(totalPadX / 2);
  const padRight = totalPadX - padLeft;
  const padTop = Math.floor(totalPadY / 2);
  const padBottom = totalPadY - padTop;

  const exactIntegerCentered =
    totalPadX % 2 === 0 &&
    totalPadY % 2 === 0 &&
    (isNativeScale ||
      (Math.abs(srcW * scale - resizedWidth) < 1e-12 &&
        Math.abs(srcH * scale - resizedHeight) < 1e-12));

  const outData = new Float32Array(3 * targetSize * targetSize);
  if (UNI_LETTERBOX_PAD_VALUE !== 0.0) {
    outData.fill(UNI_LETTERBOX_PAD_VALUE);
  }

  const srcHW = srcW * srcH;
  const dstHW = targetSize * targetSize;

  if (isNativeScale) {
    // Exact pixel copy with zero interpolation drift (`U30`)
    for (let c = 0; c < 3; c++) {
      const srcChan = c * srcHW;
      const dstChan = c * dstHW;
      for (let y = 0; y < srcH; y++) {
        const srcRow = srcChan + y * srcW;
        const dstRow = dstChan + (padTop + y) * targetSize + padLeft;
        for (let x = 0; x < srcW; x++) {
          outData[dstRow + x] = image.data[srcRow + x];
        }
      }
    }
  } else {
    // Bilinear resize (`align_corners=False`) into centered letterbox region
    const scaleY = srcH / resizedHeight;
    const scaleX = srcW / resizedWidth;
    for (let y = 0; y < resizedHeight; y++) {
      const rawSy = (y + 0.5) * scaleY - 0.5;
      const sy = rawSy < 0 ? 0 : rawSy > srcH - 1 ? srcH - 1 : rawSy;
      const y0 = Math.floor(sy);
      const y1 = y0 + 1 < srcH ? y0 + 1 : srcH - 1;
      const wy = sy - y0;
      const wy0 = 1.0 - wy;

      for (let x = 0; x < resizedWidth; x++) {
        const rawSx = (x + 0.5) * scaleX - 0.5;
        const sx = rawSx < 0 ? 0 : rawSx > srcW - 1 ? srcW - 1 : rawSx;
        const x0 = Math.floor(sx);
        const x1 = x0 + 1 < srcW ? x0 + 1 : srcW - 1;
        const wx = sx - x0;
        const wx0 = 1.0 - wx;

        const w00 = wy0 * wx0;
        const w01 = wy0 * wx;
        const w10 = wy * wx0;
        const w11 = wy * wx;

        const dstIdx = (padTop + y) * targetSize + (padLeft + x);
        for (let c = 0; c < 3; c++) {
          const base = c * srcHW;
          const v00 = image.data[base + y0 * srcW + x0];
          const v01 = image.data[base + y0 * srcW + x1];
          const v10 = image.data[base + y1 * srcW + x0];
          const v11 = image.data[base + y1 * srcW + x1];
          outData[c * dstHW + dstIdx] = Math.fround(
            clamp01(w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11),
          );
        }
      }
    }
  }

  const transformedBoxes = image.groundTruthBoxes
    ? Object.freeze(
        image.groundTruthBoxes.map((box) => {
          const sx = resizedWidth / srcW;
          const sy = resizedHeight / srcH;
          const [x1, y1, x2, y2] = box.pixelBox;
          return Object.freeze({
            tag: box.tag,
            pixelBox: Object.freeze([
              x1 * sx + padLeft,
              y1 * sy + padTop,
              x2 * sx + padLeft,
              y2 * sy + padTop,
            ] as const),
          });
        }),
      )
    : undefined;

  const info: UniLetterboxInfo = Object.freeze({
    sourceWidth: srcW,
    sourceHeight: srcH,
    targetSize,
    scale,
    resizedWidth,
    resizedHeight,
    padLeft,
    padTop,
    padRight,
    padBottom,
    exactIntegerCentered,
  });

  const letterboxedImage: UniRgbImage = Object.freeze({
    presetId: image.presetId,
    label: image.label ? `${image.label} -> ${targetSize}x${targetSize}` : undefined,
    width: targetSize,
    height: targetSize,
    data: outData,
    groundTruthBoxes: transformedBoxes,
  });

  return Object.freeze({
    image: letterboxedImage,
    info,
  });
}

// ============================================================================
// 2. DETERMINISTIC WEIGHTS (31 TENSORS = 8,296 FLOATS)
// ============================================================================

const PATCH_EMBED_TENSOR_NAME = 'encoder.vision_tower.patch_generator.embedder.weight';
const POS_EMBED_TENSOR_NAME = 'encoder.vision_tower.pos_embed';
const CLS_TOKEN_TENSOR_NAME = 'encoder.vision_tower.cls_token';

const PROJECTOR_NORM_TENSOR_NAME = 'encoder.mm_projector.norm.weight';
const PROJECTOR_FC1_TENSOR_NAME = 'encoder.mm_projector.fc1.weight';
const PROJECTOR_FC2_TENSOR_NAME = 'encoder.mm_projector.fc2.weight';

function vitBlockTensorName(blockIndex: number, suffix: string): string {
  return `encoder.vision_tower.blocks.${blockIndex}.${suffix}`;
}

/**
 * Create deterministic synthetic weights for the miniature continuous vision subsystem:
 *   - `27` Mini RADIO-H ViT tensors (`7,648` floats)
 *   - `3` Mini RadioHProjector tensors (`640` floats)
 *   - `1` dead `encoder.image_newline` tensor (`8` floats, `unusedInForward: true`, `U8`)
 * Total: `31` tensors = `8,296` floats (`33,184 B`).
 */
export function createUniVisionSubsystemWeights(
  seed: number = UNI_MULTIMODAL_SEED,
): UniVisionSubsystemWeights {
  const s = seed >>> 0;
  const embedDim = UNI_MINI_VIT_EMBED_DIM; // 16
  const patchDim = UNI_MINI_VIT_PATCH_DIM; // 48
  const basePatches = UNI_MINI_VIT_BASE_GRID * UNI_MINI_VIT_BASE_GRID; // 16
  const prefixTokens = UNI_MINI_VIT_PREFIX_TOKENS; // 4
  const mlpDim = UNI_MINI_VIT_MLP_DIM; // 64
  const projInDim = UNI_MINI_PROJECTOR_IN_DIM; // 64
  const hiddenSize = UNI_MINI_CONFIG.hiddenSize; // 8

  const patchEmbedWeight = sampleGaussianTensor(
    s,
    PATCH_EMBED_TENSOR_NAME,
    embedDim * patchDim,
    0.25 / Math.sqrt(patchDim),
  );
  const posEmbed = sampleGaussianTensor(
    s,
    POS_EMBED_TENSOR_NAME,
    basePatches * embedDim,
    0.05,
  );
  const clsToken = sampleGaussianTensor(
    s,
    CLS_TOKEN_TENSOR_NAME,
    prefixTokens * embedDim,
    0.05,
  );

  const qkvStd = 0.25 / Math.sqrt(embedDim);
  const projStd = 0.25 / Math.sqrt(embedDim * 4);
  const fc1Std = 0.25 / Math.sqrt(embedDim);
  const fc2Std = 0.25 / Math.sqrt(mlpDim * 4);
  const biasStd = 0.02;

  const blocks: UniRadioVitBlockWeights[] = [];
  for (let b = 0; b < UNI_MINI_VIT_NUM_LAYERS; b++) {
    blocks.push(
      Object.freeze({
        blockIndex: b,
        norm1Weight: sampleNormWeightTensor(s, vitBlockTensorName(b, 'norm1.weight'), embedDim),
        norm1Bias: sampleGaussianTensor(s, vitBlockTensorName(b, 'norm1.bias'), embedDim, biasStd),
        qkvWeight: sampleGaussianTensor(
          s,
          vitBlockTensorName(b, 'attn.qkv.weight'),
          3 * embedDim * embedDim,
          qkvStd,
        ),
        qkvBias: sampleGaussianTensor(
          s,
          vitBlockTensorName(b, 'attn.qkv.bias'),
          3 * embedDim,
          biasStd,
        ),
        projWeight: sampleGaussianTensor(
          s,
          vitBlockTensorName(b, 'attn.proj.weight'),
          embedDim * embedDim,
          projStd,
        ),
        projBias: sampleGaussianTensor(
          s,
          vitBlockTensorName(b, 'attn.proj.bias'),
          embedDim,
          biasStd,
        ),
        norm2Weight: sampleNormWeightTensor(s, vitBlockTensorName(b, 'norm2.weight'), embedDim),
        norm2Bias: sampleGaussianTensor(s, vitBlockTensorName(b, 'norm2.bias'), embedDim, biasStd),
        fc1Weight: sampleGaussianTensor(
          s,
          vitBlockTensorName(b, 'mlp.fc1.weight'),
          mlpDim * embedDim,
          fc1Std,
        ),
        fc1Bias: sampleGaussianTensor(s, vitBlockTensorName(b, 'mlp.fc1.bias'), mlpDim, biasStd),
        fc2Weight: sampleGaussianTensor(
          s,
          vitBlockTensorName(b, 'mlp.fc2.weight'),
          embedDim * mlpDim,
          fc2Std,
        ),
        fc2Bias: sampleGaussianTensor(s, vitBlockTensorName(b, 'mlp.fc2.bias'), embedDim, biasStd),
      }),
    );
  }

  const visionTower: UniRadioVisionTowerWeights = Object.freeze({
    patchEmbedWeight,
    posEmbed,
    clsToken,
    blocks: Object.freeze(blocks),
  });

  const projector: UniRadioProjectorWeights = Object.freeze({
    normWeight: sampleNormWeightTensor(s, PROJECTOR_NORM_TENSOR_NAME, projInDim),
    fc1Weight: sampleGaussianTensor(
      s,
      PROJECTOR_FC1_TENSOR_NAME,
      hiddenSize * projInDim,
      0.25 / Math.sqrt(projInDim),
    ),
    fc2Weight: sampleGaussianTensor(
      s,
      PROJECTOR_FC2_TENSOR_NAME,
      hiddenSize * hiddenSize,
      0.25 / Math.sqrt(hiddenSize),
    ),
  });

  const imageNewline = sampleGaussianTensor(
    s,
    UNI_IMAGE_NEWLINE_TENSOR_NAME,
    hiddenSize,
    0.05,
  );

  const subsystem: UniVisionSubsystemWeights = Object.freeze({
    seed: s,
    visionTower,
    projector,
    imageNewline,
  });

  const collected = collectUniVisionWeights(subsystem);
  if (collected.length !== UNI_MINI_VISION_SUBSYSTEM_TENSOR_COUNT) {
    throw new Error(
      `Expected ${UNI_MINI_VISION_SUBSYSTEM_TENSOR_COUNT} vision subsystem tensors, got ${collected.length}`,
    );
  }
  let totalFloats = 0;
  for (let i = 0; i < collected.length; i++) {
    totalFloats += collected[i].data.length;
  }
  if (totalFloats !== UNI_MINI_VISION_SUBSYSTEM_FLOAT_COUNT) {
    throw new Error(
      `Expected ${UNI_MINI_VISION_SUBSYSTEM_FLOAT_COUNT} vision subsystem floats, got ${totalFloats}`,
    );
  }

  return subsystem;
}

/**
 * Collect all `31` canonical continuous-vision tensors (`8,296` floats) in exact pinned order:
 *   - `27` `vision_tower` tensors (`7,648` floats)
 *   - `3` `mm_projector` tensors (`640` floats)
 *   - `1` `image_newline` tensor (`8` floats, `unusedInForward: true`)
 */
export function collectUniVisionWeights(
  weights: UniVisionSubsystemWeights,
): UniMultimodalTensorEntry[] {
  const embedDim = UNI_MINI_VIT_EMBED_DIM;
  const patchDim = UNI_MINI_VIT_PATCH_DIM;
  const basePatches = UNI_MINI_VIT_BASE_GRID * UNI_MINI_VIT_BASE_GRID;
  const prefixTokens = UNI_MINI_VIT_PREFIX_TOKENS;
  const mlpDim = UNI_MINI_VIT_MLP_DIM;
  const projInDim = UNI_MINI_PROJECTOR_IN_DIM;
  const hiddenSize = UNI_MINI_CONFIG.hiddenSize;

  const entries: UniMultimodalTensorEntry[] = [
    {
      name: PATCH_EMBED_TENSOR_NAME,
      subsystem: 'vision_tower',
      shape: [embedDim, patchDim],
      data: weights.visionTower.patchEmbedWeight,
    },
    {
      name: POS_EMBED_TENSOR_NAME,
      subsystem: 'vision_tower',
      shape: [basePatches, embedDim],
      data: weights.visionTower.posEmbed,
    },
    {
      name: CLS_TOKEN_TENSOR_NAME,
      subsystem: 'vision_tower',
      shape: [prefixTokens, embedDim],
      data: weights.visionTower.clsToken,
    },
  ];

  for (let b = 0; b < weights.visionTower.blocks.length; b++) {
    const blk = weights.visionTower.blocks[b];
    const idx = blk.blockIndex;
    entries.push(
      {
        name: vitBlockTensorName(idx, 'norm1.weight'),
        subsystem: 'vision_tower',
        shape: [embedDim],
        data: blk.norm1Weight,
      },
      {
        name: vitBlockTensorName(idx, 'norm1.bias'),
        subsystem: 'vision_tower',
        shape: [embedDim],
        data: blk.norm1Bias,
      },
      {
        name: vitBlockTensorName(idx, 'attn.qkv.weight'),
        subsystem: 'vision_tower',
        shape: [3 * embedDim, embedDim],
        data: blk.qkvWeight,
      },
      {
        name: vitBlockTensorName(idx, 'attn.qkv.bias'),
        subsystem: 'vision_tower',
        shape: [3 * embedDim],
        data: blk.qkvBias,
      },
      {
        name: vitBlockTensorName(idx, 'attn.proj.weight'),
        subsystem: 'vision_tower',
        shape: [embedDim, embedDim],
        data: blk.projWeight,
      },
      {
        name: vitBlockTensorName(idx, 'attn.proj.bias'),
        subsystem: 'vision_tower',
        shape: [embedDim],
        data: blk.projBias,
      },
      {
        name: vitBlockTensorName(idx, 'norm2.weight'),
        subsystem: 'vision_tower',
        shape: [embedDim],
        data: blk.norm2Weight,
      },
      {
        name: vitBlockTensorName(idx, 'norm2.bias'),
        subsystem: 'vision_tower',
        shape: [embedDim],
        data: blk.norm2Bias,
      },
      {
        name: vitBlockTensorName(idx, 'mlp.fc1.weight'),
        subsystem: 'vision_tower',
        shape: [mlpDim, embedDim],
        data: blk.fc1Weight,
      },
      {
        name: vitBlockTensorName(idx, 'mlp.fc1.bias'),
        subsystem: 'vision_tower',
        shape: [mlpDim],
        data: blk.fc1Bias,
      },
      {
        name: vitBlockTensorName(idx, 'mlp.fc2.weight'),
        subsystem: 'vision_tower',
        shape: [embedDim, mlpDim],
        data: blk.fc2Weight,
      },
      {
        name: vitBlockTensorName(idx, 'mlp.fc2.bias'),
        subsystem: 'vision_tower',
        shape: [embedDim],
        data: blk.fc2Bias,
      },
    );
  }

  entries.push(
    {
      name: PROJECTOR_NORM_TENSOR_NAME,
      subsystem: 'mm_projector',
      shape: [projInDim],
      data: weights.projector.normWeight,
    },
    {
      name: PROJECTOR_FC1_TENSOR_NAME,
      subsystem: 'mm_projector',
      shape: [hiddenSize, projInDim],
      data: weights.projector.fc1Weight,
    },
    {
      name: PROJECTOR_FC2_TENSOR_NAME,
      subsystem: 'mm_projector',
      shape: [hiddenSize, hiddenSize],
      data: weights.projector.fc2Weight,
    },
    {
      name: UNI_IMAGE_NEWLINE_TENSOR_NAME,
      subsystem: 'image_newline',
      shape: [hiddenSize],
      data: weights.imageNewline,
      unusedInForward: true,
    },
  );

  // Sanity guard reconciling pinned subsystem totals
  if (
    entries.length !==
    UNI_MINI_VISION_TOWER_TENSOR_COUNT +
      UNI_MINI_PROJECTOR_TENSOR_COUNT +
      UNI_MINI_IMAGE_NEWLINE_TENSOR_COUNT
  ) {
    throw new Error(`Unexpected vision tensor count: ${entries.length}`);
  }
  const expectedFloats =
    UNI_MINI_VISION_TOWER_FLOAT_COUNT +
    UNI_MINI_PROJECTOR_FLOAT_COUNT +
    UNI_MINI_IMAGE_NEWLINE_FLOAT_COUNT;
  let sumFloats = 0;
  for (let i = 0; i < entries.length; i++) {
    sumFloats += entries[i].data.length;
  }
  if (sumFloats !== expectedFloats) {
    throw new Error(`Unexpected vision float count: ${sumFloats} !== ${expectedFloats}`);
  }

  return entries;
}

// ============================================================================
// 3. EXACT FORWARD MATH HELPERS (CLIP, BILINEAR CPE, LAYERNORM, EXACT GELU)
// ============================================================================

/**
 * High-precision error function `erf(x)` accurate to `< 1e-15` in Float64:
 *   - For `|x| <= 3.5`: positive-term confluent series `2/sqrt(pi) * x * exp(-x^2) * sum((2x^2)^n / (2n+1)!!)`
 *   - For `3.5 < |x| < 6.0`: Laplace continued fraction for `erfc(|x|)`
 *   - For `|x| >= 6.0`: `+/- 1.0` (exact to IEEE-754 Float64 precision)
 */
function erfHighPrecision(x: number): number {
  if (x === 0) return 0;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);

  if (ax >= 6.0) {
    return sign;
  }

  if (ax <= 3.5) {
    const twoX2 = 2.0 * ax * ax;
    let term = 1.0;
    let sum = 1.0;
    for (let n = 1; n <= 60; n++) {
      term *= twoX2 / (2 * n + 1);
      sum += term;
      if (term <= sum * 1e-16) break;
    }
    return sign * TWO_OVER_SQRT_PI * ax * Math.exp(-ax * ax) * sum;
  }

  // 3.5 < ax < 6.0: Laplace continued fraction for erfc(ax)
  let cf = 0.0;
  for (let n = 36; n >= 1; n--) {
    cf = (0.5 * n) / (ax + cf);
  }
  const erfcVal = Math.exp(-ax * ax) / (SQRT_PI * (ax + cf));
  return sign * (1.0 - erfcVal);
}

/** Exact GELU matching `torch.nn.functional.gelu(x, approximate='none')` in Float32. */
function geluExactF32(x: number): number {
  return Math.fround(0.5 * x * (1.0 + erfHighPrecision(x * Math.SQRT1_2)));
}

function applyLayerNorm(
  x: Float32Array,
  seqLen: number,
  dim: number,
  weight: Float32Array,
  bias: Float32Array,
  eps: number,
): Float32Array {
  const out = new Float32Array(seqLen * dim);
  for (let t = 0; t < seqLen; t++) {
    const base = t * dim;
    let sum = 0;
    for (let d = 0; d < dim; d++) {
      sum += x[base + d];
    }
    const mean = sum / dim;
    let sumSq = 0;
    for (let d = 0; d < dim; d++) {
      const diff = x[base + d] - mean;
      sumSq += diff * diff;
    }
    const variance = sumSq / dim;
    const invStd = 1.0 / Math.sqrt(variance + eps);
    for (let d = 0; d < dim; d++) {
      const normed = (x[base + d] - mean) * invStd;
      out[base + d] = Math.fround(normed * weight[d] + bias[d]);
    }
  }
  return out;
}

function linearWithBias(
  input: Float32Array,
  seqLen: number,
  inDim: number,
  weight: Float32Array,
  bias: Float32Array,
  outDim: number,
): Float32Array {
  const out = new Float32Array(seqLen * outDim);
  for (let t = 0; t < seqLen; t++) {
    const xBase = t * inDim;
    const yBase = t * outDim;
    for (let o = 0; o < outDim; o++) {
      const wBase = o * inDim;
      let acc = 0;
      for (let i = 0; i < inDim; i++) {
        acc += input[xBase + i] * weight[wBase + i];
      }
      out[yBase + o] = Math.fround(acc + bias[o]);
    }
  }
  return out;
}

/**
 * CLIP channel-wise normalization `(x - UNI_CLIP_MEAN[c]) / UNI_CLIP_STD[c]` in Float32.
 */
export function clipNormalizeImage(image: UniRgbImage): Float32Array {
  validateRgbImage(image);
  const hw = image.width * image.height;
  const out = new Float32Array(3 * hw);
  for (let c = 0; c < 3; c++) {
    const mean = UNI_CLIP_MEAN[c];
    const std = UNI_CLIP_STD[c];
    const base = c * hw;
    for (let p = 0; p < hw; p++) {
      out[base + p] = Math.fround((image.data[base + p] - mean) / std);
    }
  }
  return out;
}

/**
 * 2D bilinear interpolation of patch positional embeddings matching PyTorch
 * `F.interpolate(..., mode='bilinear', align_corners=False)`.
 *
 * Returns an exact copy when `targetRows === baseRows && targetCols === baseCols`.
 */
export function interpolatePosEmbed2d(
  posEmbed: Float32Array,
  baseRows: number,
  baseCols: number,
  targetRows: number,
  targetCols: number,
  embedDim: number,
): Float32Array {
  if (
    !Number.isInteger(baseRows) ||
    baseRows < 1 ||
    !Number.isInteger(baseCols) ||
    baseCols < 1 ||
    !Number.isInteger(targetRows) ||
    targetRows < 1 ||
    !Number.isInteger(targetCols) ||
    targetCols < 1 ||
    !Number.isInteger(embedDim) ||
    embedDim < 1
  ) {
    throw new RangeError(
      `Grid dimensions (${baseRows}x${baseCols} -> ${targetRows}x${targetCols}) and embedDim (${embedDim}) must be integers >= 1`,
    );
  }
  const baseLen = baseRows * baseCols * embedDim;
  if (posEmbed.length < baseLen) {
    throw new RangeError(
      `posEmbed length (${posEmbed.length}) is smaller than baseRows * baseCols * embedDim (${baseLen})`,
    );
  }

  if (targetRows === baseRows && targetCols === baseCols) {
    return new Float32Array(posEmbed.subarray(0, baseLen));
  }

  const out = new Float32Array(targetRows * targetCols * embedDim);
  const scaleR = baseRows / targetRows;
  const scaleC = baseCols / targetCols;

  for (let tr = 0; tr < targetRows; tr++) {
    const rawR = (tr + 0.5) * scaleR - 0.5;
    const srcR = rawR < 0 ? 0 : rawR > baseRows - 1 ? baseRows - 1 : rawR;
    const r0 = Math.floor(srcR);
    const r1 = r0 + 1 < baseRows ? r0 + 1 : baseRows - 1;
    const wy = srcR - r0;
    const wy0 = 1.0 - wy;

    for (let tc = 0; tc < targetCols; tc++) {
      const rawC = (tc + 0.5) * scaleC - 0.5;
      const srcC = rawC < 0 ? 0 : rawC > baseCols - 1 ? baseCols - 1 : rawC;
      const c0 = Math.floor(srcC);
      const c1 = c0 + 1 < baseCols ? c0 + 1 : baseCols - 1;
      const wx = srcC - c0;
      const wx0 = 1.0 - wx;

      const w00 = wy0 * wx0;
      const w01 = wy0 * wx;
      const w10 = wy * wx0;
      const w11 = wy * wx;

      const p00Base = (r0 * baseCols + c0) * embedDim;
      const p01Base = (r0 * baseCols + c1) * embedDim;
      const p10Base = (r1 * baseCols + c0) * embedDim;
      const p11Base = (r1 * baseCols + c1) * embedDim;
      const dstBase = (tr * targetCols + tc) * embedDim;

      for (let d = 0; d < embedDim; d++) {
        out[dstBase + d] = Math.fround(
          w00 * posEmbed[p00Base + d] +
            w01 * posEmbed[p01Base + d] +
            w10 * posEmbed[p10Base + d] +
            w11 * posEmbed[p11Base + d],
        );
      }
    }
  }

  return out;
}

// ============================================================================
// 4. MINI RADIO-H VISION TOWER & MINI RADIO-H PROJECTOR (`D3`, `U8`)
// ============================================================================

/**
 * Execute the miniature 2-layer RADIO-H ViT vision tower on a CHW RGB image:
 *   - Requires `height % 8 === 0` and `width % 8 === 0` (even `4x4` patch grid for `2x2` merge)
 *   - CLIP normalization -> row-major `4x4` CHW patches (`48` floats/patch) -> bias-free `Linear(48 -> 16)`
 *   - Add bilinearly interpolated `pos_embed` -> prepend `4` prefix tokens (`2` CLS + `2` reg)
 *   - Run `2` standard timm ViT blocks (`LayerNorm(1e-6)`, 4-head bidirectional SDPA `scale = 0.5`, exact `GELU` MLP)
 *   - Strip prefix via `[:, 4:]` to return spatial `patchFeatures [N_p, 16]`
 */
export function runMiniRadioVisionTower(
  image: UniRgbImage,
  weights: UniRadioVisionTowerWeights,
): UniRadioVisionTrace {
  validateRgbImage(image);
  const { width, height } = image;
  if (height < 8 || height % 8 !== 0 || width < 8 || width % 8 !== 0) {
    throw new RangeError(
      `Mini RADIO-H requires height and width to be positive multiples of 8 (even 4x4 patch grid), got ${width}x${height}`,
    );
  }

  const patchSize = UNI_MINI_VIT_PATCH_SIZE; // 4
  const patchDim = UNI_MINI_VIT_PATCH_DIM; // 48
  const embedDim = UNI_MINI_VIT_EMBED_DIM; // 16
  const numHeads = UNI_MINI_VIT_NUM_HEADS; // 4
  const headDim = UNI_MINI_VIT_HEAD_DIM; // 4
  const mlpDim = UNI_MINI_VIT_MLP_DIM; // 64
  const numPrefixTokens = UNI_MINI_VIT_PREFIX_TOKENS; // 4
  const eps = UNI_MINI_VIT_LAYER_NORM_EPS; // 1e-6

  const patchGridRows = height / patchSize;
  const patchGridCols = width / patchSize;
  const numPatches = patchGridRows * patchGridCols;
  const seqLen = numPrefixTokens + numPatches;

  // 1. CLIP normalize
  const clipNormalized = clipNormalizeImage(image);

  // 2. Extract row-major 4x4 patches in CHW order (48 floats per patch)
  const rawPatches = new Float32Array(numPatches * patchDim);
  const hw = width * height;
  const patchArea = patchSize * patchSize;

  for (let pr = 0; pr < patchGridRows; pr++) {
    for (let pc = 0; pc < patchGridCols; pc++) {
      const patchBase = (pr * patchGridCols + pc) * patchDim;
      for (let c = 0; c < 3; c++) {
        const chanBase = c * hw;
        const patchChanBase = patchBase + c * patchArea;
        for (let dy = 0; dy < patchSize; dy++) {
          const srcRowBase = chanBase + (pr * patchSize + dy) * width + pc * patchSize;
          const dstRowBase = patchChanBase + dy * patchSize;
          for (let dx = 0; dx < patchSize; dx++) {
            rawPatches[dstRowBase + dx] = clipNormalized[srcRowBase + dx];
          }
        }
      }
    }
  }

  // 3. Bias-free patch linear embedder [numPatches, 48] -> [numPatches, 16]
  const patchEmb = linearProject(
    rawPatches,
    numPatches,
    patchDim,
    weights.patchEmbedWeight,
    embedDim,
  );

  // 4. Interpolate 4x4 CPE pos_embed to patchGridRows x patchGridCols
  const interpolatedPosEmbed = interpolatePosEmbed2d(
    weights.posEmbed,
    UNI_MINI_VIT_BASE_GRID,
    UNI_MINI_VIT_BASE_GRID,
    patchGridRows,
    patchGridCols,
    embedDim,
  );

  // 5. Add pos_embed and prepend [4, 16] cls/register prefix tokens
  const vitInputTokens = new Float32Array(seqLen * embedDim);
  vitInputTokens.set(weights.clsToken.subarray(0, numPrefixTokens * embedDim), 0);
  const patchStartOffset = numPrefixTokens * embedDim;
  for (let i = 0; i < numPatches * embedDim; i++) {
    vitInputTokens[patchStartOffset + i] = Math.fround(
      patchEmb[i] + interpolatedPosEmbed[i],
    );
  }

  // 6. Run 2 standard timm ViT blocks
  const attnScale = 1.0 / Math.sqrt(headDim); // 0.5
  let hidden = vitInputTokens;
  const blockTraces: UniRadioVitBlockTrace[] = [];

  for (let b = 0; b < weights.blocks.length; b++) {
    const blk = weights.blocks[b];
    const norm1 = applyLayerNorm(hidden, seqLen, embedDim, blk.norm1Weight, blk.norm1Bias, eps);

    // QKV projection [seqLen, 48] split into Q, K, V [seqLen, 16]
    const qkv = linearWithBias(norm1, seqLen, embedDim, blk.qkvWeight, blk.qkvBias, 3 * embedDim);

    const attnProbs = new Float32Array(numHeads * seqLen * seqLen);
    const attnConcat = new Float32Array(seqLen * embedDim);
    const headScores = new Float32Array(seqLen * seqLen);

    for (let h = 0; h < numHeads; h++) {
      const headOffset = h * headDim;
      for (let qRow = 0; qRow < seqLen; qRow++) {
        const qBase = qRow * (3 * embedDim) + headOffset;
        const scoreRowBase = qRow * seqLen;
        for (let kCol = 0; kCol < seqLen; kCol++) {
          const kBase = kCol * (3 * embedDim) + embedDim + headOffset;
          let dot = 0;
          for (let i = 0; i < headDim; i++) {
            dot += qkv[qBase + i] * qkv[kBase + i];
          }
          headScores[scoreRowBase + kCol] = Math.fround(dot * attnScale);
        }
      }

      const headProbs = softmaxRows(headScores, seqLen, seqLen);
      attnProbs.set(headProbs, h * seqLen * seqLen);

      for (let qRow = 0; qRow < seqLen; qRow++) {
        const probRowBase = qRow * seqLen;
        const outBase = qRow * embedDim + headOffset;
        for (let i = 0; i < headDim; i++) {
          let acc = 0;
          for (let kCol = 0; kCol < seqLen; kCol++) {
            const vBase = kCol * (3 * embedDim) + 2 * embedDim + headOffset;
            acc += headProbs[probRowBase + kCol] * qkv[vBase + i];
          }
          attnConcat[outBase + i] = Math.fround(acc);
        }
      }
    }

    const projOut = linearWithBias(
      attnConcat,
      seqLen,
      embedDim,
      blk.projWeight,
      blk.projBias,
      embedDim,
    );
    const postAttn = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < postAttn.length; i++) {
      postAttn[i] = Math.fround(hidden[i] + projOut[i]);
    }

    const norm2 = applyLayerNorm(
      postAttn,
      seqLen,
      embedDim,
      blk.norm2Weight,
      blk.norm2Bias,
      eps,
    );
    const fc1Out = linearWithBias(norm2, seqLen, embedDim, blk.fc1Weight, blk.fc1Bias, mlpDim);
    const geluOut = new Float32Array(fc1Out.length);
    for (let i = 0; i < fc1Out.length; i++) {
      geluOut[i] = geluExactF32(fc1Out[i]);
    }
    const fc2Out = linearWithBias(geluOut, seqLen, mlpDim, blk.fc2Weight, blk.fc2Bias, embedDim);

    const blockOut = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < blockOut.length; i++) {
      blockOut[i] = Math.fround(postAttn[i] + fc2Out[i]);
    }

    blockTraces.push(
      Object.freeze({
        blockIndex: blk.blockIndex,
        seqLen,
        norm1,
        attnProbs,
        postAttn,
        norm2,
        output: blockOut,
      }),
    );

    hidden = blockOut;
  }

  // 7. Strip 4 prefix tokens via [:, 4:]
  const patchFeatures = new Float32Array(hidden.subarray(numPrefixTokens * embedDim));

  return Object.freeze({
    patchGridRows,
    patchGridCols,
    numPatches,
    numPrefixTokens,
    clipNormalized,
    rawPatches,
    interpolatedPosEmbed,
    vitInputTokens,
    blockTraces: Object.freeze(blockTraces),
    patchFeatures,
  });
}

/**
 * Execute the miniature `RadioHProjector` (`D3` dual-gain preset support):
 *   - `2x2` row-major (`ps_version='v2'`) spatial gather `[patch(2r,2c), patch(2r,2c+1), patch(2r+1,2c), patch(2r+1,2c+1)]` -> `[N_tok, 64]`
 *   - FP32 `RMSNorm(64, eps=1e-5)` -> bias-free `fc1 [8, 64]` -> `SquaredReLU(max(0, x)^2)` -> bias-free `fc2 [8, 8]`
 *   - Multiply by `UNI_GAIN_PRESETS[gainPresetId].radioProjectorMultiply` (`0.25` for `mini_default`, `0.01` for `prod_exact`)
 */
export function runMiniRadioHProjector(
  patchFeatures: Float32Array,
  patchGridRows: number,
  patchGridCols: number,
  weights: UniRadioProjectorWeights,
  gainPresetId: UniGainPresetId = UNI_MINI_CONFIG.defaultGainPreset,
): UniRadioProjectorTrace {
  if (
    !Number.isInteger(patchGridRows) ||
    patchGridRows < 2 ||
    patchGridRows % 2 !== 0 ||
    !Number.isInteger(patchGridCols) ||
    patchGridCols < 2 ||
    patchGridCols % 2 !== 0
  ) {
    throw new RangeError(
      `patchGridRows (${patchGridRows}) and patchGridCols (${patchGridCols}) must be even integers >= 2`,
    );
  }

  const embedDim = UNI_MINI_VIT_EMBED_DIM; // 16
  const projInDim = UNI_MINI_PROJECTOR_IN_DIM; // 64
  const hiddenSize = UNI_MINI_CONFIG.hiddenSize; // 8
  const expectedPatchFloats = patchGridRows * patchGridCols * embedDim;
  if (patchFeatures.length < expectedPatchFloats) {
    throw new RangeError(
      `patchFeatures length (${patchFeatures.length}) is smaller than patchGridRows * patchGridCols * ${embedDim} (${expectedPatchFloats})`,
    );
  }

  const preset = UNI_GAIN_PRESETS[gainPresetId];
  if (!preset) {
    throw new RangeError(`Unknown gainPresetId: ${String(gainPresetId)}`);
  }
  const radioMultiply = preset.radioProjectorMultiply;

  const tokenGridRows = patchGridRows / 2;
  const tokenGridCols = patchGridCols / 2;
  const numVisualTokens = tokenGridRows * tokenGridCols;

  // 1. 2x2 row-major (`ps_version='v2'`) spatial gather
  const gathered2x2 = new Float32Array(numVisualTokens * projInDim);
  for (let br = 0; br < tokenGridRows; br++) {
    for (let bc = 0; bc < tokenGridCols; bc++) {
      const dstBase = (br * tokenGridCols + bc) * projInDim;
      const p00 = ((2 * br) * patchGridCols + (2 * bc)) * embedDim;
      const p01 = ((2 * br) * patchGridCols + (2 * bc + 1)) * embedDim;
      const p10 = ((2 * br + 1) * patchGridCols + (2 * bc)) * embedDim;
      const p11 = ((2 * br + 1) * patchGridCols + (2 * bc + 1)) * embedDim;

      gathered2x2.set(patchFeatures.subarray(p00, p00 + embedDim), dstBase);
      gathered2x2.set(patchFeatures.subarray(p01, p01 + embedDim), dstBase + embedDim);
      gathered2x2.set(patchFeatures.subarray(p10, p10 + embedDim), dstBase + 2 * embedDim);
      gathered2x2.set(patchFeatures.subarray(p11, p11 + embedDim), dstBase + 3 * embedDim);
    }
  }

  // 2. FP32 RMSNorm(64, eps = 1e-5)
  const normed = applyUniRmsNorm(
    gathered2x2,
    numVisualTokens,
    projInDim,
    weights.normWeight,
    UNI_MINI_PROJECTOR_NORM_EPS,
  );

  // 3. Bias-free fc1 [8, 64]
  const fc1Out = linearProject(normed, numVisualTokens, projInDim, weights.fc1Weight, hiddenSize);

  // 4. SquaredReLU: max(0, x)^2
  const squaredReluOut = new Float32Array(fc1Out.length);
  for (let i = 0; i < fc1Out.length; i++) {
    const v = fc1Out[i];
    squaredReluOut[i] = v > 0 ? Math.fround(v * v) : 0.0;
  }

  // 5. Bias-free fc2 [8, 8]
  const fc2Out = linearProject(
    squaredReluOut,
    numVisualTokens,
    hiddenSize,
    weights.fc2Weight,
    hiddenSize,
  );

  // 6. Scale by locked D3 radioProjectorMultiply (0.25 mini_default | 0.01 prod_exact)
  const visualEmbeddings = new Float32Array(fc2Out.length);
  for (let i = 0; i < fc2Out.length; i++) {
    visualEmbeddings[i] = Math.fround(fc2Out[i] * radioMultiply);
  }

  return Object.freeze({
    gainPresetId,
    radioMultiply,
    tokenGridRows,
    tokenGridCols,
    numVisualTokens,
    gathered2x2,
    normed,
    fc1Out,
    squaredReluOut,
    fc2Out,
    visualEmbeddings,
  });
}

/**
 * Run the full continuous vision pipeline (`MiniRadioVisionTower` -> `MiniRadioHProjector`).
 * Strictly NEVER reads `weights.imageNewline` (`U8`).
 */
export function runMiniVisionPipeline(
  image: UniRgbImage,
  weights: UniVisionSubsystemWeights,
  gainPresetId: UniGainPresetId = UNI_MINI_CONFIG.defaultGainPreset,
): UniVisionPipelineTrace {
  const vision = runMiniRadioVisionTower(image, weights.visionTower);
  const projector = runMiniRadioHProjector(
    vision.patchFeatures,
    vision.patchGridRows,
    vision.patchGridCols,
    weights.projector,
    gainPresetId,
  );

  return Object.freeze({
    vision,
    projector,
    imageNewlineUnused: true as const,
  });
}

// ============================================================================
// 5. PROMPT VISUAL-SLOT EXPANSION (`fixOnePrompt`) & NON-MUTATING SPLICING
// ============================================================================

/**
 * Expand the `<SPECIAL_23>` (`23`) image placeholder in a prompt into `tokenGridRows` rows
 * of `tokenGridCols` `<SPECIAL_23>` tokens separated by literal `\n` (`24`) tokens (`U8`).
 *
 * Examples:
 *   - `2x2` -> `[23, 23, 24, 23, 23]` (`4` visual slots + `1` newline = `5` tokens)
 *   - `1x2` -> `[23, 23]` (`2` visual slots + `0` newlines = `2` tokens)
 */
export function fixOnePrompt(
  promptTokenIds: ArrayLike<number>,
  tokenGridRows: number,
  tokenGridCols: number,
): UniFixedPromptResult {
  if (!Number.isInteger(tokenGridRows) || tokenGridRows < 1) {
    throw new RangeError(`tokenGridRows must be an integer >= 1, got ${tokenGridRows}`);
  }
  if (!Number.isInteger(tokenGridCols) || tokenGridCols < 1) {
    throw new RangeError(`tokenGridCols must be an integer >= 1, got ${tokenGridCols}`);
  }

  const originalTokenIds = Int32Array.from(promptTokenIds);
  let placeholderIndex = -1;
  let placeholderCount = 0;
  for (let i = 0; i < originalTokenIds.length; i++) {
    if (originalTokenIds[i] === UNI_MINI_UND_PAD_TOKEN_ID) {
      placeholderIndex = i;
      placeholderCount++;
    }
  }
  if (placeholderCount !== 1) {
    throw new RangeError(
      `fixOnePrompt requires exactly one <SPECIAL_23> (${UNI_MINI_UND_PAD_TOKEN_ID}) image placeholder, found ${placeholderCount}`,
    );
  }

  const numVisualSlots = tokenGridRows * tokenGridCols;
  const numNewlines = tokenGridRows - 1;
  const expandedImageLen = numVisualSlots + numNewlines;
  const expandedLen = originalTokenIds.length - 1 + expandedImageLen;

  const expandedTokenIds = new Int32Array(expandedLen);
  const visualSlotIndices = new Int32Array(numVisualSlots);
  const newlineIndices = new Int32Array(numNewlines);

  let writePos = 0;
  for (let i = 0; i < placeholderIndex; i++) {
    expandedTokenIds[writePos++] = originalTokenIds[i];
  }

  let visIdx = 0;
  let nlIdx = 0;
  for (let r = 0; r < tokenGridRows; r++) {
    for (let c = 0; c < tokenGridCols; c++) {
      visualSlotIndices[visIdx++] = writePos;
      expandedTokenIds[writePos++] = UNI_MINI_UND_PAD_TOKEN_ID;
    }
    if (r < tokenGridRows - 1) {
      newlineIndices[nlIdx++] = writePos;
      expandedTokenIds[writePos++] = UNI_MINI_NEWLINE_TOKEN_ID;
    }
  }

  for (let i = placeholderIndex + 1; i < originalTokenIds.length; i++) {
    expandedTokenIds[writePos++] = originalTokenIds[i];
  }

  return Object.freeze({
    originalTokenIds,
    expandedTokenIds,
    visualSlotIndices,
    newlineIndices,
    tokenGridRows,
    tokenGridCols,
    usedDeadImageNewline: false as const,
  });
}

/**
 * Non-mutating splice of projected visual embeddings into the expanded prompt's
 * text embedding buffer at `visualSlotIndices` (each of which must hold `<SPECIAL_23>` = `23`).
 */
export function spliceVisualEmbeddingsIntoPrompt(
  textTokenIds: ArrayLike<number>,
  textEmbeddings: Float32Array,
  visualSlotIndices: ArrayLike<number>,
  visualEmbeddings: Float32Array,
  hiddenSize: number = UNI_MINI_CONFIG.hiddenSize,
): Float32Array {
  if (!Number.isInteger(hiddenSize) || hiddenSize < 1) {
    throw new RangeError(`hiddenSize must be an integer >= 1, got ${hiddenSize}`);
  }
  const seqLen = textTokenIds.length;
  const expectedTextFloats = seqLen * hiddenSize;
  if (textEmbeddings.length < expectedTextFloats) {
    throw new RangeError(
      `textEmbeddings length (${textEmbeddings.length}) is smaller than seqLen * hiddenSize (${expectedTextFloats})`,
    );
  }
  const numVisualSlots = visualSlotIndices.length;
  const expectedVisualFloats = numVisualSlots * hiddenSize;
  if (visualEmbeddings.length < expectedVisualFloats) {
    throw new RangeError(
      `visualEmbeddings length (${visualEmbeddings.length}) is smaller than visualSlotIndices.length * hiddenSize (${expectedVisualFloats})`,
    );
  }

  const out = new Float32Array(expectedTextFloats);
  out.set(textEmbeddings.subarray(0, expectedTextFloats), 0);

  for (let i = 0; i < numVisualSlots; i++) {
    const slot = visualSlotIndices[i];
    if (!Number.isInteger(slot) || slot < 0 || slot >= seqLen) {
      throw new RangeError(
        `visualSlotIndices[${i}] = ${slot} is out of bounds for seqLen = ${seqLen}`,
      );
    }
    if (textTokenIds[slot] !== UNI_MINI_UND_PAD_TOKEN_ID) {
      throw new RangeError(
        `textTokenIds[${slot}] must be <SPECIAL_23> (${UNI_MINI_UND_PAD_TOKEN_ID}), got ${textTokenIds[slot]}`,
      );
    }
    out.set(
      visualEmbeddings.subarray(i * hiddenSize, (i + 1) * hiddenSize),
      slot * hiddenSize,
    );
  }

  return out;
}
