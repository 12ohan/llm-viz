/**
 * 1D Dyadic YaRN RoPE, Llama-4 Query Scaling & 2D SinCos Spatial Embeddings
 * (`src/uni/rope.ts`)
 *
 * Clean-room first-principles TypeScript implementation matching the pinned
 * mathematical equations of Nemotron-Labs-Diffusion-Uni (`arXiv:2610.36466v1`):
 *   - HF YaRN (`truncate = true` default, half-split `rotate_half`: `[-x2, x1]`)
 *   - Exact 0-ULP dyadic miniature inverse frequencies `[1.0, 0.28125, 0.03125, 0.015625]`
 *   - Llama-4 post-RoPE query-only temperature scaling `1 + beta * ln(1 + floor(p / L_orig))`
 *   - Additive 2D SinCos spatial coordinates (`canonicalContentSize = 32.0`, `scale = 0.0018`,
 *     `markerColumn` width adjustment for 0-ULP source-edit vs. target-gen content alignment)
 *
 * Pure headless TypeScript — zero DOM, WebGL, ambient RNG, or forbidden imports.
 */

import { fmtSigned } from '../math/formulaFormat';
import {
  UNI_MINI_CONFIG,
  type Uni2dSinCosConfig,
  type Uni2dSinCosResult,
  type UniRopeCosSin,
  type UniYarnInvFreqResult,
  type UniYarnPlaneZone,
  type UniYarnRopeConfig,
} from './types';

/**
 * Exact HF YaRN magnitude scaling helper (`get_mscale`).
 *
 * For `mscale = 1, mscaleAllDim = 1`, `getUniYarnMscale(f, 1) / getUniYarnMscale(f, 1) === 1.0`.
 */
export function getUniYarnMscale(scale: number, mscale = 1): number {
  if (scale <= 1) return 1.0;
  return 0.1 * mscale * Math.log(scale) + 1.0;
}

/**
 * Compute YaRN inverse frequencies, ramp boundaries, extrapolation factors,
 * attention scaling factor, and per-plane zone classification.
 *
 * For `UNI_MINI_CONFIG.rope` (`headDim=8, ropeTheta=16, origMaxPos=16,
 * factor=8, betaFast=32, betaSlow=1, mscale=1, mscaleAllDim=1, truncate=true`):
 *   - `low === 0`, `high === 2`
 *   - `extrapolationFactor === [1.0, 0.5, 0.0, 0.0]`
 *   - `invFreq === [1.0, 0.28125, 0.03125, 0.015625]` (0 ULP exact dyadic fractions)
 *   - `attentionScaling === 1.0`
 *   - `zones === ['original', 'blended', 'interpolated', 'interpolated']`
 */
export function computeUniYarnInvFreq(
  ropeConfig: UniYarnRopeConfig,
): UniYarnInvFreqResult {
  const {
    headDim,
    ropeTheta,
    originalMaxPositionEmbeddings: origMaxPos,
    factor,
    betaFast,
    betaSlow,
    mscale,
    mscaleAllDim,
    truncate,
  } = ropeConfig;

  if (!Number.isInteger(headDim) || headDim < 2 || headDim % 2 !== 0) {
    throw new RangeError(`headDim must be an even positive integer, got ${headDim}`);
  }
  if (!(ropeTheta > 0) || !(origMaxPos > 0) || !(factor > 0)) {
    throw new RangeError(
      `ropeTheta (${ropeTheta}), origMaxPos (${origMaxPos}), and factor (${factor}) must be positive`,
    );
  }

  const halfDim = headDim / 2;
  const attentionScaling =
    mscale && mscaleAllDim
      ? getUniYarnMscale(factor, mscale) / getUniYarnMscale(factor, mscaleAllDim)
      : getUniYarnMscale(factor);

  const findCorrectionDim = (numRotations: number): number =>
    (headDim * Math.log(origMaxPos / (numRotations * 2 * Math.PI))) /
    (2 * Math.log(ropeTheta));

  const rawLow = findCorrectionDim(betaFast);
  const rawHigh = findCorrectionDim(betaSlow);

  const low = truncate
    ? Math.max(Math.floor(rawLow), 0)
    : Math.max(rawLow, 0);
  const high = truncate
    ? Math.min(Math.ceil(rawHigh), halfDim - 1)
    : Math.min(rawHigh, halfDim - 1);

  const denom = high === low ? 0.001 : high - low;

  const invFreq = new Float32Array(halfDim);
  const extrapolationFactor = new Float32Array(halfDim);
  const zones: UniYarnPlaneZone[] = new Array<UniYarnPlaneZone>(halfDim);

  for (let i = 0; i < halfDim; i++) {
    const baseInvFreq = 1.0 / Math.pow(ropeTheta, (2 * i) / headDim);
    const invFreqExtrap = baseInvFreq;
    const invFreqInterp = baseInvFreq / factor;

    const linear = (i - low) / denom;
    const clamped = linear < 0 ? 0 : linear > 1 ? 1 : linear;
    const extrap = 1.0 - clamped;

    extrapolationFactor[i] = Math.fround(extrap);
    invFreq[i] = Math.fround(
      invFreqInterp * (1.0 - extrap) + invFreqExtrap * extrap,
    );

    if (extrap === 1.0) {
      zones[i] = 'original';
    } else if (extrap === 0.0) {
      zones[i] = 'interpolated';
    } else {
      zones[i] = 'blended';
    }
  }

  return {
    invFreq,
    attentionScaling,
    low,
    high,
    extrapolationFactor,
    zones: Object.freeze(zones),
  };
}

/**
 * Compute `[seqLen, headDim]` row-major `cos` and `sin` tables for half-split RoPE.
 *
 * Angles `freqs[t, i] = Math.fround(positionIds[t] * invFreq[i])` are duplicated
 * across both halves `[0..halfDim-1]` and `[halfDim..headDim-1]`, then scaled by
 * `attentionScaling`.
 */
export function computeUniRopeCosSin(
  invFreq: Float32Array,
  positionIds: Int32Array,
  attentionScaling: number,
  headDim: number,
): UniRopeCosSin {
  if (!Number.isInteger(headDim) || headDim < 2 || headDim % 2 !== 0) {
    throw new RangeError(`headDim must be an even positive integer, got ${headDim}`);
  }
  const halfDim = headDim / 2;
  if (invFreq.length !== halfDim) {
    throw new RangeError(
      `invFreq length (${invFreq.length}) must equal headDim / 2 (${halfDim})`,
    );
  }

  const seqLen = positionIds.length;
  const cos = new Float32Array(seqLen * headDim);
  const sin = new Float32Array(seqLen * headDim);

  for (let t = 0; t < seqLen; t++) {
    const pos = positionIds[t];
    const rowBase = t * headDim;
    for (let i = 0; i < halfDim; i++) {
      const emb = Math.fround(pos * invFreq[i]);
      const c = Math.fround(Math.cos(emb) * attentionScaling);
      const s = Math.fround(Math.sin(emb) * attentionScaling);
      cos[rowBase + i] = c;
      cos[rowBase + halfDim + i] = c;
      sin[rowBase + i] = s;
      sin[rowBase + halfDim + i] = s;
    }
  }

  return { cos, sin, seqLen, headDim };
}

/**
 * Apply half-split `rotate_half` (`[-x2, x1]`) rotary position embedding to a
 * `[seqLen, numHeads * headDim]` row-major tensor (`[t * numHeads * headDim + h * headDim + d]`).
 */
export function applyUniRotaryPosEmb(
  tensor: Float32Array,
  seqLen: number,
  numHeads: number,
  headDim: number,
  cos: Float32Array,
  sin: Float32Array,
): Float32Array {
  if (!Number.isInteger(seqLen) || seqLen < 0) {
    throw new RangeError(`seqLen must be a non-negative integer, got ${seqLen}`);
  }
  if (!Number.isInteger(numHeads) || numHeads < 1) {
    throw new RangeError(`numHeads must be a positive integer, got ${numHeads}`);
  }
  if (!Number.isInteger(headDim) || headDim < 2 || headDim % 2 !== 0) {
    throw new RangeError(`headDim must be an even positive integer, got ${headDim}`);
  }

  const halfDim = headDim / 2;
  const stride = numHeads * headDim;
  if (tensor.length < seqLen * stride) {
    throw new RangeError(
      `tensor length (${tensor.length}) is smaller than seqLen * numHeads * headDim (${seqLen * stride})`,
    );
  }
  if (cos.length < seqLen * headDim || sin.length < seqLen * headDim) {
    throw new RangeError(
      `cos/sin length (${cos.length}/${sin.length}) is smaller than seqLen * headDim (${seqLen * headDim})`,
    );
  }

  const out = new Float32Array(seqLen * stride);

  for (let t = 0; t < seqLen; t++) {
    const csBase = t * headDim;
    const tBase = t * stride;
    for (let h = 0; h < numHeads; h++) {
      const hBase = tBase + h * headDim;
      for (let i = 0; i < halfDim; i++) {
        const x1 = tensor[hBase + i];
        const x2 = tensor[hBase + halfDim + i];
        const c1 = cos[csBase + i];
        const s1 = sin[csBase + i];
        const c2 = cos[csBase + halfDim + i];
        const s2 = sin[csBase + halfDim + i];

        out[hBase + i] = Math.fround(
          Math.fround(x1 * c1) + Math.fround(-x2 * s1),
        );
        out[hBase + halfDim + i] = Math.fround(
          Math.fround(x2 * c2) + Math.fround(x1 * s2),
        );
      }
    }
  }

  return out;
}

/**
 * Llama-4 post-RoPE query temperature scaling (`_get_llama_4_attn_scale`):
 *   `1.0 + beta * ln(1.0 + floor(cachePosition / origMaxPos))`
 *
 * At `origMaxPos = 16, beta = 0.1`:
 *   - `cachePosition in 0..15`  -> `1.0`
 *   - `cachePosition in 16..31` -> `1 + 0.1 * ln(2) ≈ 1.06931472`
 */
export function computeLlama4AttnScale(
  cachePosition: number,
  beta: number,
  origMaxPos: number,
): number {
  if (!Number.isFinite(cachePosition) || cachePosition < 0) {
    throw new RangeError(
      `cachePosition must be a finite non-negative number, got ${cachePosition}`,
    );
  }
  if (!(origMaxPos > 0)) {
    throw new RangeError(`origMaxPos must be positive, got ${origMaxPos}`);
  }
  const bucket = Math.floor(cachePosition / origMaxPos);
  return 1.0 + beta * Math.log(1.0 + bucket);
}

/**
 * Compute per-token Float32 Llama-4 query temperature scales from `cachePositions`.
 */
export function computeLlama4AttnScales(
  cachePositions: Int32Array,
  beta: number,
  origMaxPos: number,
): Float32Array {
  const out = new Float32Array(cachePositions.length);
  for (let t = 0; t < cachePositions.length; t++) {
    out[t] = Math.fround(
      computeLlama4AttnScale(cachePositions[t], beta, origMaxPos),
    );
  }
  return out;
}

/**
 * Compute additive 2D SinCos spatial position embeddings for generation (`G`)
 * image grids (exact §2.1 `positions.py:31-74` specification).
 *
 * When `markerColumn === true` (target generation grid with appended row/end
 * marker column), `contentWidth = width - 1`, so `(H, W_content + 1, d, true)`
 * matches `(H, W_content, d, false)` bit-for-bit (0 ULP) on all content columns
 * `c in 0..W_content-1`.
 */
export function compute2dSinCosPosEmbed(
  height: number,
  width: number,
  hiddenSize: number,
  markerColumn: boolean,
  config: Uni2dSinCosConfig = UNI_MINI_CONFIG.sinCos2d,
): Uni2dSinCosResult {
  if (!Number.isInteger(height) || height < 1) {
    throw new RangeError(`height must be an integer >= 1, got ${height}`);
  }
  if (!Number.isInteger(width) || width < 1) {
    throw new RangeError(`width must be an integer >= 1, got ${width}`);
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize < 4 || hiddenSize % 4 !== 0) {
    throw new RangeError(
      `hiddenSize must be a positive integer divisible by 4, got ${hiddenSize}`,
    );
  }

  const contentWidth = markerColumn ? width - 1 : width;
  if (contentWidth < 1) {
    throw new RangeError(
      `contentWidth must be >= 1 (width=${width}, markerColumn=${markerColumn})`,
    );
  }

  const { canonicalContentSize, scale, baseTheta } = config;
  if (!(canonicalContentSize > 0) || !(baseTheta > 0)) {
    throw new RangeError(
      `canonicalContentSize (${canonicalContentSize}) and baseTheta (${baseTheta}) must be positive`,
    );
  }

  const quarter = hiddenSize / 4;
  const frequencies = new Float32Array(quarter);
  for (let f = 0; f < quarter; f++) {
    frequencies[f] = Math.fround(Math.pow(baseTheta, -f / quarter));
  }

  const columnStride = canonicalContentSize / contentWidth;
  const rowStride = canonicalContentSize / height;
  const columnStrideF32 = Math.fround(columnStride);
  const rowStrideF32 = Math.fround(rowStride);
  const scaleF32 = Math.fround(scale);

  const embeddings = new Float32Array(height * width * hiddenSize);

  for (let r = 0; r < height; r++) {
    const rowCoord = Math.fround(r * rowStrideF32);
    for (let c = 0; c < width; c++) {
      const colCoord = Math.fround(c * columnStrideF32);
      const cellBase = (r * width + c) * hiddenSize;

      for (let f = 0; f < quarter; f++) {
        const omega = frequencies[f];
        const colAngle = Math.fround(colCoord * omega);
        const rowAngle = Math.fround(rowCoord * omega);

        embeddings[cellBase + f] = Math.fround(
          Math.fround(Math.sin(colAngle)) * scaleF32,
        );
        embeddings[cellBase + quarter + f] = Math.fround(
          Math.fround(Math.cos(colAngle)) * scaleF32,
        );
        embeddings[cellBase + 2 * quarter + f] = Math.fround(
          Math.fround(Math.sin(rowAngle)) * scaleF32,
        );
        embeddings[cellBase + 3 * quarter + f] = Math.fround(
          Math.fround(Math.cos(rowAngle)) * scaleF32,
        );
      }
    }
  }

  return {
    embeddings,
    height,
    width,
    contentWidth,
    hiddenSize,
    markerColumn,
    columnStride,
    rowStride,
    frequencies,
  };
}

export interface UniRopeCellFormulaParams {
  readonly role: 'Q' | 'K';
  readonly tokenIndex: number;
  readonly positionId: number;
  readonly headIndex: number;
  readonly dimIndex: number;
  readonly headDim: number;
  readonly invFreq: number;
  readonly zone: UniYarnPlaneZone;
  readonly x1: number;
  readonly x2: number;
  readonly cosVal: number;
  readonly sinVal: number;
  readonly rotatedVal: number;
  readonly llama4Scale?: number;
  readonly scaledVal?: number;
  readonly branch?: 'U' | 'G';
}

/**
 * Format a concise human-readable equation for a single half-split YaRN RoPE
 * output coordinate (plus optional Llama-4 query temperature multiplier).
 */
export function formatUniRopeCellFormula(
  params: UniRopeCellFormulaParams,
): string {
  const {
    role,
    positionId,
    headIndex,
    dimIndex,
    headDim,
    invFreq,
    zone,
    x1,
    x2,
    cosVal,
    sinVal,
    rotatedVal,
    llama4Scale,
    scaledVal,
    branch,
  } = params;

  const halfDim = headDim / 2;
  const isFirstHalf = dimIndex < halfDim;
  const planeIdx = isFirstHalf ? dimIndex : dimIndex - halfDim;
  const angle = Math.fround(positionId * invFreq);

  const rotExpr = isFirstHalf
    ? `${fmtSigned(x1)}·${fmtSigned(cosVal)} − ${fmtSigned(x2)}·${fmtSigned(sinVal)}`
    : `${fmtSigned(x2)}·${fmtSigned(cosVal)} + ${fmtSigned(x1)}·${fmtSigned(sinVal)}`;

  const branchTag = branch ? `${branch}:` : '';
  const header = `${branchTag}${role}_rot[p=${positionId}, h=${headIndex}, d=${dimIndex}] (plane ${planeIdx} ${zone}, φ=${fmtSigned(angle)} rad)`;

  if (role === 'Q' && llama4Scale !== undefined && scaledVal !== undefined) {
    return `${header} = ${rotExpr} = ${fmtSigned(rotatedVal)} → ×s_L4(${fmtSigned(llama4Scale, 4)}) = ${fmtSigned(scaledVal)}`;
  }

  return `${header} = ${rotExpr} = ${fmtSigned(rotatedVal)}`;
}
