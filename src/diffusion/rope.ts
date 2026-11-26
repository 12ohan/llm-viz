/**
 * Dyadic YaRN Rotary Position Embeddings & Llama-4 Query Temperature Scaling
 * (`src/diffusion/rope.ts`)
 *
 * Clean-room first-principles TypeScript implementation matching the exact
 * mathematical specification of Ministral 3 / Nemotron-Labs-Diffusion-14B:
 *   - Hugging Face YaRN (`truncate = true` default, half-split `rotate_half`)
 *   - Exact dyadic miniature frequencies `[1.0, 0.3125, 0.0625, 0.03125]`
 *   - Llama-4 post-RoPE query-only temperature scaling `1 + beta * ln(1 + floor(p / L_orig))`
 *
 * Pure headless TypeScript — deterministic, self-contained math kernel.
 */

import { fmtSigned } from '../math/formulaFormat';
import type {
  DiffusionRopeCosSin,
  DiffusionYarnInvFreqResult,
  DiffusionYarnRopeConfig,
  YarnPlaneZone,
} from './types';

/**
 * Exact HF YaRN magnitude scaling helper (`get_mscale`).
 *
 * For `mscale = 1, mscaleAllDim = 1`, `getYarnMscale(f, 1) / getYarnMscale(f, 1) === 1.0`.
 */
export function getYarnMscale(scale: number, mscale = 1): number {
  if (scale <= 1) return 1.0;
  return 0.1 * mscale * Math.log(scale) + 1.0;
}

/**
 * Compute YaRN inverse frequencies, ramp boundaries, extrapolation factors,
 * attention scaling factor, and per-plane zone classification.
 *
 * For `DIFFUSION_MINI_CONFIG.rope` (`headDim=8, ropeTheta=16, origMaxPos=16,
 * factor=4, betaFast=32, betaSlow=1, truncate=true`):
 *   - `low === 0`, `high === 2`
 *   - `extrapolationFactor === [1.0, 0.5, 0.0, 0.0]`
 *   - `invFreq === [1.0, 0.3125, 0.0625, 0.03125]` (0 ULP exact binary fractions)
 */
export function computeYarnInvFreq(
  ropeConfig: DiffusionYarnRopeConfig,
): DiffusionYarnInvFreqResult {
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
      ? getYarnMscale(factor, mscale) / getYarnMscale(factor, mscaleAllDim)
      : getYarnMscale(factor);

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
  const zones: YarnPlaneZone[] = new Array<YarnPlaneZone>(halfDim);

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
export function computeRopeCosSin(
  invFreq: Float32Array,
  positionIds: Int32Array,
  attentionScaling: number,
  headDim: number,
): DiffusionRopeCosSin {
  const halfDim = headDim / 2;
  if (!Number.isInteger(halfDim) || invFreq.length !== halfDim) {
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
 * Apply half-split `rotate_half` rotary position embedding to a `[seqLen, numHeads * headDim]`
 * row-major tensor (`[t * numHeads * headDim + h * headDim + d]`).
 *
 * For each rotary plane `i in 0..halfDim-1`:
 *   - `out[..., i]           = x1 * cos[t, i]           + (-x2) * sin[t, i]`
 *   - `out[..., i + halfDim] = x2 * cos[t, i + halfDim] +   x1  * sin[t, i + halfDim]`
 */
export function applyRotaryPosEmb(
  tensor: Float32Array,
  seqLen: number,
  numHeads: number,
  headDim: number,
  cos: Float32Array,
  sin: Float32Array,
): Float32Array {
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

export interface RopeCellFormulaParams {
  readonly role: 'Q' | 'K';
  readonly tokenIndex: number;
  readonly positionId: number;
  readonly headIndex: number;
  readonly dimIndex: number;
  readonly headDim: number;
  readonly invFreq: number;
  readonly zone: YarnPlaneZone;
  readonly x1: number;
  readonly x2: number;
  readonly cosVal: number;
  readonly sinVal: number;
  readonly rotatedVal: number;
  readonly llama4Scale?: number;
  readonly scaledVal?: number;
}

/**
 * Format a concise human-readable equation for a single half-split YaRN RoPE
 * output coordinate (plus optional Llama-4 query temperature multiplier).
 */
export function formatRopeCellFormula(params: RopeCellFormulaParams): string {
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
  } = params;

  const halfDim = headDim / 2;
  const isFirstHalf = dimIndex < halfDim;
  const planeIdx = isFirstHalf ? dimIndex : dimIndex - halfDim;
  const angle = Math.fround(positionId * invFreq);

  const rotExpr = isFirstHalf
    ? `${fmtSigned(x1)}·${fmtSigned(cosVal)} − ${fmtSigned(x2)}·${fmtSigned(sinVal)}`
    : `${fmtSigned(x2)}·${fmtSigned(cosVal)} + ${fmtSigned(x1)}·${fmtSigned(sinVal)}`;

  const header = `${role}_rot[p=${positionId}, h=${headIndex}, d=${dimIndex}] (plane ${planeIdx} ${zone}, φ=${fmtSigned(angle)} rad)`;

  if (role === 'Q' && llama4Scale !== undefined && scaledVal !== undefined) {
    return `${header} = ${rotExpr} = ${fmtSigned(rotatedVal)} → ×s_L4(${fmtSigned(llama4Scale, 4)}) = ${fmtSigned(scaledVal)}`;
  }

  return `${header} = ${rotExpr} = ${fmtSigned(rotatedVal)}`;
}
