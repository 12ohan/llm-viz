/**
 * Full-nGPT Hypersphere Math Kernels & Scale Transforms (`src/dory/ngptMath.ts`).
 *
 * Implements `DoryNgptMathModuleContract` strictly enforcing `D5` (Canonical Browser
 * Numerical Contract):
 *   - `Float32Array` (binary32) storage for all activation/parameter tensors.
 *   - JS native IEEE-754 binary64 (`Number`) accumulation & transcendentals (ZERO per-op `Math.fround()`).
 *   - Binary32 rounding strictly at named `Float32Array` write-back boundaries.
 *   - Fixed ascending `0 .. dim - 1` reduction order for cross-platform 0-ULP determinism.
 *   - Anti-SLERP (`DR7`): channel-wise NLERP `N(a + |alpha_hat| * (b - a))` with unclamped `|alpha_hat| >= 0`.
 *   - All 5 scale families (`DR3`, `DR7`, `DR9`):
 *       1. `attn_alpha` / `mlp_alpha`: `|raw * (alpha_init / sigma_0)|` (linear, `.abs()`, unclamped)
 *       2. `sqk`: `(raw * (sqrt(d_h) / sigma_0))^2` (squared, applied to Q only)
 *       3. `s_gate`: `(raw * (sqrt(g_0) / sigma_0))^2` (squared, honors custom `g_0`, `null -> sqrt(d)`)
 *       4. `suv_gate` / `suv_up`: `raw * sqrt(d)` (linear, SIGNED — NO `.abs()`!)
 *       5. `lm_head.logit_scale`: `raw / sigma_0` (linear, signed)
 *   - Literal loop boundaries (`DR8`): `u_t = N(s_{t-1} + x)` (incl. `u_1 = N(2x)`) and `s_t = N(F_theta(u_t))`.
 *
 * 100% clean-room original TypeScript — zero DOM, WebGL, Node built-ins, or sibling imports.
 */

import {
  DORY_MINI_CONFIG,
  type DoryAnchorJacobianMetrics,
  type DoryMiniConfig,
  type DoryNlerpTrace,
  type DoryScaleFamilyMultipliers,
} from './types';

// ============================================================================
// WEIGHTLESS UNIT-SPHERE L2 NORMALIZATION (ALL 7 ARCHITECTURAL SITES)
// ============================================================================

/**
 * Unit-sphere L2 normalization `x / sqrt(sum(x_c^2) + eps)` from `src[srcOffset..]` into
 * `dst[dstOffset..]`. Safe for in-place (`src === dst && srcOffset === dstOffset`) and
 * near-zero vectors (`N(0) = 0`, zero NaN/Inf). Fixed ascending `0..length-1` reduction.
 */
export function l2NormSliceInto(
  src: Float32Array,
  srcOffset: number,
  dst: Float32Array,
  dstOffset: number,
  length: number,
  eps: number = DORY_MINI_CONFIG.l2NormEps,
): void {
  let sumSq = 0;
  for (let i = 0; i < length; i++) {
    const v = src[srcOffset + i];
    sumSq += v * v;
  }
  const invNorm = 1 / Math.sqrt(sumSq + eps);
  for (let i = 0; i < length; i++) {
    dst[dstOffset + i] = src[srcOffset + i] * invNorm;
  }
}

export function l2NormVector(
  input: Float32Array,
  eps: number = DORY_MINI_CONFIG.l2NormEps,
): Float32Array {
  const out = new Float32Array(input.length);
  l2NormSliceInto(input, 0, out, 0, input.length, eps);
  return out;
}

export function l2NormRows(
  input: Float32Array,
  rows: number,
  cols: number,
  eps: number = DORY_MINI_CONFIG.l2NormEps,
): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    l2NormSliceInto(input, base, out, base, cols, eps);
  }
  return out;
}

export function l2NormHeadsInto(
  src: Float32Array,
  dst: Float32Array,
  seqLen: number,
  numHeads: number,
  headDim: number,
  eps: number = DORY_MINI_CONFIG.l2NormEps,
): void {
  const totalHeads = seqLen * numHeads;
  for (let h = 0; h < totalHeads; h++) {
    const base = h * headDim;
    l2NormSliceInto(src, base, dst, base, headDim, eps);
  }
}

// ============================================================================
// ALL 5 FULL-nGPT LEARNED SCALE FAMILY TRANSFORMS (AUDIT §5.3, DR3, DR7, DR9)
// ============================================================================

/**
 * Resolves `g_0 = attentionOutputGateLogitScaleInit ?? sqrt(hiddenSize)` (`DR3`).
 */
export function resolveOutputGateInitScale(
  config: Pick<DoryMiniConfig, 'hiddenSize' | 'attentionOutputGateLogitScaleInit'>,
): number {
  const custom = config.attentionOutputGateLogitScaleInit;
  if (custom !== null && custom !== undefined) {
    if (!Number.isFinite(custom) || custom <= 0) {
      throw new RangeError(
        `Invalid attentionOutputGateLogitScaleInit ${String(custom)}: expected finite positive number or null.`,
      );
    }
    return custom;
  }
  return Math.sqrt(config.hiddenSize);
}

export function computeScaleFamilyMultipliers(
  config: Pick<
    DoryMiniConfig,
    | 'hiddenSize'
    | 'headDim'
    | 'initializerRange'
    | 'ngptAlphaInit'
    | 'attentionOutputGateLogitScaleInit'
  > = DORY_MINI_CONFIG,
): DoryScaleFamilyMultipliers {
  const sigma0 = config.initializerRange;
  const sGateInitValue = resolveOutputGateInitScale(config);
  return Object.freeze({
    alphaMultiplier: config.ngptAlphaInit / sigma0,
    sqkLinearMultiplier: Math.sqrt(config.headDim) / sigma0,
    sGateInitValue,
    sGateLinearMultiplier: Math.sqrt(sGateInitValue) / sigma0,
    suvMultiplier: Math.sqrt(config.hiddenSize),
    logitScaleMultiplier: 1 / sigma0,
  });
}

/**
 * Family 1 (`attn_alpha` / `mlp_alpha`): `|raw * (alpha_init / sigma_0)|` (unclamped `>= 0`).
 */
export function effectiveResidualAlpha(
  rawAlpha: Float32Array,
  config: Pick<DoryMiniConfig, 'ngptAlphaInit' | 'initializerRange'> = DORY_MINI_CONFIG,
): Float32Array {
  const mult = config.ngptAlphaInit / config.initializerRange;
  const out = new Float32Array(rawAlpha.length);
  for (let i = 0; i < rawAlpha.length; i++) {
    out[i] = Math.abs(rawAlpha[i] * mult);
  }
  return out;
}

/**
 * Family 2 (`mixer.sqk`): `(raw * (sqrt(d_h) / sigma_0))^2` (squared, applied to Q only).
 */
export function effectiveSqkSquared(
  rawSqk: Float32Array,
  config: Pick<DoryMiniConfig, 'headDim' | 'initializerRange'> = DORY_MINI_CONFIG,
): Float32Array {
  const mult = Math.sqrt(config.headDim) / config.initializerRange;
  const out = new Float32Array(rawSqk.length);
  for (let i = 0; i < rawSqk.length; i++) {
    const s = rawSqk[i] * mult;
    out[i] = s * s;
  }
  return out;
}

/**
 * Family 3 (`mixer.s_gate`): `(raw * (sqrt(g_0) / sigma_0))^2` (squared, honors custom `g_0`).
 */
export function effectiveSGateSquared(
  rawSGate: Float32Array,
  config: Pick<
    DoryMiniConfig,
    'hiddenSize' | 'attentionOutputGateLogitScaleInit' | 'initializerRange'
  > = DORY_MINI_CONFIG,
): Float32Array {
  const g0 = resolveOutputGateInitScale(config);
  const mult = Math.sqrt(g0) / config.initializerRange;
  const out = new Float32Array(rawSGate.length);
  for (let i = 0; i < rawSGate.length; i++) {
    const s = rawSGate[i] * mult;
    out[i] = s * s;
  }
  return out;
}

/**
 * Family 4 (`mixer.suv_gate`, `mixer.suv_up`): `raw * sqrt(d)` (linear, SIGNED — NO `.abs()`).
 */
export function effectiveSuvScale(
  rawSuv: Float32Array,
  config: Pick<DoryMiniConfig, 'hiddenSize'> = DORY_MINI_CONFIG,
): Float32Array {
  const mult = Math.sqrt(config.hiddenSize);
  const out = new Float32Array(rawSuv.length);
  for (let i = 0; i < rawSuv.length; i++) {
    out[i] = rawSuv[i] * mult;
  }
  return out;
}

/**
 * Family 5 (`lm_head.logit_scale`): `raw / sigma_0` (linear, signed).
 */
export function effectiveLogitScale(
  rawLogitScale: Float32Array,
  config: Pick<DoryMiniConfig, 'initializerRange'> = DORY_MINI_CONFIG,
): Float32Array {
  const mult = 1 / config.initializerRange;
  const out = new Float32Array(rawLogitScale.length);
  for (let i = 0; i < rawLogitScale.length; i++) {
    out[i] = rawLogitScale[i] * mult;
  }
  return out;
}

// ============================================================================
// CHANNEL-WISE nGPT NLERP RESIDUAL BLENDING (ANTI-SLERP GUARD `DR7`)
// ============================================================================

/**
 * Channel-wise normalized linear interpolation (nGPT NLERP):
 *   `a = N_d(residual)`, `b = N_d(sublayerOut)`, `lambda = |rawAlpha * alpha_init / sigma_0|`,
 *   `output = N_d(a + lambda * (b - a))`.
 * Unclamped `lambda >= 0` allows extrapolation past `b` when `lambda_c > 1`.
 * Produces 0-ULP identical `output` to `ngptNlerpResidualRowsDetailed`.
 */
export function ngptNlerpResidualRows(
  residual: Float32Array,
  sublayerOut: Float32Array,
  rawAlpha: Float32Array,
  seqLen: number,
  hiddenSize: number,
  config: Pick<
    DoryMiniConfig,
    'ngptAlphaInit' | 'initializerRange' | 'l2NormEps'
  > = DORY_MINI_CONFIG,
): Float32Array {
  const eps = config.l2NormEps;
  const effAlpha = effectiveResidualAlpha(rawAlpha, config);
  const aRow = new Float32Array(hiddenSize);
  const bRow = new Float32Array(hiddenSize);
  const blendedRow = new Float32Array(hiddenSize);
  const output = new Float32Array(seqLen * hiddenSize);

  for (let t = 0; t < seqLen; t++) {
    const base = t * hiddenSize;
    l2NormSliceInto(residual, base, aRow, 0, hiddenSize, eps);
    l2NormSliceInto(sublayerOut, base, bRow, 0, hiddenSize, eps);
    for (let c = 0; c < hiddenSize; c++) {
      const a = aRow[c];
      const b = bRow[c];
      blendedRow[c] = a + effAlpha[c] * (b - a);
    }
    l2NormSliceInto(blendedRow, 0, output, base, hiddenSize, eps);
  }

  return output;
}

/**
 * Full-trace variant of `ngptNlerpResidualRows` capturing `aNorm`, `bNorm`, `delta`,
 * `effectiveAlpha`, `blendedPreNorm`, `output`, and `> 1` extrapolation counts (`DR7`).
 */
export function ngptNlerpResidualRowsDetailed(
  residual: Float32Array,
  sublayerOut: Float32Array,
  rawAlpha: Float32Array,
  seqLen: number,
  hiddenSize: number,
  config: Pick<
    DoryMiniConfig,
    'ngptAlphaInit' | 'initializerRange' | 'l2NormEps'
  > = DORY_MINI_CONFIG,
): DoryNlerpTrace {
  const eps = config.l2NormEps;
  const aNorm = l2NormRows(residual, seqLen, hiddenSize, eps);
  const bNorm = l2NormRows(sublayerOut, seqLen, hiddenSize, eps);
  const effectiveAlpha = effectiveResidualAlpha(rawAlpha, config);

  let extrapolatedChannelCount = 0;
  let maxEffectiveAlpha = 0;
  for (let c = 0; c < hiddenSize; c++) {
    const lam = effectiveAlpha[c];
    if (lam > 1.0) {
      extrapolatedChannelCount++;
    }
    if (lam > maxEffectiveAlpha) {
      maxEffectiveAlpha = lam;
    }
  }

  const total = seqLen * hiddenSize;
  const delta = new Float32Array(total);
  const blendedPreNorm = new Float32Array(total);
  const output = new Float32Array(total);

  for (let t = 0; t < seqLen; t++) {
    const base = t * hiddenSize;
    for (let c = 0; c < hiddenSize; c++) {
      const a = aNorm[base + c];
      const b = bNorm[base + c];
      const diff = b - a;
      delta[base + c] = diff;
      blendedPreNorm[base + c] = a + effectiveAlpha[c] * diff;
    }
    l2NormSliceInto(blendedPreNorm, base, output, base, hiddenSize, eps);
  }

  return {
    seqLen,
    hiddenSize,
    aNorm,
    bNorm,
    delta,
    effectiveAlpha,
    blendedPreNorm,
    output,
    extrapolatedChannelCount,
    maxEffectiveAlpha,
  };
}

// ============================================================================
// LITERAL LOOP-BOUNDARY NORMALIZATIONS (`DR8`: `u_t = N(s_{t-1} + x)`, `s_t = N(F(u_t))`)
// ============================================================================

/**
 * Literal loop-entry anchor re-injection `u_t = N_d(prevState + xAnchor)` (`DR8`).
 * Never elides `u_1 = N_d(2x)` when `prevState` equals `xAnchor`.
 */
export function loopEntryAnchorInject(
  prevState: Float32Array,
  xAnchor: Float32Array,
  seqLen: number,
  hiddenSize: number,
  eps: number = DORY_MINI_CONFIG.l2NormEps,
): Float32Array {
  const sumRow = new Float32Array(hiddenSize);
  const out = new Float32Array(seqLen * hiddenSize);
  for (let t = 0; t < seqLen; t++) {
    const base = t * hiddenSize;
    for (let c = 0; c < hiddenSize; c++) {
      sumRow[c] = prevState[base + c] + xAnchor[base + c];
    }
    l2NormSliceInto(sumRow, 0, out, base, hiddenSize, eps);
  }
  return out;
}

/**
 * Literal loop-exit normalization `s_t = N_d(F_theta(u_t))` (`DR8`).
 */
export function loopExitNormalize(
  coreOut: Float32Array,
  seqLen: number,
  hiddenSize: number,
  eps: number = DORY_MINI_CONFIG.l2NormEps,
): Float32Array {
  return l2NormRows(coreOut, seqLen, hiddenSize, eps);
}

// ============================================================================
// BIAS-FREE LINEAR PROJECTIONS (FIXED ASCENDING REDUCTION ORDER `0..inDim-1`)
// ============================================================================

/**
 * Bias-free linear projection `dst = input @ weight^T` (`input: [seqLen, inDim]`,
 * `weight: [outDim, inDim]`, `dst: [seqLen, outDim]`). Binary64 dot accumulation in
 * fixed ascending `0..inDim-1` order; binary32 write-back on `dst`.
 */
export function linearProjectInto(
  input: Float32Array,
  weight: Float32Array,
  dst: Float32Array,
  seqLen: number,
  inDim: number,
  outDim: number,
): void {
  for (let t = 0; t < seqLen; t++) {
    const inBase = t * inDim;
    const outBase = t * outDim;
    for (let o = 0; o < outDim; o++) {
      const wBase = o * inDim;
      let acc = 0;
      for (let k = 0; k < inDim; k++) {
        acc += input[inBase + k] * weight[wBase + k];
      }
      dst[outBase + o] = acc;
    }
  }
}

export function linearProject(
  input: Float32Array,
  weight: Float32Array,
  seqLen: number,
  inDim: number,
  outDim: number,
): Float32Array {
  const dst = new Float32Array(seqLen * outDim);
  linearProjectInto(input, weight, dst, seqLen, inDim, outDim);
  return dst;
}

// ============================================================================
// SCALAR TRANSCENDENTALS, BINARY64 SOFTMAX & HYPERSPHERE GEOMETRY METRICS
// ============================================================================

export function sigmoidScalar(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

export function siluScalar(x: number): number {
  return x * sigmoidScalar(x);
}

/**
 * Binary64 softmax over `logits[offset .. offset + length - 1]` with temperature `T >= 0`.
 * At `T === 0` returns a deterministic one-hot distribution at `argmaxIndex`.
 */
export function softmaxFloat64(
  logits: ArrayLike<number>,
  offset = 0,
  length: number = logits.length - offset,
  temperature = 1.0,
): Float64Array {
  if (length <= 0) {
    return new Float64Array(0);
  }
  if (!Number.isFinite(temperature) || temperature < 0) {
    throw new RangeError(
      `Invalid softmax temperature ${String(temperature)}: expected finite number >= 0.`,
    );
  }

  const probs = new Float64Array(length);
  if (temperature === 0) {
    probs[argmaxIndex(logits, offset, length)] = 1.0;
    return probs;
  }

  const invTemp = 1 / temperature;
  let maxVal = -Infinity;
  for (let i = 0; i < length; i++) {
    const v = logits[offset + i];
    const scaled = v === -Infinity ? -Infinity : v * invTemp;
    if (scaled > maxVal) {
      maxVal = scaled;
    }
  }

  if (!Number.isFinite(maxVal)) {
    return probs;
  }

  let sumExp = 0;
  for (let i = 0; i < length; i++) {
    const v = logits[offset + i];
    if (v === -Infinity) {
      probs[i] = 0;
      continue;
    }
    const e = Math.exp(v * invTemp - maxVal);
    probs[i] = e;
    sumExp += e;
  }

  if (sumExp > 0) {
    for (let i = 0; i < length; i++) {
      probs[i] /= sumExp;
    }
  }
  return probs;
}

/**
 * Deterministic argmax in `[0, length - 1]` (ties broken to lowest index).
 */
export function argmaxIndex(
  values: ArrayLike<number>,
  offset = 0,
  length: number = values.length - offset,
): number {
  if (length <= 0) {
    throw new RangeError(`argmaxIndex requires length > 0, got ${String(length)}.`);
  }
  let bestIdx = 0;
  let bestVal = values[offset];
  for (let i = 1; i < length; i++) {
    const v = values[offset + i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

export function vectorL2Norm(
  v: ArrayLike<number>,
  offset = 0,
  length: number = v.length - offset,
): number {
  let sumSq = 0;
  for (let i = 0; i < length; i++) {
    const x = v[offset + i];
    sumSq += x * x;
  }
  return Math.sqrt(sumSq);
}

export function l2Distance(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  aOffset = 0,
  bOffset = 0,
  length: number = Math.min(a.length - aOffset, b.length - bOffset),
): number {
  let sumSq = 0;
  for (let i = 0; i < length; i++) {
    const d = a[aOffset + i] - b[bOffset + i];
    sumSq += d * d;
  }
  return Math.sqrt(sumSq);
}

export function cosineSimilarity8d(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  aOffset = 0,
  bOffset = 0,
  length: number = Math.min(a.length - aOffset, b.length - bOffset),
): number {
  let dot = 0;
  let normASq = 0;
  let normBSq = 0;
  for (let i = 0; i < length; i++) {
    const va = a[aOffset + i];
    const vb = b[bOffset + i];
    dot += va * vb;
    normASq += va * va;
    normBSq += vb * vb;
  }
  const denom = Math.sqrt(normASq * normBSq);
  if (denom === 0) {
    return 0;
  }
  const cos = dot / denom;
  return cos > 1 ? 1 : cos < -1 ? -1 : cos;
}

/**
 * Local anchor-map Jacobian sensitivity metrics for unit vectors `s, x in S^{d-1}` separated
 * by `angleRadians in [0, pi]` (Audit §6.2.1, Finding 9).
 *
 * NOTE: `isLocalContractiveRegime` (`theta < 120°`) describes ONLY the boundary map
 * `s -> N_eps(s + x)`, NOT contraction or convergence of the full 40-sublayer recurrent core.
 */
export function anchorMapJacobianNorm(
  angleRadians: number,
  eps: number = DORY_MINI_CONFIG.l2NormEps,
): DoryAnchorJacobianMetrics {
  const clampedAngle = Math.max(0, Math.min(Math.PI, Math.abs(angleRadians)));
  const angleDegrees = (clampedAngle * 180) / Math.PI;
  const cosHalf = Math.max(0, Math.cos(clampedAngle * 0.5));
  const sumNormIdeal = 2 * cosHalf;
  const idealJacobianNorm = sumNormIdeal === 0 ? Infinity : 1 / sumNormIdeal;
  const epsStabilizedJacobianNorm = 1 / Math.sqrt(4 * cosHalf * cosHalf + eps);
  const isLocalContractiveRegime = clampedAngle < (2 * Math.PI) / 3;

  return Object.freeze({
    angleRadians: clampedAngle,
    angleDegrees,
    sumNormIdeal,
    idealJacobianNorm,
    epsStabilizedJacobianNorm,
    isLocalContractiveRegime,
  });
}
