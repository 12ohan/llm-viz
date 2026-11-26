/**
 * Elastic-MoT Routed RMSNorm, Bias-Free Linear Projections, Softmax & 8Q/2KV GQA
 * (`src/uni/attention.ts`)
 *
 * Clean-room first-principles TypeScript implementation matching NLD-Uni:
 *   - Routed pre-attention / post-attention RMSNorm (`U` vs `G` weights, `skipU` pruning)
 *   - Bias-free `F.linear` & per-token routed `U`/`G` linear projections
 *   - Shared 1D YaRN RoPE + Llama-4 query-only temperature scaling
 *   - 8Q / 2KV (`kvGroupSize = 4`) Grouped-Query Attention with joint (`L0..17`) vs.
 *     split-tower (`L18..33`) routing isolation
 *   - D5 0-ULP invariance: `dropCachedUnderstanding` ignores `pastKv` columns `0..pastLen-1`
 *     entirely (supporting `pastKv = null` vs non-null `U` `pastKv` at 0 ULP), and
 *     `pruneUnderstandingTokens` skips `U` rows cleanly at 0 ULP on `G` tokens
 *   - Strictly non-mutating on `pastKv` and `pastRouting`
 *
 * Pure headless TypeScript — zero DOM, WebGL, ambient RNG, or forbidden imports.
 */

import { fmtSigned } from '../math/formulaFormat';
import {
  applyElasticSplitTowerMask,
  buildBidirectionalMask,
  buildCausalMask,
  buildSbdImageMask,
  canonicalizeTokenRouting,
} from './masks';
import {
  applyUniRotaryPosEmb,
  computeLlama4AttnScales,
  computeUniRopeCosSin,
} from './rope';
import type {
  UniAttentionMaskMode,
  UniElasticAttentionInput,
  UniElasticAttentionTrace,
} from './types';

/**
 * Ministral 3 RMSNorm: computes mean square per row, multiplies by `rsqrt(meanSq + eps)`,
 * rounds normalized coordinate to Float32, and scales by `weight[d]` in Float32.
 */
export function applyUniRmsNorm(
  x: Float32Array,
  seqLen: number,
  hiddenSize: number,
  weight: Float32Array,
  eps: number,
): Float32Array {
  if (!Number.isInteger(seqLen) || seqLen < 0) {
    throw new RangeError(`seqLen must be a non-negative integer, got ${seqLen}`);
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize < 1) {
    throw new RangeError(`hiddenSize must be >= 1, got ${hiddenSize}`);
  }
  if (x.length < seqLen * hiddenSize) {
    throw new RangeError(
      `x length (${x.length}) is smaller than seqLen * hiddenSize (${seqLen * hiddenSize})`,
    );
  }
  if (weight.length !== hiddenSize) {
    throw new RangeError(
      `weight length (${weight.length}) must equal hiddenSize (${hiddenSize})`,
    );
  }

  const out = new Float32Array(seqLen * hiddenSize);

  for (let t = 0; t < seqLen; t++) {
    const base = t * hiddenSize;
    let sumSq = 0;
    for (let d = 0; d < hiddenSize; d++) {
      const v = x[base + d];
      sumSq += v * v;
    }
    const meanSq = sumSq / hiddenSize;
    const invRms = 1.0 / Math.sqrt(meanSq + eps);
    for (let d = 0; d < hiddenSize; d++) {
      const normed = Math.fround(x[base + d] * invRms);
      out[base + d] = Math.fround(normed * weight[d]);
    }
  }

  return out;
}

/**
 * Per-token routed RMSNorm (`0` = `weightU`, `1` = `weightG`).
 * When `skipU` is true, `U` rows (`routing[t] === 0`) are left zeroed without reading `weightU`.
 */
export function applyRoutedUniRmsNorm(
  x: Float32Array,
  seqLen: number,
  hiddenSize: number,
  routing: Uint8Array,
  weightU: Float32Array,
  weightG: Float32Array,
  eps: number,
  skipU = false,
): Float32Array {
  if (!Number.isInteger(seqLen) || seqLen < 0) {
    throw new RangeError(`seqLen must be a non-negative integer, got ${seqLen}`);
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize < 1) {
    throw new RangeError(`hiddenSize must be >= 1, got ${hiddenSize}`);
  }
  if (routing.length !== seqLen) {
    throw new RangeError(
      `routing length (${routing.length}) must equal seqLen (${seqLen})`,
    );
  }
  if (x.length < seqLen * hiddenSize) {
    throw new RangeError(
      `x length (${x.length}) is smaller than seqLen * hiddenSize (${seqLen * hiddenSize})`,
    );
  }
  if (weightU.length !== hiddenSize || weightG.length !== hiddenSize) {
    throw new RangeError(
      `weightU (${weightU.length}) and weightG (${weightG.length}) lengths must equal hiddenSize (${hiddenSize})`,
    );
  }

  const out = new Float32Array(seqLen * hiddenSize);

  for (let t = 0; t < seqLen; t++) {
    const r = routing[t];
    if (r !== 0 && r !== 1) {
      throw new RangeError(`routing[${t}] must be 0 or 1, received ${r}`);
    }
    if (skipU && r === 0) continue;

    const w = r === 0 ? weightU : weightG;
    const base = t * hiddenSize;
    let sumSq = 0;
    for (let d = 0; d < hiddenSize; d++) {
      const v = x[base + d];
      sumSq += v * v;
    }
    const meanSq = sumSq / hiddenSize;
    const invRms = 1.0 / Math.sqrt(meanSq + eps);
    for (let d = 0; d < hiddenSize; d++) {
      const normed = Math.fround(x[base + d] * invRms);
      out[base + d] = Math.fround(normed * w[d]);
    }
  }

  return out;
}

/**
 * Bias-free linear projection matching PyTorch `F.linear(input, weight)`:
 *   - `input`: `[seqLen, inDim]` row-major
 *   - `weight`: `[outDim, inDim]` row-major
 *   - returns `[seqLen, outDim]` row-major Float32Array
 */
export function linearProject(
  input: Float32Array,
  seqLen: number,
  inDim: number,
  weight: Float32Array,
  outDim: number,
): Float32Array {
  if (!Number.isInteger(seqLen) || seqLen < 0) {
    throw new RangeError(`seqLen must be a non-negative integer, got ${seqLen}`);
  }
  if (!Number.isInteger(inDim) || inDim < 1 || !Number.isInteger(outDim) || outDim < 1) {
    throw new RangeError(`inDim (${inDim}) and outDim (${outDim}) must be >= 1`);
  }
  if (input.length < seqLen * inDim) {
    throw new RangeError(
      `input length (${input.length}) is smaller than seqLen * inDim (${seqLen * inDim})`,
    );
  }
  if (weight.length < outDim * inDim) {
    throw new RangeError(
      `weight length (${weight.length}) is smaller than outDim * inDim (${outDim * inDim})`,
    );
  }

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
      out[yBase + o] = Math.fround(acc);
    }
  }

  return out;
}

/**
 * Per-token routed bias-free linear projection (`0` = `weightU`, `1` = `weightG`).
 * When `skipU` is true, `U` rows (`routing[t] === 0`) are left zeroed without reading `weightU`.
 */
export function routedLinearProject(
  input: Float32Array,
  seqLen: number,
  inDim: number,
  routing: Uint8Array,
  weightU: Float32Array,
  weightG: Float32Array,
  outDim: number,
  skipU = false,
): Float32Array {
  if (!Number.isInteger(seqLen) || seqLen < 0) {
    throw new RangeError(`seqLen must be a non-negative integer, got ${seqLen}`);
  }
  if (!Number.isInteger(inDim) || inDim < 1 || !Number.isInteger(outDim) || outDim < 1) {
    throw new RangeError(`inDim (${inDim}) and outDim (${outDim}) must be >= 1`);
  }
  if (routing.length !== seqLen) {
    throw new RangeError(
      `routing length (${routing.length}) must equal seqLen (${seqLen})`,
    );
  }
  if (input.length < seqLen * inDim) {
    throw new RangeError(
      `input length (${input.length}) is smaller than seqLen * inDim (${seqLen * inDim})`,
    );
  }
  if (weightU.length < outDim * inDim || weightG.length < outDim * inDim) {
    throw new RangeError(
      `weightU (${weightU.length}) and weightG (${weightG.length}) must be >= outDim * inDim (${outDim * inDim})`,
    );
  }

  const out = new Float32Array(seqLen * outDim);

  for (let t = 0; t < seqLen; t++) {
    const r = routing[t];
    if (r !== 0 && r !== 1) {
      throw new RangeError(`routing[${t}] must be 0 or 1, received ${r}`);
    }
    if (skipU && r === 0) continue;

    const w = r === 0 ? weightU : weightG;
    const xBase = t * inDim;
    const yBase = t * outDim;
    for (let o = 0; o < outDim; o++) {
      const wBase = o * inDim;
      let acc = 0;
      for (let i = 0; i < inDim; i++) {
        acc += input[xBase + i] * w[wBase + i];
      }
      out[yBase + o] = Math.fround(acc);
    }
  }

  return out;
}

/**
 * Numerically stable row-wise softmax returning Float32 probabilities.
 *
 * Rows where every entry is `-Infinity` (fully masked) emit exact `0.0`s.
 * Masked entries (`-Infinity`) always emit exact `0.0`.
 */
export function softmaxRows(
  scores: Float32Array,
  rows: number,
  cols: number,
): Float32Array {
  if (!Number.isInteger(rows) || rows < 0 || !Number.isInteger(cols) || cols < 0) {
    throw new RangeError(`rows (${rows}) and cols (${cols}) must be >= 0`);
  }
  if (scores.length < rows * cols) {
    throw new RangeError(
      `scores length (${scores.length}) is smaller than rows * cols (${rows * cols})`,
    );
  }

  const out = new Float32Array(rows * cols);
  if (rows === 0 || cols === 0) return out;

  const expScratch = new Float64Array(cols);

  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let maxVal = -Infinity;
    for (let c = 0; c < cols; c++) {
      const s = scores[base + c];
      if (s > maxVal) maxVal = s;
    }

    if (!Number.isFinite(maxVal)) {
      continue;
    }

    let sumExp = 0;
    for (let c = 0; c < cols; c++) {
      const s = scores[base + c];
      if (s === -Infinity || (!Number.isFinite(s) && s < 0)) {
        expScratch[c] = 0;
      } else {
        const e = Math.exp(s - maxVal);
        expScratch[c] = e;
        sumExp += e;
      }
    }

    if (sumExp <= 0 || !Number.isFinite(sumExp)) continue;

    for (let c = 0; c < cols; c++) {
      const e = expScratch[c];
      out[base + c] = e === 0 ? 0.0 : Math.fround(e / sumExp);
    }
  }

  return out;
}

function resolveBaseMask(
  maskMode: UniAttentionMaskMode,
  seqLen: number,
  pastLen: number,
  kvLen: number,
  customMask?: Uint8Array,
  imageBlockStart?: number,
  imageBlockEnd?: number,
): Uint8Array {
  if (maskMode === 'causal') {
    return buildCausalMask(seqLen, pastLen);
  }
  if (maskMode === 'bidirectional') {
    return buildBidirectionalMask(seqLen, pastLen);
  }
  if (maskMode === 'sbd_image') {
    if (imageBlockStart === undefined) {
      throw new RangeError(
        "imageBlockStart is required when maskMode === 'sbd_image'",
      );
    }
    const full = buildSbdImageMask(kvLen, imageBlockStart, imageBlockEnd);
    return pastLen === 0 ? full : full.subarray(pastLen * kvLen);
  }
  if (!customMask || customMask.length !== seqLen * kvLen) {
    throw new RangeError(
      `customMask of length seqLen * kvLen (${seqLen * kvLen}) is required when maskMode === 'custom'`,
    );
  }
  return customMask;
}

/**
 * Run single-layer Elastic-MoT 8Q/2KV Grouped-Query Attention with shared 1D YaRN RoPE,
 * Llama-4 query temperature scaling, joint/split-tower routing isolation, and strict
 * D5 0-ULP pruned-execution invariance.
 */
export function runUniElasticAttention(
  input: UniElasticAttentionInput,
): UniElasticAttentionTrace {
  const {
    hiddenStates,
    seqLen,
    routing,
    layerWeights,
    config,
    invFreq,
    attentionScaling,
    positionIds,
    cachePositions,
    maskMode,
    customMask,
    imageBlockStart,
    imageBlockEnd,
    pastKv,
    pastRouting,
    pruneUnderstandingTokens = false,
  } = input;

  if (!Number.isInteger(seqLen) || seqLen < 1) {
    throw new RangeError(`seqLen must be an integer >= 1, got ${seqLen}`);
  }

  const d = config.hiddenSize;
  const nQ = config.numAttentionHeads;
  const nKv = config.numKeyValueHeads;
  const kvGroupSize = config.kvGroupSize;
  const headDim = config.headDim;
  const qDim = config.qProjOutDim;
  const kvDim = config.kvProjOutDim;

  if (hiddenStates.length < seqLen * d) {
    throw new RangeError(
      `hiddenStates length (${hiddenStates.length}) is smaller than seqLen * hiddenSize (${seqLen * d})`,
    );
  }
  if (routing.length !== seqLen) {
    throw new RangeError(
      `routing length (${routing.length}) must equal seqLen (${seqLen})`,
    );
  }
  if (positionIds.length !== seqLen) {
    throw new RangeError(
      `positionIds length (${positionIds.length}) must equal seqLen (${seqLen})`,
    );
  }
  const effectiveCachePositions = cachePositions ?? positionIds;
  if (effectiveCachePositions.length !== seqLen) {
    throw new RangeError(
      `cachePositions length (${effectiveCachePositions.length}) must equal seqLen (${seqLen})`,
    );
  }

  const routingPlan = canonicalizeTokenRouting(routing);
  const routingUint8 = routingPlan.routing;
  const canonicalRouting = routingPlan.canonical;
  const splitTower = layerWeights.splitTower;

  const pastLen = pastRouting ? pastRouting.length : pastKv ? pastKv.seqLen : 0;
  if (pastKv && pastRouting && pastKv.seqLen !== pastRouting.length) {
    throw new RangeError(
      `pastKv.seqLen (${pastKv.seqLen}) must equal pastRouting.length (${pastRouting.length})`,
    );
  }
  const kvLen = pastLen + seqLen;

  let cachedHasGeneration = false;
  if (pastRouting) {
    for (let i = 0; i < pastRouting.length; i++) {
      const r = pastRouting[i];
      if (r !== 0 && r !== 1) {
        throw new RangeError(`pastRouting[${i}] must be 0 or 1, received ${r}`);
      }
      if (r === 1) cachedHasGeneration = true;
    }
  }

  const dropCachedUnderstanding =
    splitTower &&
    canonicalRouting === 'g_only' &&
    pastLen > 0 &&
    !cachedHasGeneration;

  if (pastLen > 0 && !dropCachedUnderstanding) {
    if (!pastKv || pastKv.seqLen !== pastLen) {
      throw new RangeError(
        `Non-null pastKv with seqLen === ${pastLen} is required when pastLen > 0 and !dropCachedUnderstanding`,
      );
    }
    const reqPastFloats = pastLen * kvDim;
    if (pastKv.k.length < reqPastFloats || pastKv.v.length < reqPastFloats) {
      throw new RangeError(
        `pastKv buffer too small for pastLen=${pastLen} * kvDim=${kvDim}`,
      );
    }
  }

  const skipU = Boolean(splitTower && pruneUnderstandingTokens);
  const understandingPruned = Boolean(skipU && routingPlan.hasUnderstanding);

  // 1. Routed Q, K, V linear projections (bias-free)
  const qPre = routedLinearProject(
    hiddenStates,
    seqLen,
    d,
    routingUint8,
    layerWeights.u.selfAttn.qProj,
    layerWeights.g.selfAttn.qProj,
    qDim,
    skipU,
  );
  const kPre = routedLinearProject(
    hiddenStates,
    seqLen,
    d,
    routingUint8,
    layerWeights.u.selfAttn.kProj,
    layerWeights.g.selfAttn.kProj,
    kvDim,
    skipU,
  );
  const vNew = routedLinearProject(
    hiddenStates,
    seqLen,
    d,
    routingUint8,
    layerWeights.u.selfAttn.vProj,
    layerWeights.g.selfAttn.vProj,
    kvDim,
    skipU,
  );

  // 2. Shared 1D YaRN RoPE
  const { cos, sin } = computeUniRopeCosSin(
    invFreq,
    positionIds,
    attentionScaling,
    headDim,
  );
  const qRot = applyUniRotaryPosEmb(qPre, seqLen, nQ, headDim, cos, sin);
  const kRot = applyUniRotaryPosEmb(kPre, seqLen, nKv, headDim, cos, sin);
  if (skipU) {
    for (let t = 0; t < seqLen; t++) {
      if (routingUint8[t] === 0) {
        qRot.fill(0, t * qDim, (t + 1) * qDim);
        kRot.fill(0, t * kvDim, (t + 1) * kvDim);
      }
    }
  }

  // 3. Llama-4 post-RoPE query-only temperature scaling (keys untouched)
  const llama4Scales = computeLlama4AttnScales(
    effectiveCachePositions,
    config.rope.llama4Beta,
    config.rope.originalMaxPositionEmbeddings,
  );
  const qScaled = new Float32Array(seqLen * qDim);
  for (let t = 0; t < seqLen; t++) {
    if (skipU && routingUint8[t] === 0) continue;
    const s = llama4Scales[t];
    const base = t * qDim;
    for (let j = 0; j < qDim; j++) {
      qScaled[base + j] = Math.fround(qRot[base + j] * s);
    }
  }

  // 4. Joint vs. split-tower effective attention mask
  const baseMask = resolveBaseMask(
    maskMode,
    seqLen,
    pastLen,
    kvLen,
    customMask,
    imageBlockStart,
    imageBlockEnd,
  );
  const kvRouting = new Uint8Array(kvLen);
  if (pastLen > 0 && pastRouting) {
    kvRouting.set(pastRouting, 0);
  }
  kvRouting.set(routingUint8, pastLen);

  const effectiveMask = applyElasticSplitTowerMask(
    baseMask,
    routingUint8,
    kvRouting,
    splitTower ? config.dualTowerLayers : 0,
    config.dualTowerLayers,
  );

  // 5. GQA scaled dot-product attention (8Q / 2KV, 4 Q heads per KV head)
  // Notice: when effectiveMask[qRow * kvLen + kCol] === 0 (including all 0..pastLen-1
  // when dropCachedUnderstanding is true), pastKv.k and pastKv.v are NEVER read!
  const scale = 1.0 / Math.sqrt(headDim);
  const scores = new Float32Array(nQ * seqLen * kvLen);
  const probs = new Float32Array(nQ * seqLen * kvLen);
  const attnConcat = new Float32Array(seqLen * qDim);

  const pastK = dropCachedUnderstanding ? null : pastKv?.k ?? null;
  const pastV = dropCachedUnderstanding ? null : pastKv?.v ?? null;

  for (let h = 0; h < nQ; h++) {
    const kvHead = Math.floor(h / kvGroupSize);
    const headPlaneOffset = h * seqLen * kvLen;
    const kvHeadOffset = kvHead * headDim;

    for (let qRow = 0; qRow < seqLen; qRow++) {
      const rowScoreOffset = headPlaneOffset + qRow * kvLen;
      if (skipU && routingUint8[qRow] === 0) {
        scores.fill(-Infinity, rowScoreOffset, rowScoreOffset + kvLen);
        continue;
      }

      const qVecOffset = qRow * qDim + h * headDim;
      const maskRowOffset = qRow * kvLen;

      for (let kCol = 0; kCol < kvLen; kCol++) {
        if (effectiveMask[maskRowOffset + kCol] === 0) {
          scores[rowScoreOffset + kCol] = -Infinity;
        } else {
          const kSrc = kCol < pastLen ? pastK! : kRot;
          const kIdx = kCol < pastLen ? kCol : kCol - pastLen;
          const kVecOffset = kIdx * kvDim + kvHeadOffset;
          let dot = 0;
          for (let i = 0; i < headDim; i++) {
            dot += qScaled[qVecOffset + i] * kSrc[kVecOffset + i];
          }
          scores[rowScoreOffset + kCol] = Math.fround(dot * scale);
        }
      }

      const rowProbs = softmaxRows(
        scores.subarray(rowScoreOffset, rowScoreOffset + kvLen),
        1,
        kvLen,
      );
      probs.set(rowProbs, rowScoreOffset);

      const outVecOffset = qRow * qDim + h * headDim;
      for (let i = 0; i < headDim; i++) {
        let acc = 0;
        for (let kCol = 0; kCol < kvLen; kCol++) {
          if (effectiveMask[maskRowOffset + kCol] === 0) continue;
          const p = probs[rowScoreOffset + kCol];
          if (p !== 0) {
            const vSrc = kCol < pastLen ? pastV! : vNew;
            const vIdx = kCol < pastLen ? kCol : kCol - pastLen;
            acc += p * vSrc[vIdx * kvDim + kvHeadOffset + i];
          }
        }
        attnConcat[outVecOffset + i] = Math.fround(acc);
      }
    }
  }

  // 6. Routed output projection (o_proj / o_proj_gen)
  const output = routedLinearProject(
    attnConcat,
    seqLen,
    qDim,
    routingUint8,
    layerWeights.u.selfAttn.oProj,
    layerWeights.g.selfAttn.oProj,
    d,
    skipU,
  );

  return {
    layerIndex: layerWeights.layerIndex,
    splitTower,
    seqLen,
    pastLen,
    kvLen,
    canonicalRouting,
    cachedHasGeneration,
    dropCachedUnderstanding,
    understandingPruned,
    qPre,
    kPre,
    vNew,
    qRot,
    kRot,
    llama4Scales,
    qScaled,
    effectiveMask,
    scores,
    probs,
    attnConcat,
    output,
  };
}

export interface UniAttentionCellFormulaParams {
  readonly stage: 'score' | 'prob' | 'o_proj';
  readonly qHead: number;
  readonly kvHead: number;
  readonly qRow: number;
  readonly kCol?: number;
  readonly outDimIndex?: number;
  readonly headDim: number;
  readonly branch: 'U' | 'G';
  readonly splitTower: boolean;
  readonly isMasked?: boolean;
  readonly value: number;
}

/**
 * Format a concise human-readable cell equation for Elastic-MoT attention inspection.
 */
export function formatUniAttentionCellFormula(
  params: UniAttentionCellFormulaParams,
): string {
  const {
    stage,
    qHead,
    kvHead,
    qRow,
    kCol = 0,
    outDimIndex = 0,
    headDim,
    branch,
    splitTower,
    isMasked = false,
    value,
  } = params;

  const towerTag = splitTower ? 'split' : 'joint';

  if (stage === 'score') {
    if (isMasked) {
      return `score[${branch}:${towerTag} h${qHead}→kv${kvHead}, q=${qRow}, k=${kCol}] = −∞ (masked)`;
    }
    return `score[${branch}:${towerTag} h${qHead}→kv${kvHead}, q=${qRow}, k=${kCol}] = ⟨Q_scaled, K_rot⟩ / √${headDim} = ${fmtSigned(value)}`;
  }

  if (stage === 'prob') {
    if (isMasked) {
      return `prob[${branch}:${towerTag} h${qHead}→kv${kvHead}, q=${qRow}, k=${kCol}] = 0.000 (masked)`;
    }
    return `prob[${branch}:${towerTag} h${qHead}→kv${kvHead}, q=${qRow}, k=${kCol}] = softmax(score) = ${fmtSigned(value, 4)}`;
  }

  const projName = branch === 'G' ? 'W_O^G' : 'W_O^U';
  return `o_proj[${branch}:${towerTag} t=${qRow}, d=${outDimIndex}] = ${projName}·a = ${fmtSigned(value)}`;
}
